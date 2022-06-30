import torch
from deepspeed import PipelineModule
from deepspeed.runtime.pipe import TiedLayerSpec, LayerSpec
from torch.nn import LayerNorm

from megatron.enums import AttnMaskType, LayerType

from megatron.model.transformer import ParallelTransformerLayerPipe

from megatron.model.language_model import EmbeddingPipe, parallel_lm_logits

from megatron.model.utils import init_method_normal, scaled_init_method_normal

from megatron import get_args, mpu

from megatron.model.module import MegatronModule, fp32_to_16bit, float16_to_fp32

def cross_entropy(output, labels):
    labels, loss_mask = labels[0], labels[1]

    losses = mpu.vocab_parallel_cross_entropy(output.contiguous().float(), labels)

    expected_number_of_tokens = loss_mask.sum()

    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / expected_number_of_tokens
    return loss

class SharedT5ModelPipe(PipelineModule, MegatronModule):
    """Share encoder decoder language model."""

    def __init__(
        self,
        num_tokentypes=0,
        parallel_output=True,
    ):
        args = get_args()
        self.parallel_output = parallel_output

        init_method = init_method_normal(args.init_method_std)

        self.specs = []

        def _to_16bit(inputs):
            if args.fp16:
                return fp32_to_16bit(inputs, lambda v: v.half())
            elif args.bf16:
                return fp32_to_16bit(inputs, lambda v: v.bfloat16())
            else:
                return inputs

        self.specs.append(lambda inputss: tuple(_to_16bit(inputs) for inputs in inputss))

        # Embedding layer
        self.specs.append(TiedLayerSpec('embed',
                                        EmbeddingPipe,
                                        args.hidden_size,
                                        args.padded_vocab_size,
                                        args.hidden_dropout,
                                        forward_fn=lambda module, input_and_target: (module(input_and_target[:3]), module(input_and_target[3:])),
                                        init_method=init_method,
                                        num_tokentypes=num_tokentypes,
                                        tied_weight_attr='word_embeddings_weight'))

        assert hasattr(args, 'attn_mask'), "Deepspeed integration should have attention mask s"
        # Drop everything beside tokens
        # self.specs.append(lambda inputs, targets: (inputs[0], targets[0]))
        if args.fp32_residual_connection:
            self.specs.append(lambda input_and_target: (input_and_target[0].transpose(0, 1).contiguous().float(), input_and_target[1].transpose(0, 1).contiguous().float()))
        else:
            self.specs.append(lambda input_and_target: (input_and_target[0].transpose(0, 1).contiguous(), input_and_target[1].transpose(0, 1).contiguous()))

        ### -----  Encoder -----
        for layer_idx in range(args.num_layers):
            self.specs.append(
                TiedLayerSpec(
                    f"block_{layer_idx}",
                    ParallelTransformerLayerPipe,
                    init_method=init_method,
                    forward_fn=lambda module, input_and_target: (module(input_and_target[0]), input_and_target[1]),
                    output_layer_init_method=scaled_init_method_normal(args.init_method_std,
                                                                       args.num_layers),
                    layer_type=LayerType.encoder,
                    layer_number=layer_idx,
                    self_attn_mask_type=AttnMaskType.causal,
                    tied_weight_attr=None,
                    tied_weight_attrs=["self_attention", "mlp"]
                ))

        # Final layernorm after encoder layers
        self.specs.append(
            LayerSpec(
                LayerNorm,
                args.hidden_size,
                forward_fn=lambda module, input_and_target: (module(input_and_target[0]), input_and_target[1]),
                eps=args.layernorm_epsilon
            ))

        # Decoder
        for layer_idx in range(args.num_layers):
            self.specs.append(
                TiedLayerSpec(
                    f"block_{layer_idx}",
                    ParallelTransformerLayerPipe,
                    init_method=init_method,
                    forward_fn=lambda module, encoded_and_target: (encoded_and_target[0], module(encoded_and_target[1], encoder_output=encoded_and_target[0])),
                    output_layer_init_method=scaled_init_method_normal(args.init_method_std,
                                                                       args.num_layers),
                    layer_number=layer_idx,
                    layer_type=LayerType.decoder,
                    self_attn_mask_type=AttnMaskType.padding,
                    tied_weight_attr=None,
                    tied_weight_attrs=["self_attention", "mlp"]
                )
            )

        # Drop encoded tokens
        self.specs.append(lambda encoded_and_target: encoded_and_target[1])

        # Final layernorm after decoder layers
        self.specs.append(
            LayerSpec(
                LayerNorm,
                args.hidden_size,
                eps=args.layernorm_epsilon
            ))

        # Undo data format change
        self.specs.append(lambda x: x.transpose(0, 1).contiguous())

        def _logits_helper(embedding, lm_output):
            """A wrapper to massage inputs/outputs from pipeline. """
            return parallel_lm_logits(
                lm_output,
                embedding.word_embeddings_weight,
                self.parallel_output)

        self.specs.append(
            TiedLayerSpec('embed',
                          EmbeddingPipe,
                          args.hidden_size,
                          args.padded_vocab_size,
                          args.hidden_dropout,
                          init_method=init_method,
                          num_tokentypes=num_tokentypes,
                          forward_fn=_logits_helper,
                          tied_weight_attr='word_embeddings_weight')
        )

        # Convert to fp32 if needed
        if args.fp16 or args.bf16:
            self.specs.append(float16_to_fp32)

        if args.checkpoint_activations:
            interval = args.checkpoint_num_layers
        else:
            interval = 0

        from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
        topo = PipeModelDataParallelTopology(num_pp=mpu.get_pipeline_model_parallel_world_size(),
                                             num_mp=mpu.get_tensor_model_parallel_world_size(),
                                             num_dp=mpu.get_data_parallel_world_size())

        # here one can extend the regex to include more layers to be counted towards partitioning,
        # e.g. 'type:transformer|embedding' will add up all the transformer blocks and also the first
        # and last embedding layers and then partition that transformers+2 layers - so to get a good
        # balance you may want to use less transformer layers
        #
        # caveat emptor: the current implementation of PP fails unless each stage has at least one
        # transformer layer
        if args.pp_partition_method is not None:
            partition_method = args.pp_partition_method
        else:
            partition_method = 'type:transformer'

        super().__init__(layers=self.specs,
                         loss_fn=cross_entropy,
                         topology=topo,
                         activation_checkpoint_interval=interval,
                         partition_method=partition_method)
