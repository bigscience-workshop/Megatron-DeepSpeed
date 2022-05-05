import json
import os
import sys
from unittest.mock import patch
from parameterized import parameterized

import deepspeed
import torch

torch.set_printoptions(precision=8)

from transformers import GPT2LMHeadModel, AutoTokenizer, GPTMegLMHeadModel

from megatron import get_args, get_tokenizer, global_vars, initialize_megatron
from megatron.checkpointing import load_checkpoint, save_checkpoint
from megatron.testing_utils import (
    TestCasePlus,
    mockenv_context,
    require_deepspeed,
    require_torch_gpu,
    set_seed,
    torch_assert_close,
    torch_assert_equal,
)
from megatron.training import setup_model_and_optimizer
from pretrain_gpt import get_batch_pipe as get_gpt_batch_pipe
from pretrain_gpt import model_provider as gpt_model_provider
from test_model import flatten_arguments, get_default_args
from tools.convert_checkpoint.deepspeed_to_megatron import main as megatron_ds_to_megatron_main
from tools.convert_checkpoint.deepspeed_to_transformers import main as megatron_ds_to_transformers_main


def reset_global_variables():
    global_vars._GLOBAL_ARGS = None
    global_vars._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
    global_vars._GLOBAL_TOKENIZER = None
    global_vars._GLOBAL_TENSORBOARD_WRITER = None
    global_vars._GLOBAL_ADLR_AUTORESUME = None
    global_vars._GLOBAL_TIMERS = None



def apply_overrides():

    # 1. layer norm needs to be done in fp32 and then cast back to fp16 to match meg.
    torch_layer_norm_orig = torch.layer_norm
    def torch_layer_norm_force_fp32(input, normalized_shape, weight, bias, eps, cuddn):
        out = torch_layer_norm_orig(input.float(), normalized_shape, weight.float(), bias.float(), eps, torch.backends.cudnn.enabled).half()
        print(out)
        #die
        return out
    torch.layer_norm = torch_layer_norm_force_fp32


    # 2. MLP uses a slightly different activation function with a custom bwd
    import transformers.activations
    @torch.jit.script
    def gelu_megatron_fwd(x):
        return  x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

    @torch.jit.script
    def gelu_megatron_bwd(g, x):
        tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
        # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
        ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
        return ff*g

    class GeLUFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return gelu_megatron_fwd(input)

        @staticmethod
        def backward(ctx, grad_output):
            input = ctx.saved_tensors
            tmp = gelu_megatron_bwd(grad_output, input)
            return tmp, tmp

    transformers.activations.gelu_fast = GeLUFunction.apply
    transformers.activations.ACT2FN["gelu_fast"] = transformers.activations.gelu_fast


    # 3. torch.baddbmm() (meg) produces slightly different results than torch.matmul, so override to use `torch.baddbmm`
    import transformers.models.gpt2.modeling_gpt2
    from torch import nn
    def new_attn(self, query, key, value, attention_mask=None, head_mask=None):
        output_size = (query.size(0), key.size(1), query.size(2), key.size(2))
        matmul_result = torch.empty(output_size[0] * output_size[1], output_size[2], output_size[3], dtype=query.dtype, device=query.device)

        factor = float(value.size(-1)) ** 0.5
        matmul_result = torch.baddbmm(
            matmul_result,
            query.reshape(-1, query.shape[2], query.shape[3]),  # [b * np, sq, hn]
            key.reshape(-1, query.shape[2], query.shape[3]).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=1.0 / factor
        )
        attn_weights = matmul_result.view(*output_size)

        # attn_weights = torch.matmul(query, key.transpose(-1, -2))
        #
        # if self.scale_attn_weights:
        #     attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    transformers.models.gpt2.modeling_gpt2.GPT2Attention._attn = new_attn


@require_deepspeed
@require_torch_gpu
class TestCheckpointConversion(TestCasePlus):
    def setUp(self) -> None:
        super().setUp()
        set_seed()
        reset_global_variables()

        # set this for the test's duration only and unset when done
        self.cudnn_deterministic = torch.backends.cudnn.deterministic
        torch.backends.cudnn.deterministic = True

        self.dist_env_1_gpu = dict(
            MASTER_ADDR="localhost",
            MASTER_PORT=str(os.getpid())[-5:], #"1123",
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        )

    def tearDown(self):
        # restore state
        torch.backends.cudnn.deterministic = self.cudnn_deterministic
        super().tearDown()


    def get_megatron_ds_args(self, fp16):
        # get megatron and deepspeed args
        megatron_args = get_default_args()

        # reset the number of layers
        # megatron_args['--num-layers'] = '50'

        if fp16:
            ds_config_path = f"{self.test_file_dir_str}/ds_config.json"
        else:
            megatron_args.pop("--fp16")
            ds_config_path = f"{self.test_file_dir_str}/ds_config_fp32.json"

        megatron_args.pop("--checkpoint-activations")
        # megatron_args["--tokenizer-type"] = "GPT2BPETokenizer"
        # megatron_args["--vocab-file"] = "xxx/vocab.json"
        # megatron_args["--merge-file"] = "xxx/merges.txt"

        ds_args = f"""
            --deepspeed
            --deepspeed_config {ds_config_path}
            --zero-stage 1
        """.split()
#            --deepspeed-activation-checkpointing

        return megatron_args, ds_args

    def save_and_get_megatron_ds_output(
        self, args, megatron_ds_model, megatron_ds_token, megatron_ds_dir
    ):
        # args for checkpointing
        args.save, args.no_save_optim = megatron_ds_dir, True
        # save deepspeed megatron model
        save_checkpoint(1, megatron_ds_model, None, None)

        megatron_ds_model = megatron_ds_model[0]

        # get processed batch
        _, _, attention_mask = megatron_ds_token

        # disable activation checkpoint
        megatron_ds_model.module.activation_checkpoint_interval = 0
        # resemble _exec_schedule in /deepspeed/runtine/pipe/engine.py
        megatron_ds_model._compute_loss = False
        megatron_ds_model.fwd_outputs = []

        megatron_ds_model.eval()
        # due to the hack in https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/68b46f201aad8803d0699f76b100663553e89e1b/pretrain_gpt.py#L74
        args.attn_mask = attention_mask

        # use deepspeed pipeline thing
        megatron_ds_model.pipe_buffers["inputs"].append(megatron_ds_token)
        megatron_ds_model.pipe_buffers["outputs"].append(None)

        with torch.no_grad():
            megatron_ds_model._exec_forward_pass(buffer_id=0)

        # gather output
        megatron_ds_output = megatron_ds_model.pipe_buffers["outputs"][0]
        return megatron_ds_output


    @parameterized.expand(["fp16", "fp32"])
    def test_megatron_ds_to_megatron(self, name):
        # 1. convert megatron-deepspeed to megatron
        # 2. convert megatron to transformers
        # 3. compare the difference
        fp16 = True if name == "fp16" else False

        megatron_args, ds_args = self.get_megatron_ds_args(fp16)
        # run deepspeed megatron
        with patch("sys.argv", flatten_arguments(megatron_args) + ds_args):
            with mockenv_context(**self.dist_env_1_gpu):
                deepspeed.init_distributed()
                initialize_megatron()
                args = get_args()

                # get deepspeed megatron model
                megatron_ds_model, _, _ = setup_model_and_optimizer(gpt_model_provider)

                megatron_tokenizer = get_tokenizer()
                #print(megatron_tokenizer)
                text = "This is a big test"
                token_ids = torch.tensor([megatron_tokenizer.tokenize(text)])
                print("TOKENS", token_ids)
                #die

                print('EOD', megatron_tokenizer.eod)

                # # always use this token_ids
                # token_ids = torch.randint(
                #     args.padded_vocab_size, (args.micro_batch_size, args.seq_length)
                # )
                # # process batch
                # token_ids[token_ids == megatron_tokenizer.eod] += 1
                # token_ids[token_ids == megatron_tokenizer.eod] %= args.padded_vocab_size
                # print(token_ids)

                # get processed batch
                megatron_token, _ = get_gpt_batch_pipe({"text": token_ids})

                # save directory
                megatron_ds_dir = self.get_auto_remove_tmp_dir()

                # gather deepspeed model output
                megatron_ds_output = self.save_and_get_megatron_ds_output(
                    args, megatron_ds_model, megatron_token, megatron_ds_dir
                )

        # run conversion file
        megatron_dir = self.get_auto_remove_tmp_dir()
        conversion_args = {
            "--input_folder": f"{megatron_ds_dir}",
            "--output_folder": f"{megatron_dir}",
        }
        with patch("sys.argv", flatten_arguments(conversion_args)):
            megatron_ds_to_megatron_main()

        # run megatron
        with patch("sys.argv", flatten_arguments(megatron_args)):
            with mockenv_context(**self.dist_env_1_gpu):
                args.deepspeed = False
                megatron_model, _, _ = setup_model_and_optimizer(gpt_model_provider)

                args.load, args.no_load_rng = megatron_dir, True
                load_checkpoint(megatron_model, None, None)

                megatron_model = megatron_model[0]

                megatron_model.eval()
                megatron_output = megatron_model(*megatron_token)

        # compare the difference
        # torch_assert_equal(megatron_ds_output.data.cpu(), megatron_output.data.cpu())
        torch_assert_close(megatron_ds_output.data.cpu(), megatron_output.data.cpu())


    @parameterized.expand(["fp16", "fp32"])
    def test_megatron_ds_to_transformers(self, name):
        # 1. convert megatron-deepspeed to transformers
        # 2. compare the difference
        fp16 = True if name == "fp16" else False

        megatron_args, ds_args = self.get_megatron_ds_args(fp16)

        # run deepspeed megatron
        with patch("sys.argv", flatten_arguments(megatron_args) + ds_args):
            with mockenv_context(**self.dist_env_1_gpu):
                deepspeed.init_distributed()
                initialize_megatron()
                args = get_args()
                #print(args)

                # get deepspeed megatron model
                megatron_ds_model, _, _ = setup_model_and_optimizer(gpt_model_provider)

                megatron_tokenizer = get_tokenizer()
                text = "This is a long drill" # fails at fp32
                #text = "This is a very long drill" # succeeds at fp32
                #text = "I know what I want and I want it now, I want you cause I'm Mr. Vain"
                #text = "This"
#                 text = """
# No, you can't always get what you want
# You can't always get what you want
# But if you try sometime you find
# You get what you need
#                 """
                megatron_token_ids = torch.tensor([megatron_tokenizer.tokenize(text) + [megatron_tokenizer.eod]])
                print("TOKENS", megatron_token_ids)

                # # always use this token_ids
                # token_ids = torch.randint(
                #     args.padded_vocab_size, (args.micro_batch_size, args.seq_length)
                # )
                # # process batch
                # token_ids[token_ids == megatron_tokenizer.eod] += 1
                # token_ids[token_ids == megatron_tokenizer.eod] %= args.padded_vocab_size

                # get processed batch
                megatron_token, _ = get_gpt_batch_pipe({"text": megatron_token_ids})

                # save directory
                megatron_ds_dir = self.get_auto_remove_tmp_dir()

                # gather deepspeed model output
                megatron_ds_output = self.save_and_get_megatron_ds_output(
                    args, megatron_ds_model, megatron_token, megatron_ds_dir
                )

        # run conversion file
        megatron_dir = self.get_auto_remove_tmp_dir()
        conversion_args = {
            "--input_folder": f"{megatron_ds_dir}",
            "--output_folder": f"{megatron_dir}",
        }
        with patch("sys.argv", flatten_arguments(conversion_args)):
            megatron_ds_to_megatron_main()

        # run megatron
        with patch("sys.argv", flatten_arguments(megatron_args)):
            with mockenv_context(**self.dist_env_1_gpu):
                args.deepspeed = False
                megatron_model, _, _ = setup_model_and_optimizer(gpt_model_provider)

                args.load, args.no_load_rng = megatron_dir, True
                load_checkpoint(megatron_model, None, None)

                megatron_model = megatron_model[0]

                megatron_model.eval()
                megatron_output = megatron_model(*megatron_token)

        megatron_ds_output = megatron_output

        # run conversion file
        transformer_dir = self.get_auto_remove_tmp_dir() # "./xxx") # XXX: fix
        conversion_args = {
            "--input_folder": f"{megatron_ds_dir}",
            "--output_folder": f"{transformer_dir}",
        }
        with patch("sys.argv", flatten_arguments(conversion_args)):
            megatron_ds_to_transformers_main()

        #if fp16:
        #    apply_overrides()

        torch_dtype = torch.float16 if fp16 else torch.get_default_dtype()
        transformers_model = GPTMegLMHeadModel.from_pretrained(transformer_dir, torch_dtype=torch_dtype)
        #transformers_model = GPT2LMHeadModel.from_pretrained(transformer_dir, torch_dtype=torch_dtype)
        transformers_model.cuda()
        transformers_model.eval()
        transformers_tokenizer = AutoTokenizer.from_pretrained(transformer_dir)
        inputs = transformers_tokenizer(text, return_tensors="pt").to("cuda");
        print("TOKENS", inputs)
        transformers_token_ids = inputs.input_ids.clone()

        # test tokenization
        torch_assert_equal(megatron_token_ids[:,:-1].cpu(), transformers_token_ids.cpu(), check_stride=False)

        if fp16:
            megatron_logits = megatron_ds_output.data.cpu().half()
            transformers_output = transformers_model(**inputs).logits #.half()
        else:
            megatron_logits = megatron_ds_output.data.cpu()
            transformers_output = transformers_model(**inputs).logits

        transformers_logits = transformers_output.data.cpu() #[:, :-1, :]

        #torch.set_printoptions(precision=10)

        print(f"DS SHAPE: {megatron_logits.shape}")
        print(f"HF SHAPE: {transformers_logits.shape}")
        print(f"DS DTYPE: {megatron_logits.dtype}")
        print(f"HF DTYPE: {transformers_logits.dtype}")
        print(f" DS: {megatron_logits}")
        print(f" HF: {transformers_logits}")

        # compare the difference
        # FIXME: it could sometimes now pass torch_assert_close, but not torch_assert_equal
        # torch_assert_equal(megatron_ds_output.data.cpu(), transformers_output.data.cpu())
        torch_assert_close(megatron_logits, transformers_logits, check_stride=False)#, rtol=1e-05, atol=1e-08)
