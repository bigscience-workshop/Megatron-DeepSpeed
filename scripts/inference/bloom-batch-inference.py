import torch
import deepspeed
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from argparse import ArgumentParser
import os
from transformers.models.bloom.modeling_bloom import BloomBlock as BloomBlock
from transformers.models.bloom.modeling_bloom import BloomPreTrainedModel
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import BloomTokenizerFast


def get_checkpoint_files(pretrained_model_name_or_path):
    # XXX: I just hacked this one together to automatically handle the fetching of the model file or
    # shards into cache and returning the cached entries - note that I removed most arguments

    from transformers.utils import WEIGHTS_NAME, WEIGHTS_INDEX_NAME, cached_path, hf_bucket_url, is_offline_mode
    from transformers.utils.hub import EntryNotFoundError
    from transformers.modeling_utils import get_checkpoint_shard_files

    cache_dir = None
    is_sharded = False

    # XXX: preparation for revision branches if needed
    revision = None
    #revision = "sharded"

    # this supports nodes with no network (so you need to pre-cache the model and the tokenizer with
    # python -c "from transformers import AutoModel; AutoModel.from_pretrained('bigscience/bloom')"
    if is_offline_mode():
        print("Offline mode: forcing local_files_only=True")
        local_files_only = True
    else:
        local_files_only = False

    filename = WEIGHTS_NAME
    archive_file = hf_bucket_url(pretrained_model_name_or_path, filename=filename, revision=revision)

    try:
        resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir, local_files_only=local_files_only,)
        return [resolved_archive_file]

    except (EntryNotFoundError, FileNotFoundError):
        if filename == WEIGHTS_NAME:
            # Maybe the checkpoint is sharded, we try to grab the index name in this case.
            archive_file = hf_bucket_url(
                pretrained_model_name_or_path,
                filename=WEIGHTS_INDEX_NAME,
                revision=revision,
            )
            resolved_archive_file = cached_path(
                archive_file,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            is_sharded = True

    if is_sharded:
        # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            resolved_archive_file,
            cache_dir=cache_dir,
            revision=revision
        )

        return resolved_archive_file

parser = ArgumentParser()
parser.add_argument("--name", required=True, type=str)
parser.add_argument("--local_rank", required=False, type=int)
args = parser.parse_args()
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

print(
    "***************** Creating model in RANK ({0}) with WORLD_SIZE = {1} *****************"
    .format(local_rank,
            world_size))
tokenizer = BloomTokenizerFast.from_pretrained(args.name)
config = AutoConfig.from_pretrained(args.name)
model_hidden_size = config.hidden_size
train_batch_size = 1 * world_size
ds_config = {
    "fp16": {
        "enabled": True
    },
    "bf16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
        "stage3_param_persistence_threshold": 0
    },
    "steps_per_print": 2000,
    "train_batch_size": train_batch_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
}
dschf = HfDeepSpeedConfig(ds_config)
model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16).eval()

from deepspeed.runtime.utils import see_memory_usage
see_memory_usage("after model load ", force=True)
ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
see_memory_usage("after zero-init ", force=True)
ds_engine.module.eval()  # inference
model = ds_engine.module

ds_engine.destroy()
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
def ds_clear_params(ds_engine):
    for name, p in ds_engine.named_parameters():
        if hasattr(p, "ds_tensor"):
            p.ds_tensor = torch.empty(0, dtype=p.dtype, device=p.device)
            p.ds_status = ZeroParamStatus.NOT_AVAILABLE
# this frees the memory used by zero
ds_clear_params(ds_engine)
del ds_engine

checkpoints_json = "checkpoints.json"
def write_checkponts_json():
    model_name = args.name
    with io.open(checkpoints_json, 'w', encoding='utf-8') as f:
        checkpoint_files = get_checkpoint_files(model_name)

        print("Checkpoint files:", checkpoint_files)

        data = {
            "type": "BLOOM-176B",
            "checkpoints": checkpoint_files,
            "version": 1.0
        }
        json.dump(data, f)

model = deepspeed.init_inference(model, 
                                 mp_size=world_size,
                                 dtype=torch.half,
                                 checkpoint=checkpoints_json,
                                 replace_with_kernel_inject=True,
                                 )
model = model.module
input_sentence = ["DeepSpeed is", 
                  "He is working on", 
                  "He has a", 
                  "He got all", 
                  "Everyone is happy and I can", 
                  "The new movie that got Oscar this year", 
                  "In the far far distance from our galaxy,", 
                  "Peace is the only way"]

print("inference-engine created \n")

tokenizer = BloomTokenizerFast.from_pretrained(args.name, padding_side="left")
tokens = tokenizer.batch_encode_plus(input_sentence, return_tensors="pt", padding=True)

for t in tokens:
    if torch.is_tensor(tokens[t]):
        tokens[t] = tokens[t].to(torch.cuda.current_device())

greedy_output = model.generate(
    **tokens, max_length=100, do_sample=True
)

for i in range(len(greedy_output)):
    out = tokenizer.decode(greedy_output[i], skip_special_tokens=True)
    if torch.distributed.get_rank() == 0:
        print(out)
        print