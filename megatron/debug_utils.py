
import torch.distributed as dist
import torch
import os
import socket
import fcntl

def printflock(*msgs):
    """ print """
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            print(*msgs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def get_fingerprint_header():
    return f"{'min':^13} {'max':^13} {'mean':^13} {'l2 norm':^12} metadata"


def get_fingerprint(p):
    return f"{p.min():13.6e} {p.max():13.6e} {p.mean():13.6e} {p.norm():12.6e}"


def dump_weights(preamble, iteration, model, optimizer, tensor=None):
    return

    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    dp_rank = mpu.get_data_parallel_rank()
    dp_size = mpu.get_data_parallel_world_size()
    fn = f"debug-bf16-{iteration}-pp{pp_rank}-tp{tp_rank}-dp{dp_rank}-{preamble}.txt"

    # only care for first and last pp stages and dp0 tp0
    if not (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()):
        return

    if not (tp_rank == 0 and dp_rank == 0):
        return

    if tensor is not None:
        orig_tensor = tensor
        if hasattr(tensor, "_hp_param"):
            numel = tensor._hp_param.numel() # // dp_size
            tensor = tensor.flatten().narrow(0, 0, numel)

    #print(fn)
    with open(fn, "w") as fh:
        fh.write(f"{get_fingerprint_header()}\n")

        if tensor is not None:
            fh.write(f"{get_fingerprint(tensor)} tensor {tensor.shape}\n")
        else:
            for n, p in model[0].named_parameters():
                fh.write(f"{get_fingerprint(p)} {n} {p.shape}\n")

    # until we figure out how to dump the actual fp32 values don't do this
    fn = f"debug-fp32-{iteration}-pp{pp_rank}-tp{tp_rank}-dp{dp_rank}-{preamble}.txt"
    with open(fn, "w") as fh:
        fh.write(f"{get_fingerprint_header()}\n")
        if tensor is not None:
            tensor = orig_tensor
            if hasattr(tensor, "_hp_param"):
                fh.write(f"{get_fingerprint(tensor._hp_param)} tensor {tensor._hp_param.shape}\n")
                fh.write(f"{get_fingerprint(tensor._hp_grad)} tensor grad\n")
            else:
                fh.write(f"{get_fingerprint(tensor)} tensor {tensor.shape}\n")
                fh.write(f"{get_fingerprint(tensor.grad)} tensor grad\n")

        else:
            if hasattr(model[0].module.tied_modules, "embed"):
                p = model[0].module.tied_modules.embed.word_embeddings.weight._hp_param
                fh.write(f"{get_fingerprint(p)} module.tied_modules.embed.word_embeddings.weight._hp_param {p.shape}\n")

        # for i, param_group in enumerate(optimizer.param_groups):
        #     fh.write(f"{get_fingerprint(optimizer.fp32_groups_flat_partition[i])} group={i}\n")
            #fh.write(f"{i}={optimizer.fp32_groups_flat_partition[i]}\n")
    #     if mpu.is_pipeline_first_stage():
    #         x = optimizer.fp32_groups_flat_partition[0]
    #         fh.write(f"fp32={x[:402432]}\n")
    #     if mpu.is_pipeline_last_stage()):
    #         x = optimizer.fp32_groups_flat_partition[1]
    #         fh.write(f"fp32={x[-402432:]}\n")

    # import os
    # import socket
    # hostname = socket.gethostname()
    # pid = os.getpid()
    # global_rank = torch.distributed.get_rank()
    #fn = f"debug-{iteration}-pp{pp_rank}-tp{tp_rank}-dp{dp_rank}-global{global_rank}-{preamble}-{pid}.txt"




# compare before
# perl -le 'print qx[diff -u debug-$_-pp0-tp0-dp0-global0-before-iteration.txt debug-$_-pp1-tp0-dp0-global1-before-iteration.txt] for 301..320'
# compare after
# perl -le 'print qx[diff -u debug-$_-pp0-tp0-dp0-global0-after-iteration.txt debug-$_-pp1-tp0-dp0-global1-after-iteration.txt] for 301..320'


import torch

def dump_emb(preamble, iteration, model):
    return

    # torch.set_printoptions(
    #     threshold=10000000000, # print all data (without ... skipping) - can be huge!
    #     sci_mode=False,      # print all data on the same scale of 1 (this disables scientific notation)
    #     precision=6,         # print X decimal points for floats (default 4)
    # )

    # only care for first and last pp stages and dp0 tp0
    if not (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()):
        return

    #printflock(f"pp rank={pp_rank} {preamble} {model[0].module.tied_modules.embed.word_embeddings.weight}")

    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    dp_rank = mpu.get_data_parallel_rank()
    #global_rank = torch.distributed.get_rank()

    if not (tp_rank == 0 and dp_rank == 0):
        return

    # fn = f"debug-emb-bf16-{iteration}-pp{pp_rank}-tp{tp_rank}-dp{dp_rank}-{preamble}.zip"
    # torch.save(model[0].module.tied_modules.embed.word_embeddings.weight, fn)

    fn = f"debug-emb-bf16-{iteration}-pp{pp_rank}-tp{tp_rank}-dp{dp_rank}-{preamble}.txt"
    #fn = f"debug-{iteration}-pp{pp_rank}-tp{tp_rank}-dp{dp_rank}-global{global_rank}-{preamble}.txt"
    #print(fn)
    with open(fn, "w") as fh:
        fh.write(f"module.tied_modules.embed.word_embeddings.weight={model[0].module.tied_modules.embed.word_embeddings.weight.cpu()}\n")
        # if pp_rank == 0:
        #     fh.write(f"module.tied_modules.embed.word_embeddings.norm.weight={model[0].module.tied_modules.embed.word_embeddings.norm.weight.cpu()}\n")
        #     fh.write(f"module.tied_modules.embed.word_embeddings.norm.bias={model[0].module.tied_modules.embed.word_embeddings.norm.bias.cpu()}\n")
