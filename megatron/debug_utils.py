import torch
import sys
import collections

from packaging import version

from torch.utils.tensorboard import SummaryWriter

class ModuleInspector:
    """

    Args:
        args (:obj:`dict`):
            Megatron args object
        model (:obj:`nn.Module`):
            The model to debug.
    """

    def __init__(self, rank, model, args):
        self.rank = rank # pp_rank usually, or any other unique id
        self.args = args
        self.model = model
        self.tb_path = args.tensorboard_debug_dir
        self.rank = args.rank
        self.tbs = {}

        # XXX: possible additions
        # - iteration-interval: don't log on each iteration to have less data
        # - log-on-threshold: log only when values are larger than certain value - e.g. 75% of 64k (max fp16 number before overflow)
        # - the caller could also skip logging on all layers/pipe stages

        self.analyse_model()

        self.register_forward_hook()
        self.register_backward_hook()

    def analyse_model(self):
        # extract the fully qualified module names, to be able to report at run time. e.g.:
        # encoder.block.2.layer.0.SelfAttention.o
        #
        # for shared weights only the first shared module name will be registered
        self.module_names = {m: name for name, m in self.model.named_modules()}

    def register_forward_hook(self):
        self.model.apply(self._register_forward_hook)

    def register_backward_hook(self):
        self.model.apply(self._register_backward_hook)

    def _register_forward_hook(self, module):
        module.register_forward_hook(self.forward_hook)

    def _register_backward_hook(self, module):
        # XXX: register_full_backward_hook leads to a huge leak
        # https://github.com/microsoft/DeepSpeed/issues/1572 so for now will use the deprecated
        # function:
        module.register_backward_hook(self.backward_hook)
        # if version.parse(torch.__version__) >= version.parse("1.10"):
        #     module.register_full_backward_hook(self.backward_hook)
        # else:
        #     module.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        # - input is a tuple of packed inputs (could be non-Tensors)
        # - output could be a Tensor or a tuple of Tensors and non-Tensors
        self.process(module, input, output, direction='forward')

    def backward_hook(self, module, input, output):
        self.process(module, input, output, direction='backward')

    def process(self, module, input, output, direction):
        iteration = self.args.iteration

        tb_dir = f"{self.tb_path}/{self.rank:0>3d}-{self.module_names[module]}"

        if tb_dir not in self.tbs:
            print(f"Allocating new writer for {tb_dir}")
            self.tbs[tb_dir] = SummaryWriter(log_dir=tb_dir, max_queue=5)

        tb = self.tbs[tb_dir]

        def log_to_tb(var, name):
            if not (torch.is_tensor(var) and var.is_floating_point() and var is not None):
                return
            # need to prefix with numbers to get TB to sort graphs the way we want
            prefix = "0.fwd" if direction == 'forward' else "1.bwd"
            prefix += f"/{name}"
            abs_var = var.abs()
            tb.add_scalar(f"{prefix}/0.abs_min", abs_var.min(), iteration)
            tb.add_scalar(f"{prefix}/1.abs_argmin", abs_var.argmin(), iteration)
            tb.add_scalar(f"{prefix}/2.abs_max", abs_var.max(), iteration)
            tb.add_scalar(f"{prefix}/3.abs_argmax", abs_var.argmax(), iteration)
            tb.add_scalar(f"{prefix}/4.norm", torch.linalg.norm(var.data), iteration)

        # should be the same for fwd/bwd, so log only one of them
        if direction == "forward":
            for name, p in module.named_parameters(recurse=False):
                log_to_tb(p, name)

        # inputs
        if isinstance(input, tuple):
            for i, x in enumerate(input):
                log_to_tb(x, f"input[{i}]")
        else:
            log_to_tb(input, "input")

        # outputs
        if isinstance(output, tuple):
            for i, x in enumerate(output):
                # possibly a tuple of tuples
                if isinstance(x, tuple):
                    for j, y in enumerate(x):
                        log_to_tb(y, f"output[{i}][{j}]")
                else:
                    log_to_tb(x, f"output[{i}]")
        else:
            log_to_tb(output, "output")


class DebugGradientNorm:
    def __init__(self, model, layers=[]):
        self.model = model
        self.total_calls = 0
        self.prefix = "                 "
        self.analyze_model(layers)

    def analyze_model(self, layers):
        if layers:
            try:
                self.module_names = {self.model[layer]: layer for layer in layers}
            except TypeError:
                self.module_names = {getattr(self.model, layer): layer for layer in layers}
        else:
            self.module_names = {module: name for name, module in self.model.named_modules()}

    def backward_hook(self, module, grad_inputs, grad_outputs):
        if self.total_calls == 0:
            self.print_header()
        self.total_calls += 1
        min_grad_norm = min(
            grad_output.abs().min().item() for grad_output in grad_outputs if grad_output is not None
        )
        max_grad_norm = max(
            grad_output.abs().max().item() for grad_output in grad_outputs if grad_output is not None
        )
        total_norm = sum(
            grad_output.norm() ** 2 for grad_output in grad_outputs if grad_output is not None
        ).sqrt().item()
        module_name = self.module_names[module] or "full model"
        message = (
            f"{min_grad_norm:8.2e} {max_grad_norm:8.2e} {total_norm:8.2e} {module_name}"
        )
        print(message)

    def print_header(self):
        print(f"\n\n{self.prefix} *** Starting gradient logging ***")
        print(f"{'abs min':8} {'abs max':8} {'l2 norm':8} metadata")

    def register_backward_hook(self):
        self.model.apply(self._register_backward_hook)

    def _register_backward_hook(self, module):
        if version.parse(torch.__version__) >= version.parse("1.10"):
            module.register_full_backward_hook(self.backward_hook)
        else:
            module.register_backward_hook(self.backward_hook)
