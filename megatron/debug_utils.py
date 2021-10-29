import torch
from packaging import version


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
