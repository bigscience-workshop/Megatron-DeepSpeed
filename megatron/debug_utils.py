class DebugGradientNorm:
    def __init__(self, model, layers=[]):
        self.model = model
        if layers:
            try:
                self.module2key = {model[layer]: layer for layer in layers}
            except TypeError:
                self.module2key = {getattr(model, layer): layer for layer in layers}
        else:
            self.module2key = {module: name for name, module in model.named_parameters()}

    def backward_hook(self, module, grad_inputs, grad_outputs):
        total_norm = sum(grad_output.norm() ** 2 for grad_output in grad_outputs)
        total_norm_squared = total_norm.sqrt().item()
        min_grad_norm = grad_inputs.min().item()
        max_grad_norm = grad_inputs.max().item()
        module_name = self.module2key[module]
        print(f"{module_name}: total {total_norm_squared}, min {min_grad_norm}, max {max_grad_norm}")

    def register_backward_hook(self):
        self.model.apply(self._register_backward_hook)

    def _register_backward_hook(self, module):
        if hasattr(module, "register_full_backward_hook"):
            module.register_full_backward_hook(self.backward_hook)
        else:
            module.register_backward_hook(self.backward_hook)
