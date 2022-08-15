class ActivationsAndGradients:

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = []
        self.input_activations = []
        self.output_activations = []

        target_layer.register_forward_hook(self.save_activation)

        #Backward compitability with older pytorch versions:
        if hasattr(target_layer, 'register_full_backward_hook'):
            target_layer.register_full_backward_hook(self.save_gradient)
        else:
            target_layer.register_backward_hook(self.save_gradient)

    # forward hook function
    def save_activation(self, module, input, output):
        input_activation = input[0].F
        output_activation = output.F
        # 保存当前的输出feature map
        self.input_activations.append(input_activation.cpu().detach())
        self.output_activations.append(output_activation.cpu().detach())

    # backword hook function
    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.input_activations = []
        self.output_activations = []

        p_sparse = self.model(x[1],x[2])
        p_feature = p_sparse.F
        p_coord = p_sparse.C

        return p_feature,p_coord