import torch

from torch.autograd import Function


class IdentityFunction(Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, x):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        x, = ctx.saved_tensors
        return grad_output


class SigmoidFunction(Function):
    @staticmethod
    def forward(ctx, input):
        out = 1.0 / (1.0 + torch.exp(-input))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        return grad_output * out * (1.0 - out)


class LinearFunction(Function):
    @staticmethod
    def forward(ctx, inp, weight, bias):
        ctx.save_for_backward(inp, weight, bias)
        return torch.mm(inp, weight.t()) + bias

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight, bias = ctx.saved_tensors
        grad_inp = torch.mm(grad_output, weight)
        grad_weight = torch.mm(grad_output.t(), inp)
        grad_bias = grad_output.sum(0)
        return grad_inp, grad_weight, grad_bias


class CrossEntropyFunction(Function):
    @staticmethod
    def forward(ctx, logits, target):
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)
        shifted_logits = logits - max_logits
        
        log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=1, keepdim=True))
        
        log_probs = shifted_logits - log_sum_exp
        
        N = logits.size(0)
        loss = -torch.sum(log_probs[torch.arange(N), target]) / N
        
        probs = torch.exp(log_probs)
        ctx.save_for_backward(probs, target)
        
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        probs, target = ctx.saved_tensors
        N = probs.size(0)
        
        grad_logits = probs.clone()
        grad_logits[torch.arange(N), target] -= 1.0
        grad_logits = grad_logits / N
        
        return grad_output * grad_logits, None


if __name__ == "__main__":
    from torch.autograd import gradcheck

    num = 4
    inp = 3

    x = torch.rand((num, inp), requires_grad=True).double()

    sigmoid = SigmoidFunction.apply

    assert gradcheck(sigmoid, x)
    print("Backward pass for sigmoid function is implemented correctly")

    out = 2

    x = torch.rand((num, inp), requires_grad=True).double()
    weight = torch.rand((out, inp), requires_grad=True).double()
    bias = torch.rand(out, requires_grad=True).double()

    linear = LinearFunction.apply
    assert gradcheck(linear, (x, weight, bias))
    print("Backward pass for linear function is implemented correctly")

    activations = torch.rand((15, 10), requires_grad=True).double()
    target = torch.randint(10, (15,))
    crossentropy = CrossEntropyFunction.apply
    assert gradcheck(crossentropy, (activations, target))
    print("Backward pass for crossentropy function is implemented correctly")
