import torch


class GradReverse(torch.autograd.Function):

    def __init__(self, lambd):
        super(GradReverse, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


if __name__ == '__main__':
    print('Testing Gradient Reveral Layer:')
    for i in range(10):
        x = torch.randn(100, 100, requires_grad=True)
        y = torch.randn(100, 100, requires_grad=True)
        lambd = torch.randn(())
        rev = GradReverse(lambd)
        loss = rev(x * y).sum()
        loss.backward()
        assert torch.eq(x.grad,y * -lambd).all()
        print('[lambda = {:7.4f}] test {} passed'.format(lambd, i))
