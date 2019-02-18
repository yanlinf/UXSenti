import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):

    def __init__(self, lambd):
        super(GradReverse, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


class MyNLLLoss(nn.Module):

    def __init__(self):
        super(MyNLLLoss, self).__init__()

    def forward(self, pred, target):
        """
        pred: Tensor of shape (N, C)
        target: LongTensor of shape (N)
        """
        idx = torch.arange(target.size(0))
        return -pred[idx, target].mean()


class WindowSmoothedNLLLoss(nn.Module):

    def __init__(self, eps):
        super(WindowSmoothedNLLLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target, smooth_idx):
        """
        pred: Tensor of shape (N, C)
        target: LongTensor of shape (N)
        smooth_idx: LongTensor of shape (N, W)
        """
        N, W = smooth_idx.size()
        idx1 = torch.arange(N)
        idx2 = torch.stack([torch.arange(N) for _ in range(W)], -1)
        neg_loss = (1 - eps) * pred[idx1, target].mean() + eps * pred[idx2, smooth_idx].mean()
        return -neg_loss

if __name__ == '__main__':
    print('Testing Gradient Reveral Layer:')
    for i in range(10):
        x = torch.randn(100, 100, requires_grad=True)
        y = torch.randn(100, 100, requires_grad=True)
        lambd = torch.randn(())
        rev = GradReverse(lambd)
        loss = rev(x * y).sum()
        loss.backward()
        assert torch.eq(x.grad, y * -lambd).all()
        print('[lambda = {:7.4f}] test {} passed'.format(lambd, i))
    print()

    print('Testing MyNLLLossr:')
    eps = 1e-10
    nll = nn.NLLLoss()
    my_nll = MyNLLLoss()
    for i in range(10):
        x = torch.randn(100, 100)
        y = torch.randint(0, 100, (100,))
        loss = nll(F.log_softmax(x, -1), y)
        my_loss = my_nll(F.log_softmax(x, -1), y)
        print('[{:7.4f} = {:7.4f}] test {} passed'.format(loss, my_loss, i))
        assert (loss - my_loss)**2 < eps
    print()

    print('Testing WindowSmoothedNLLLoss:')

    for i in range(10):
        eps = torch.rand(()).item() * 0.5
        criterion = WindowSmoothedNLLLoss(eps)
        x = F.log_softmax(torch.randn(100, 5), -1)
        y = torch.randint(0, 5, (100,))
        s = torch.randint(0, 5, (100, 7))
        loss = 0
        for j in range(100):
            loss = loss + x[j, y[j]] * (1 - eps) + (x[j][s[j]] * eps / 7).sum()
        loss = -loss / 100
        my_loss = criterion(x, y, s)
        print('[{:7.4f} = {:7.4f}] test {} passed'.format(loss, my_loss, i))
        assert (loss - my_loss)**2 < 1e-10
    print()
