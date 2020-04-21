import torch
from torch.nn import functional as F

def loss_nonsaturating(g, d, x_real, *, device):
    '''
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating discriminator loss
    - g_loss (torch.Tensor): nonsaturating generator loss
    '''
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # YOUR CODE STARTS HERE
    # You may find some or all of the below useful:
    #   - F.binary_cross_entropy_with_logits
    #   - F.logsigmoid
    d_pred = d(g(z))
    g_loss = -F.logsigmoid(d_pred).mean()
    d_loss = F.binary_cross_entropy_with_logits(d(x_real), torch.ones(x_real.shape[0], device=x_real.device)).mean() \
             + F.binary_cross_entropy_with_logits(d_pred, torch.zeros(x_real.shape[0], device=x_real.device)).mean()
    # YOUR CODE ENDS HERE

    return d_loss, g_loss

def conditional_loss_nonsaturating(g, d, x_real, y_real, *, device):
    '''
    Arguments:
    - g (codebase.network.ConditionalGenerator): The generator network
    - d (codebase.network.ConditionalDiscriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - y_real (torch.Tensor): training data labels (64)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating conditional discriminator loss
    - g_loss (torch.Tensor): nonsaturating conditional generator loss
    '''
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    y_fake = y_real  # use the real labels as the fake labels as well

    # YOUR CODE STARTS HERE
    d_pred = d(g(z, y_fake), y_fake)
    d_loss = F.binary_cross_entropy_with_logits(d(x_real, y_real), torch.ones(x_real.shape[0], device=x_real.device)).mean() \
        + F.binary_cross_entropy_with_logits(d_pred, torch.zeros(x_real.shape[0], device=x_real.device)).mean()
    g_loss = -F.logsigmoid(d_pred).mean()
    # YOUR CODE ENDS HERE

    return d_loss, g_loss

def loss_wasserstein_gp(g, d, x_real, *, device):
    '''
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs value of discriminator
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): wasserstein discriminator loss
    - g_loss (torch.Tensor): wasserstein generator loss
    '''
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # YOUR CODE STARTS HERE
    # You may find some or all of the below useful:
    #   - torch.rand
    #   - torch.autograd.grad(..., create_graph=True)
    lambda_ = 10
    x_fake = g(z)
    d_pred = d(x_fake)
    d_loss = d_pred.mean() - d(x_real).mean()
    alpha = torch.rand(batch_size, device=device).view(-1, 1, 1, 1)
    alphap = 1 - alpha
    cvx_x = alpha * x_fake + alphap * x_real
    d_cvxx = d(cvx_x)
    d_loss += lambda_ * (torch.autograd.grad(d_cvxx, cvx_x, torch.ones_like(d_cvxx), create_graph=True)[0].view(batch_size, -1).norm(dim=1) - 1).pow_(2).mean()
    # d_loss += lambda_ * (torch.autograd.grad(d_cvxx.sum(), cvx_x, create_graph=True)[0].norm(dim=1) - 1).pow_(2).mean()
    g_loss = -d_pred.mean()
    # YOUR CODE ENDS HERE

    return d_loss, g_loss
