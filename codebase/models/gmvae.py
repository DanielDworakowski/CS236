import numpy as np
import torch
from codebase import utils as ut
from codebase.models import nns
from codebase import debug as db
from torch import nn
from torch.nn import functional as F

class GMVAE(nn.Module):
    def __init__(self, nn='v1', z_dim=2, k=500, name='gmvae'):
        super().__init__()
        self.name = name
        self.k = k
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # To help you start, we have computed the mixture of Gaussians prior
        # prior = (m_mixture, v_mixture) for you, where
        # m_mixture and v_mixture each have shape (1, self.k, self.z_dim)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        #
        # Compute the mixture of Gaussian prior
        pm, pv = ut.gaussian_parameters(self.z_pre, dim=1)
        #
        # Generate samples.
        qm, qv = self.enc.encode(x)
        z_sample = ut.sample_gaussian(qm, qv)
        rec = self.dec.decode(z_sample)
        #
        # Compute loss.
        # KL divergence between the latent distribution and the prior.
        rec = -ut.log_bernoulli_with_logits(x, rec)
        # kl = ut.kl_normal(qm, qv, pm, pv)
        kl = ut.log_normal(z_sample, qm, qv) - ut.log_normal_mixture(z_sample, pm, pv)
        #
        # The liklihood of reproducing the sample image given the parameters.
        # Would need to take the average of this otherwise.
        nelbo = (kl + rec).mean()
        # NELBO: 89.24684143066406. KL: 10.346451759338379. Rec: 78.90038299560547
        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl.mean(), rec.mean()

    @torch.no_grad()
    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################
        # Compute the mixture of Gaussian prior
        pm, pv = ut.gaussian_parameters(self.z_pre, dim=1)
        #
        # Generate samples.
        qm, qv = self.enc.encode(x)
        niwaes = []
        recs = []
        kls = []
        for i in range(iw):
            z_sample = ut.sample_gaussian(qm, qv).view(-1, qm.shape[1])
            rec = self.dec.decode(z_sample)
            logptheta_x_g_z = ut.log_bernoulli_with_logits(x, rec)
            logptheta_z = ut.log_normal_mixture(z_sample, pm, pv)
            logqphi_z_g_x = ut.log_normal(z_sample, qm, qv)
            niwae = logptheta_x_g_z + logptheta_z - logqphi_z_g_x
            #
            # Normal variables.
            rec = -ut.log_bernoulli_with_logits(x, rec)
            kl = ut.log_normal(z_sample, qm, qv) - ut.log_normal_mixture(z_sample, pm, pv)
            niwaes.append(niwae)
            recs.append(rec)
            kls.append(kl)
        niwaes = torch.stack(niwaes, -1)
        niwae = ut.log_mean_exp(niwaes, -1)
        kl = torch.stack(kls, -1)
        rec = torch.stack(recs, -1)

        ################################################################################
        # End of code modification
        ################################################################################
        return -niwae.mean(), kl.mean(), rec.mean()

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        m, v = ut.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return ut.sample_gaussian(m, v)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
