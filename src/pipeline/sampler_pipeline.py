from copy import copy

import torch as T
import torch.nn as nn
from torch.autograd import grad
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from src.scm.nn.custom_nn import CustomNN
from src.scm.distribution.continuous_distribution import UniformDistribution
from src.scm.ncm.gan_ncm import Discriminator
from src.datagen.scm_datagen import SCMDataTypes as sdt

from .base_pipeline import BasePipeline


def log(x):
    return T.log(x + 1e-8)


class SamplerPipeline(BasePipeline):
    def __init__(self, datagen, i_size, i_type, u_size, o_name, o_size, o_type, var_mapping, hyperparams=None):
        if hyperparams is None:
            hyperparams = dict()

        super().__init__(datagen, None, None, batch_size=hyperparams.get('bs', 1000))
        self.automatic_optimization = False

        self.datagen = datagen
        self.sampler_mode = hyperparams["mode"]
        self.gan_mode = hyperparams["gan-mode"]

        self.i_size = i_size
        self.i_type = i_type
        self.o_name = o_name
        self.o_size = o_size
        self.o_type = o_type
        self.var_mapping = var_mapping
        self.img_size = hyperparams["img-size"]
        self.h_layers = hyperparams["h-layers"]
        self.h_size = hyperparams["h-size"]
        self.feature_maps = hyperparams["feature-maps"]

        self.v_size = copy(i_size)
        self.v_type = copy(i_type)
        self.v_size[o_name] = o_size
        self.v_type[o_name] = o_type

        if self.sampler_mode == "projected":
            self.u_size = {'U_proj': u_size}
            self.noise = UniformDistribution({'U_proj'}, self.u_size)
        else:
            self.u_size = dict()
            self.noise = None

        self.gen = CustomNN(i_size, self.u_size, o_size, i_type, o_type, img_size=self.img_size,
                              img_embed_size=self.h_size, feature_maps=self.feature_maps, h_size=self.h_size,
                              h_layers=self.h_layers, use_batch_norm=hyperparams["batch-norm"],
                              mode=hyperparams["gan-arch"])

        if self.sampler_mode == "projected":
            self.disc = Discriminator(self.v_size, self.v_type,
                                      disc_use_sigmoid=(hyperparams.get("gan-mode", "NA") != "wgan"),
                                      hyperparams=hyperparams)
        else:
            self.disc = None

        self.batch_size = hyperparams.get('bs', 1000)
        self.d_iters = hyperparams.get('d-iters', 1)
        self.cut_batch_size = self.batch_size // self.d_iters
        self.grad_acc = hyperparams.get('grad-acc', 1)
        self.grad_clamp = hyperparams.get('grad-clamp', 0.01)
        self.gp_weight = hyperparams.get('gp-weight', 10.0)
        self.gp_one_side = hyperparams.get('gp-one-side', False)
        self.gen_lr = hyperparams['lr']
        self.disc_lr = hyperparams['disc-lr']
        self.alpha = hyperparams['alpha']

        self.compare_loss = nn.MSELoss()

        self.logged = False
        self.stored_loss = 1e8
        self.img_lists = {v: [] for v in self.v_type}

    def configure_optimizers(self):
        if self.sampler_mode != "projected":
            opt = T.optim.Adam(self.gen.parameters(), lr=self.gen_lr)
            return opt

        if self.gan_mode == "wgan":
            opt_gen = T.optim.RMSprop(self.gen.parameters(), lr=self.gen_lr, alpha=self.alpha)
            opt_disc = T.optim.RMSprop(self.disc.parameters(), lr=self.disc_lr, alpha=self.alpha)
        else:
            opt_gen = T.optim.Adam(self.gen.parameters(), lr=self.gen_lr)
            opt_disc = T.optim.Adam(self.disc.parameters(), lr=self.disc_lr)
        return opt_gen, opt_disc

    def _get_D_loss(self, real_out, fake_out):
        if self.gan_mode == "wgan" or self.gan_mode == "wgangp":
            return -(T.mean(real_out) - T.mean(fake_out))
        else:
            return -T.mean(log(real_out) + log(1 - fake_out))

    def _get_G_loss(self, fake_out):
        if self.gan_mode == "bgan":
            return 0.5 * T.mean((log(fake_out) - log(1 - fake_out)) ** 2)
        elif self.gan_mode == "wgan" or self.gan_mode == "wgangp":
            return -T.mean(fake_out)
        else:
            return -T.mean(log(fake_out))

    def _get_gradient_penalty(self, real_data, fake_data):
        interpolated_data = dict()
        alpha = T.rand(self.batch_size, 1, device=self.device, requires_grad=True)
        for V in real_data:
            if self.v_type[V] is not sdt.IMAGE:
                v_alpha = alpha.expand_as(real_data[V])
            else:
                v_alpha = alpha[:, :, None, None].expand_as(real_data[V])
            interpolated_data[V] = v_alpha * real_data[V].detach() + (1 - v_alpha) * fake_data[V].detach()

        interpolated_out, inp_set = self.disc(interpolated_data, include_inp=True)
        gradients_norm = 0
        for inp in inp_set:
            if inp is not None:
                gradients = grad(outputs=interpolated_out, inputs=inp,
                                 grad_outputs=T.ones(interpolated_out.size(), device=self.device),
                                 create_graph=True, retain_graph=True)[0]
                gradients = gradients.view(self.batch_size, -1)
                gradients_norm += T.sum(gradients ** 2, dim=1)
        gradients_norm = T.sqrt(gradients_norm + 1e-12)
        if self.gp_one_side:
            return self.gp_weight * (T.relu(gradients_norm - self.grad_clamp) ** 2).mean()
        return self.gp_weight * ((gradients_norm - self.grad_clamp) ** 2).mean()

    def sample(self, n, ncm):
        data = ncm(n=n, evaluating=True)
        cond_dict = {self.var_mapping[k]: v for (k, v) in data.items() if k in self.var_mapping}
        return self.gen_forward(n, cond_dict)

    def gen_forward(self, n, cond_dict):
        if self.sampler_mode == "projected":
            u = self.noise.sample(n)
            return self.gen(cond_dict, u)
        else:
            return self.gen(cond_dict, dict())

    def training_step(self, batch, batch_idx):
        if self.sampler_mode == "projected":
            G_opt, D_opt = self.optimizers()
            n = self.batch_size

            G_opt.zero_grad()

            # Train Discriminator
            total_d_loss = 0
            for d_iter in range(self.d_iters):
                D_opt.zero_grad()

                real_batch = {k: v[d_iter * self.cut_batch_size:(d_iter + 1) * self.cut_batch_size].float()
                              for (k, v) in batch.items()}
                fake_batch = {k: v for (k, v) in real_batch.items() if k in self.i_type}
                gen_out = self.gen_forward(self.cut_batch_size, fake_batch)
                fake_batch[self.o_name] = gen_out


                disc_real_out = self.disc(real_batch)
                disc_fake_out = self.disc(fake_batch)
                D_loss = self._get_D_loss(disc_real_out, disc_fake_out)

                if self.gan_mode == "wgangp":
                    grad_penalty = self._get_gradient_penalty(real_batch, fake_batch)
                    self.log('grad_penalty', grad_penalty, prog_bar=True)
                    D_loss += grad_penalty

                total_d_loss += D_loss.item()
                self.manual_backward(D_loss)

                if ((self.d_iters * batch_idx + d_iter + 1) % self.grad_acc) == 0:
                    D_opt.step()

                if self.gan_mode == "wgan":
                    for p in self.disc.parameters():
                        p.data.clamp_(-self.grad_clamp, self.grad_clamp)

                self.gen.zero_grad()
                self.disc.zero_grad()

            # Train Generator
            g_loss_record = 0
            fake_batch = {k: v for (k, v) in batch.items() if k in self.i_type}
            gen_out = self.gen_forward(n, fake_batch)
            fake_batch[self.o_name] = gen_out
            disc_fake_out = self.disc(fake_batch)
            G_loss = self._get_G_loss(disc_fake_out)
            g_loss_record += G_loss.item()
            self.manual_backward(G_loss)

            if ((batch_idx + 1) % self.grad_acc) == 0:
                G_opt.step()

            self.gen.zero_grad()
            self.disc.zero_grad()

            self.log('train_loss', self.stored_loss, prog_bar=True)
            self.log('G_loss', g_loss_record, prog_bar=True)
            self.log('D_loss', total_d_loss, prog_bar=True)
            self.stored_loss -= 0.1
        else:
            n = self.batch_size
            opt = self.optimizers()

            opt.zero_grad()
            batch_key = next(iter(self.i_size))
            fake_batch = {batch_key: batch[batch_key]}
            gen_out = self.gen_forward(n, fake_batch)

            loss = self.compare_loss(gen_out, batch[self.o_name])
            self.manual_backward(loss)
            opt.step()

            self.log('train_loss', self.stored_loss, prog_bar=True)
            self.log('loss', loss.item(), prog_bar=True)
            self.stored_loss -= 0.1

        # logging
        if (self.current_epoch + 1) % 10 == 0:
            if not self.logged:
                self.logged = True

                log_batch = self.datagen[:64]
                in_batch = {k: v.to(self.device) for (k, v) in log_batch.items() if k in self.i_type}
                sample = self.gen_forward(64, in_batch)
                for v in self.v_type:
                    self.img_lists[v].append(sample.detach().cpu())

        else:
            self.logged = False
