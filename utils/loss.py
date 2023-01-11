import numpy as np
import jittor as jt


######################################################
#           Ray Entropy Minimization Loss            #
######################################################

class EntropyLoss:
    def __init__(self, args):
        super(EntropyLoss, self).__init__()
        self.N_samples = args.N_rand
        self.type_ = args.entropy_type
        self.threshold = args.entropy_acc_threshold
        self.computing_entropy_all = args.computing_entropy_all
        self.smoothing = args.smoothing
        self.computing_ignore_smoothing = args.entropy_ignore_smoothing
        self.entropy_log_scaling = args.entropy_log_scaling
        self.N_entropy = args.N_entropy

        if self.N_entropy == 0:
            self.computing_entropy_all = True

    def ray(self, density, acc):
        if self.smoothing and self.computing_ignore_smoothing:
            N_smooth = density.size(0) // 2
            acc = acc[:N_smooth]
            density = density[:N_smooth]

        if not self.computing_entropy_all:
            acc = acc[self.N_samples:]
            density = density[self.N_samples:]

        density = jt.nn.relu(density[..., -1])
        sigma = 1 - jt.exp(-density)
        ray_prob = sigma / (jt.unsqueeze(jt.sum(sigma, -1), -1) + 1e-10)
        entropy_ray = jt.sum(self.entropy(ray_prob), -1)

        # masking no hitting poisition?
        mask = (acc > self.threshold).detach()
        entropy_ray *= mask
        entropy_ray_loss = jt.mean(entropy_ray, -1)
        if self.entropy_log_scaling:
            return jt.log(entropy_ray_loss + 1e-10)
        return entropy_ray_loss

    def ray_zvals(self, sigma, acc):
        if self.smoothing and self.computing_ignore_smoothing:
            N_smooth = sigma.size(0) // 2
            acc = acc[:N_smooth]
            sigma = sigma[:N_smooth]
        if not self.computing_entropy_all:
            acc = acc[self.N_samples:]
            sigma = sigma[self.N_samples:]
        ray_prob = sigma / (jt.sum(sigma, -1).unsqueeze(-1) + 1e-10)
        entropy_ray = self.entropy(ray_prob)
        entropy_ray_loss = jt.sum(entropy_ray, -1)

        # masking no hitting poisition?
        mask = (acc > self.threshold).detach()
        entropy_ray_loss *= mask
        if self.entropy_log_scaling:
            return jt.log(jt.mean(entropy_ray_loss) + 1e-10)
        return jt.mean(entropy_ray_loss)

    def ray_zvals_ver1_sigma(self, sigma, dists, acc):
        if self.smoothing and self.computing_ignore_smoothing:
            N_smooth = sigma.size(0) // 2
            acc = acc[:N_smooth]
            sigma = sigma[:N_smooth]
            dists = dists[:N_smooth]

        if not self.computing_entropy_all:
            acc = acc[self.N_samples:]
            sigma = sigma[self.N_samples:]
            dists = dists[self.N_samples:]

        ray_prob = sigma / (jt.sum(sigma * dists, -1).unsqueeze(-1) + 1e-10)
        entropy_ray = self.entropy(ray_prob)

        # intergral
        entropy_ray = entropy_ray * dists
        entropy_ray_loss = jt.sum(entropy_ray, -1)

        # masking no hitting poisition?
        mask = (acc > self.threshold).detach()
        entropy_ray_loss *= mask
        if self.entropy_log_scaling:
            return jt.log(jt.mean(entropy_ray_loss) + 1e-10)
        return jt.mean(entropy_ray_loss)

    def ray_zvals_ver2_alpha(self, alpha, dists, acc):
        if self.smoothing and self.computing_ignore_smoothing:
            N_smooth = alpha.size(0) // 2
            acc = acc[:N_smooth]
            alpha = alpha[:N_smooth]
            dists = dists[:N_smooth]

        if not self.computing_entropy_all:
            acc = acc[self.N_samples:]
            alpha = alpha[self.N_samples:]
            dists = dists[self.N_samples:]

        ray_prob = alpha / (jt.sum(alpha, -1).unsqueeze(-1) + 1e-10)

        entropy_ray = -1 * ray_prob * jt.log2(ray_prob / (dists + 1e-10) + 1e-10)
        entropy_ray_loss = jt.sum(entropy_ray, -1)
        # masking no hitting poisition?
        mask = (acc > self.threshold).detach()
        entropy_ray_loss *= mask
        if self.entropy_log_scaling:
            return jt.log(jt.mean(entropy_ray_loss) + 1e-10)
        return jt.mean(entropy_ray_loss)

    def entropy(self, prob):
        if self.type_ == 'log2':
            return -1 * prob * jt.log2(prob + 1e-10)
        elif self.type_ == '1-p':
            return prob * jt.log2(1 - prob)


######################################################
#          Infomation Gain Reduction Loss            #
######################################################

class SmoothingLoss:
    def __init__(self, args):
        super(SmoothingLoss, self).__init__()

        self.smoothing_activation = args.smoothing_activation
        self.criterion = jt.nn.KLDivLoss(reduction='batchmean')

    def __call__(self, sigma):
        half_num = sigma.size(0) // 2
        sigma_1 = sigma[:half_num]
        sigma_2 = sigma[half_num:]

        if self.smoothing_activation == 'softmax':
            p = jt.nn.softmax(sigma_1, -1)
            q = jt.nn.softmax(sigma_2, -1)
        elif self.smoothing_activation == 'norm':
            p = sigma_1 / (jt.sum(sigma_1, dims=-1, keepdims=True) + 1e-10) + 1e-10
            q = sigma_2 / (jt.sum(sigma_2, dims=-1, keepdims=True) + 1e-10) + 1e-10
        loss = self.criterion(p.log(), q)
        return loss
