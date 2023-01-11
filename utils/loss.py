import jittor as jt


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

    def entropy(self, prob):
        if self.type_ == 'log2':
            return -1 * prob * jt.log2(prob + 1e-10)
        elif self.type_ == '1-p':
            return prob * jt.log2(1 - prob)

