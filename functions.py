from typing import Tuple

import jittor as jt
import jittor.nn as nn
import numpy as np
from jittor.misc import searchsorted

from matplotlib import pyplot as plt

from utils.jittor_msssim import ssim, ms_ssim

from mylogger import logger

# Misc
img2mse = lambda x, y: jt.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * jt.log(x) / jt.log(jt.array([10.], dtype=jt.float32))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def img2psnr_redefine(x, y) -> jt.Var:
    '''
    we redefine the PSNR function,
    [previous]
    average MSE -> PSNR(average MSE)
    
    [new]
    average PSNR(each image pair)
    '''
    image_num = x.size(0)
    mses = ((x - y) ** 2).reshape(image_num, -1).mean(-1)

    psnrs = [mse2psnr(mse) for mse in mses]
    psnr = jt.stack(psnrs).mean()
    return psnr


def img2ssim(x, y, mask=None) -> Tuple[jt.Var, jt.Var]:
    if mask is not None:
        x = jt.unsqueeze(mask, -1) * x
        y = jt.unsqueeze(mask, -1) * y

    x = x.permute(0, 3, 1, 2)
    y = y.permute(0, 3, 1, 2)
    ssim_ = ssim(x, y, data_range=1)
    ms_ssim_ = ms_ssim(x, y, data_range=1)
    return ssim_, ms_ssim_


# Ray helpers
def get_rays(height: int, width: int, focal: float, camera2world: jt.Var, padding=None) -> Tuple[jt.Var, jt.Var]:
    # pyjt's meshgrid has indexing='ij'
    if padding is not None:
        i, j = jt.meshgrid(jt.linspace(-padding, width - 1 + padding, width + 2 * padding),
                           jt.linspace(-padding, height - 1 + padding, height + 2 * padding))
    else:
        i, j = jt.meshgrid(jt.linspace(0, width - 1, width), jt.linspace(0, height - 1, height))
    i = i.t()
    j = j.t()
    dirs = jt.stack([(i - width * .5) / focal, -(j - height * .5) / focal, -jt.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = jt.sum(dirs[..., np.newaxis, :] * camera2world[:3, :3],
                    -1)  # dot product, equals to: [camera2world.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = camera2world[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
# 提高 voxel 渲染效率
# 由粗到细的结构来训练
# 首先采样一组位置信息，基于 stratied sampling，然后训练一个“粗”网络。在此基础上，再训练一个"细"网络
def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / jt.sum(weights, dim=-1, keepdims=True)
    cdf = jt.cumsum(pdf, -1)
    cdf = jt.concat([jt.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = jt.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = jt.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    u = u.contiguous()
    inds = searchsorted(cdf, u, right=True)
    below = jt.maximum(jt.zeros_like(inds - 1), inds - 1)
    above = jt.minimum((cdf.shape[-1] - 1) * jt.ones_like(inds), inds)
    inds_g = jt.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = jt.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = jt.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = jt.where(denom < 1e-5, jt.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


# 从 raw 数据得到体渲染结果
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False,
                out_alpha=False, out_sigma=False, out_dist=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=jt.nn.relu: 1. - jt.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = jt.concat([dists, jt.array([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * jt.norm(rays_d[..., None, :], dim=-1)

    rgb = jt.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    # if raw_noise_std > 0.:
    #     noise = jt.randn(raw[..., 3].shape) * raw_noise_std

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    sigma = jt.nn.relu(raw[..., 3] + noise)
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * jt.cumprod(jt.concat([jt.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = jt.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = jt.sum(weights * z_vals, -1)
    disp_map = 1. / jt.maximum(jt.float32(1e-10) * jt.ones_like(depth_map), depth_map / jt.sum(weights, -1))
    acc_map = jt.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    others = {}
    if out_alpha or out_sigma or out_dist:
        if out_alpha:
            others['alpha'] = alpha
        if out_sigma:
            others['sigma'] = sigma
        if out_dist:
            others['dists'] = dists
        return rgb_map, disp_map, acc_map, weights, depth_map, others
    return rgb_map, disp_map, acc_map, weights, depth_map
