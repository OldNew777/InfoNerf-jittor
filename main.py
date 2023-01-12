import os
import sys
import time

import imageio
import jittor as jt
import numpy as np
from tqdm import tqdm
import random

from mylogger import logger

from data_loader.load_blender import load_blender_data
from config_parser import config_parser

from functions import *
from utils.func import *
from utils.loss import *
from model.model import *


def init_random_seed():
    np.random.seed(0)
    jt.set_global_seed(0)
    random.seed(0)


def batchify(fn, chunk):
    """
    Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return jt.concat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64) -> jt.Var:
    """
    Prepares inputs and applies network 'fn'.
    """
    inputs_flat = jt.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = jt.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = jt.concat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = jt.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """
    Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    all_ret = {k: jt.concat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(height: float, width: float, focal: float, chunk: int = 1024 * 32, rays=None,
           camera2world: jt.Var = None, ndc: bool = True, near: float = 0., far: float = 1.,
           use_viewdirs: bool = False, camera2world_static_camera: jt.Var = None, depths=None,
           **kwargs):
    """Render rays
    Args:
      height: int. Height of image in pixels.
      width: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      camera2world: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      camera2world_static_camera: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other camera2world argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if camera2world is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(height, width, focal, camera2world)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if camera2world_static_camera is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(height, width, focal, camera2world_static_camera)
        viewdirs = viewdirs / jt.norm(viewdirs, dim=-1, keepdims=True)
        viewdirs = jt.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]

    # Create ray batch
    rays_o = jt.reshape(rays_o, [-1, 3]).float()
    rays_d = jt.reshape(rays_d, [-1, 3]).float()

    near, far = near * jt.ones_like(rays_d[..., :1]), far * jt.ones_like(rays_d[..., :1])
    rays = jt.concat([rays_o, rays_d, near, far], -1)  # B x 8
    if depths is not None:
        rays = jt.concat([rays, depths.reshape(-1, 1)], -1)
    if use_viewdirs:
        rays = jt.concat([rays, viewdirs], -1)
    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = jt.reshape(all_ret[k], k_sh)

    # return rgb/disp/depth/... information of the rendered scene
    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs,
                gt_imgs=None, savedir=None, render_factor=0):
    height, width, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        height = height // render_factor
        width = width // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []
    accs = []
    # different camera2world matrix = different pose
    for i, camera2world in enumerate(tqdm(render_poses)):
        rgb, disp, acc, depth, extras = \
            render(height, width, focal, chunk=chunk, camera2world=camera2world[:3, :4], retraw=True, **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        accs.append(acc.cpu().numpy())

        if savedir is not None:
            rgb8 = to8b(rgb.cpu().numpy())
            rgb8[np.isnan(rgb8)] = 0
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            depth = depth.cpu().numpy()
            logger.info("max:", np.nanmax(depth))
            depth = depth / 5 * 255
            imageio.imwrite(os.path.join(savedir, '{:03d}_depth.png'.format(i)), depth)

        del rgb
        del disp
        del acc
        del extras
        del depth
        jt.sync_all()
        jt.gc()

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """
    Instantiate NeRF's MLP model.
    """

    # positional encoding
    # position
    # 3 * (1 + multires) channels output
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    # direction
    # 2 * (1 + multires_views) channels output
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)

        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                        embed_fn=embed_fn,
                                                                        embeddirs_fn=embeddirs_fn,
                                                                        netchunk=args.netchunk)

    # Create optimizer
    optimizer = jt.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        checkpoints = [args.ft_path]
    else:
        checkpoints = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                       'tar' in f]

    logger.info('Found checkpoints', checkpoints)
    if len(checkpoints) > 0 and not args.no_reload:
        checkpoint_path = checkpoints[-1]
        if args.ckpt_render_iter is not None:
            checkpoint_path = os.path.join(os.path.join(basedir, expname, f'{args.ckpt_render_iter:06d}.tar'))

        logger.info('Reloading from', checkpoint_path)
        ckpt = jt.load(checkpoint_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'entropy_ray_zvals': args.entropy,
        'extract_alpha': args.smoothing
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    ##########################

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def render_rays(ray_batch: jt.Var,
                network_fn,
                network_query_fn,
                N_samples: int,
                retraw: bool = False,
                lindisp: bool = False,
                perturb: float = 0.,
                N_importance: int = 0,
                network_fine=None,
                white_bkgd: bool = False,
                raw_noise_std: float = 0.,
                verbose: bool = False,
                sigma_loss=None,
                entropy_ray_zvals=None,
                extract_xyz=None,
                extract_alpha=None,
                extract_sigma=None,
                ):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 9 else None
    bounds = jt.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # get N_sample points averaged over the ray
    t_vals = jt.linspace(0., 1., steps=N_samples)  # [0, 1] as [near, far]
    z_vals = near * (1. - t_vals) + far * (t_vals)  # points in [near, far]

    z_vals = z_vals.expand([N_rays, N_samples])

    # get random points between every two points
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = jt.concat([mids, z_vals[..., -1:]], -1)
        lower = jt.concat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = jt.rand(z_vals.shape)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    # forwarding
    # we don't use model.train() here, because there is not dropout-layer in the model
    if network_fn is not None:
        network_query_fn_use = network_fn
    else:
        network_query_fn_use = network_fine.alpha_model if network_fine.alpha_model is not None else network_fine

    raw = network_query_fn(pts, viewdirs, network_query_fn_use)
    # get VPT rendering results from raw data
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    # Hierarchical sampling
    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        # resampling points in the ray
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
        z_samples = z_samples.detach()

        _, z_vals = jt.argsort(jt.concat([z_vals, z_samples], -1), -1)  # sorted_index, sorted_value
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                            None]  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)

        if entropy_ray_zvals or extract_sigma or extract_alpha:
            rgb_map, disp_map, acc_map, weights, depth_map, others = raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                                 white_bkgd,
                                                                                 out_sigma=True, out_alpha=True,
                                                                                 out_dist=True)
        else:
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map, 'depth_map': depth_map}

    if entropy_ray_zvals or extract_sigma or extract_alpha:
        ret['sigma'] = others['sigma']
        ret['alpha'] = others['alpha']
        ret['z_vals'] = z_vals
        ret['dists'] = others['dists']

    if extract_xyz:
        ret['xyz'] = jt.sum(jt.unsqueeze(weights, -1) * pts, -2)
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0

    for k in ret:
        if jt.isnan(ret[k]).any() or jt.isinf(ret[k]).any():
            logger.debug(f'[Numerical Error] {k} contains nan or inf.', '\n', ret[k])
    return ret


@log_exception
def train():
    parser = config_parser()
    args = parser.parse_args()

    if args.dataset_type != 'blender':
        logger.exception('Unknown dataset type', args.dataset_type, 'exiting')

    # load blender data
    images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
    logger.info('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
    i_train, i_val, i_test = i_split
    logger.info('Train/val/test sizes', len(i_train), len(i_val), len(i_test))
    near = 2.
    far = 6.

    if args.fewshot > 0:
        if args.train_scene is None:
            # generate random noise as initial scene
            np.random.seed(args.fewshot_seed)
            i_train = np.random.choice(i_train, args.fewshot, replace=False)
        else:
            i_train = np.array(args.train_scene)
        logger.info('i_train', i_train)

    if args.white_bkgd:
        images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
    else:
        images = images[..., :3]

    # Cast intrinsics to right types
    height, width, focal = hwf
    height, width = int(height), int(width)
    hwf = [height, width, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])
    elif args.render_train:
        render_poses = np.array(poses[i_train])

    # Create log dir and copy the config file
    basedir = os.path.relpath(args.basedir)
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = jt.array(render_poses)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        logger.info('RENDER ONLY')
        with jt.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            if args.render_test:
                if args.render_test_full:
                    testsavedir = os.path.join(basedir, expname, 'full_renderonly_{}_{:06d}'.format('test', start))
                else:
                    testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test', start))
            elif args.render_train:
                testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('train', start))
            else:
                testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('path', start))
            os.makedirs(testsavedir, exist_ok=True)
            logger.info('test poses shape', render_poses.shape)

            rgbs, disps = render_path(render_poses, hwf, args.chunk, render_kwargs_test, gt_imgs=images,
                                      savedir=testsavedir, render_factor=args.render_factor)
            logger.info('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'rgb.mp4'), to8b(rgbs), fps=30, quality=8)
            disps[np.isnan(disps)] = 0
            logger.info('Depth stats', np.mean(disps), np.max(disps), np.percentile(disps, 95))
            imageio.mimwrite(os.path.join(testsavedir, 'disp.mp4'), to8b(disps / np.percentile(disps, 95)), fps=30,
                             quality=8)
            return

    # Prepare raybatch tensor if batching random rays
    N_rgb = args.N_rand

    if args.entropy:
        N_entropy = args.N_entropy
        fun_entropy_loss = EntropyLoss(args)

    if args.debug:
        return

    # Move training data to GPU
    images = jt.array(images)
    poses = jt.array(poses)

    N_iters = args.N_iters + 1
    logger.info('Begin')
    logger.info('TRAIN views are', i_train)
    logger.info('TEST views are', i_test)
    logger.info('VAL views are', i_val)

    start = start + 1

    if args.eval_only:
        N_iters = start + 2
        i_testset = 1

    for i in tqdm(range(start, N_iters)):
        # Sample random ray batch
        # Random from one image
        img_i = np.random.choice(i_train)
        target = images[img_i]

        rgb_pose = poses[img_i, :3, :4]

        if args.N_rand is not None:
            # translate origins and directions from camera to world coordinates
            rays_o, rays_d = get_rays(height, width, focal, jt.array(rgb_pose))  # (height, W, 3), (height, W, 3)

            # steps to train on central crops
            if i < args.precrop_iters:
                # precrop_frac: fraction of img taken for central crops
                dH = int(height // 2 * args.precrop_frac)
                dW = int(width // 2 * args.precrop_frac)
                coords = jt.stack(
                    jt.meshgrid(
                        jt.linspace(height // 2 - dH, height // 2 + dH - 1, 2 * dH),
                        jt.linspace(width // 2 - dW, width // 2 + dW - 1, 2 * dW)
                    ), -1)
                if i == start:
                    logger.info(
                        f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
            else:
                coords = jt.stack(jt.meshgrid(jt.linspace(0, height - 1, height), jt.linspace(0, width - 1, width)),
                                  -1)  # (height, W, 2)

            coords = jt.reshape(coords, [-1, 2])  # (height * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[N_rgb], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = jt.stack([rays_o, rays_d], 0)  # (2, N_rand, 3)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            ########################################################
            #            Sampling for unseen rays                  #
            ########################################################

            if args.entropy and (args.N_entropy != 0):
                img_i = np.random.choice(len(images))
                target = images[img_i]
                pose = poses[img_i, :3, :4]
                if args.smooth_sampling_method == 'near_pixel':
                    if args.smooth_pixel_range is None:
                        raise Exception('The near pixel is not defined')
                    rays_o, rays_d = get_rays(height, width, focal, jt.array(pose),
                                              padding=args.smooth_pixel_range)  # (height, W, 3), (height, W, 3)
                else:
                    rays_o, rays_d = get_rays(height, width, focal, jt.array(pose))  # (height, W, 3), (height, W, 3)

                if i < args.precrop_iters:
                    dH = int(height // 2 * args.precrop_frac)
                    dW = int(width // 2 * args.precrop_frac)
                    coords = jt.stack(
                        jt.meshgrid(
                            jt.linspace(height // 2 - dH, height // 2 + dH - 1, 2 * dH),
                            jt.linspace(width // 2 - dW, width // 2 + dW - 1, 2 * dW)
                        ), -1)
                    if i == start:
                        logger.info(
                            f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
                else:
                    if args.smooth_sampling_method == 'near_pixel':
                        padding = args.smooth_pixel_range
                        coords = jt.stack(
                            jt.meshgrid(jt.linspace(padding, height - 1 + padding, height),
                                        jt.linspace(padding, width - 1 + padding, width)), -1)  # (height, W, 2)
                    else:
                        coords = jt.stack(
                            jt.meshgrid(jt.linspace(0, height - 1, height), jt.linspace(0, width - 1, width)),
                            -1)  # (height, W, 2)

                coords = jt.reshape(coords, [-1, 2])  # (height * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_entropy], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o_ent = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d_ent = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays_entropy = jt.stack([rays_o_ent, rays_d_ent], 0)  # (2, N_rand, 3)

        N_rgb = batch_rays.shape[1]

        if args.entropy and (args.N_entropy != 0):
            batch_rays = jt.concat([batch_rays, batch_rays_entropy], 1)

        rgb, disp, acc, depth, extras = render(height, width, focal, chunk=args.chunk, rays=batch_rays,
                                               verbose=i < 10, retraw=True,
                                               **render_kwargs_train)

        if args.entropy:
            acc_raw = acc
            alpha_raw = extras['alpha']
            dists_raw = extras['dists']

        extras = {x: extras[x][:N_rgb] for x in extras}

        rgb = rgb[:N_rgb, :]
        disp = disp[:N_rgb]
        acc = acc[:N_rgb]

        img_loss = img2mse(rgb, target_s)
        logging_info = {'rgb_loss': img_loss.item()}
        entropy_ray_zvals_loss = 0

        ########################################################
        #            Ray Entropy Minimiation Loss              #
        ########################################################

        if args.entropy:
            entropy_ray_zvals_loss = fun_entropy_loss.ray_zvals(alpha_raw, acc_raw)
            logging_info['entropy_ray_zvals'] = entropy_ray_zvals_loss.item()

        if args.entropy_end_iter is not None:
            if i > args.entropy_end_iter:
                entropy_ray_zvals_loss = 0

        trans = extras['raw'][..., -1]
        loss = img_loss \
               + args.entropy_ray_zvals_lambda * entropy_ray_zvals_loss
        psnr = mse2psnr(img_loss)
        logging_info['psnr'] = psnr.item()

        if 'rgb0' in extras and not args.no_coarse:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)
            logging_info['rgb0_loss'] = img_loss0.item()
            logging_info['psnr0'] = psnr0.item()

        if i % args.i_wandb == 0:
            logger.debug(f'[Extra] Iter {i:06d}:', logging_info)

        optimizer.step(loss)

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        # Rest is logging
        ## save model ##
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            jt.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict() if render_kwargs_train[
                                                                                               'network_fn'] is not None else None,
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict() if render_kwargs_train[
                                                                                                   'network_fine'] is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            logger.info('Saved checkpoints at', path)

        ## save video ##
        if args.i_video > 0 and i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname.split('/')[-1], i))
            os.makedirs(moviebase, exist_ok=True)
            rgb_path = os.path.join(moviebase, 'rgb.mp4')
            disp_path = os.path.join(moviebase, 'disp.mp4')
            if not (os.path.exists(rgb_path) and os.path.exists(disp_path)):
                logger.info('Rendering movie ...')
                with jt.no_grad():
                    rgbs, disps = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
                logger.info('Done, saving', rgbs.shape, disps.shape)
                imageio.mimwrite(rgb_path, to8b(rgbs), fps=30, quality=8)
                imageio.mimwrite(disp_path, to8b(disps / np.nanmax(disps)), fps=30, quality=8)

        ## save test ##
        if (i % args.i_testset == 0) and (i > 0) and (len(i_test) > 0):
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            rgb_path = os.path.join(testsavedir, 'rgb.mp4')
            disp_path = os.path.join(testsavedir, 'disp.mp4')

            if not (os.path.exists(rgb_path) and os.path.exists(disp_path)):
                logger.info('test poses shape', poses[i_test].shape)
                with jt.no_grad():
                    rgbs, disps = render_path(jt.array(poses[i_test]), hwf, args.chunk, render_kwargs_test,
                                              gt_imgs=images[i_test], savedir=testsavedir)
                logger.info('Saved test set')

                imageio.mimwrite(rgb_path, rgbs, fps=30, quality=8)
                imageio.mimwrite(disp_path, to8b(disps / np.nanmax(disps)), fps=30, quality=8)

                test_loss = img2mse(jt.array(rgbs), images[i_test])
                test_psnr = mse2psnr(test_loss)

                test_redefine_psnr = img2psnr_redefine(jt.array(rgbs), images[i_test])

                # test_ssim, test_msssim = img2ssim(jt.array(rgbs), images[i_test])

                logger.info(f'[Test] Iter {i:06d}:',
                            {'test_psnr': test_psnr.item(),
                             'test_psnr_re': test_redefine_psnr.item(),
                             # 'test_ssim': test_ssim.item(),
                             })

        ## print log ##
        if i % args.i_print == 0:
            logger.info(f"[TRAIN] Iter {i:06d}: Loss: {loss.item()}  PSNR: {psnr.item()}")
        global_step += 1


if __name__ == '__main__':
    logger.set_level(logger.INFO)
    init_random_seed()

    # disable multi-GPUs before running because of the bug of Jittor
    jt.flags.use_cuda = jt.has_cuda

    train()
