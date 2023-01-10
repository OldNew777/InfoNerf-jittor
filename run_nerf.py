import os
import sys
import time
import jittor as jt
import numpy as np
from tqdm import tqdm
import random

from dataset.load_llff import load_llff_data
from dataset.load_blender import load_blender_data
from dataset.load_dtu import load_dtu_data
from utils.generate_renderpath import generate_renderpath

from mylogger import logger
from run_nerf_helpers import *

np.random.seed(0)
jt.set_global_seed(0)
random.seed(0)


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return jt.concat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64) -> jt.Var:
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = jt.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = jt.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = jt.concat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = jt.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    if args.alpha_model_path is None:
        model = NeRF(D=args.netdepth, W=args.netwidth,
                     input_ch=input_ch, output_ch=output_ch, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        grad_vars = list(model.parameters())
    else:
        alpha_model = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                           input_ch=input_ch, output_ch=output_ch, skips=skips,
                           input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        print('Alpha model reloading from', args.alpha_model_path)
        ckpt = jt.load(args.alpha_model_path)
        alpha_model.load_state_dict(ckpt['network_fine_state_dict'])
        if not args.no_coarse:
            model = NeRF_RGB(D=args.netdepth, W=args.netwidth,
                             input_ch=input_ch, output_ch=output_ch, skips=skips,
                             input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, alpha_model=alpha_model)
            grad_vars = list(model.parameters())
        else:
            model = None
            grad_vars = []

    model_fine = None
    if args.N_importance > 0:
        if args.alpha_model_path is None:
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                              input_ch=input_ch, output_ch=output_ch, skips=skips,
                              input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        else:
            model_fine = NeRF_RGB(D=args.netdepth_fine, W=args.netwidth_fine,
                                  input_ch=input_ch, output_ch=output_ch, skips=skips,
                                  input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,
                                  alpha_model=alpha_model)
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
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        if args.ckpt_render_iter is not None:
            ckpt_path = os.path.join(os.path.join(basedir, expname, f'{args.ckpt_render_iter:06d}.tar'))

        print('Reloading from', ckpt_path)
        ckpt = jt.load(ckpt_path)

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

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp
    else:
        render_kwargs_train['ndc'] = True

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    ##########################

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def config_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--maskdir", type=str,
                        help='mask data directory')
    parser.add_argument("--fewshot_seed", type=int, default=0,
                        help='fewshot_seed')
    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    #######################################################
    #         Ray Entropy Minimization Loss               #
    #######################################################

    # entropy
    parser.add_argument("--N_entropy", type=int, default=100,
                        help='number of entropy ray')

    # entropy type
    parser.add_argument("--entropy", action='store_true',
                        help='using entropy ray loss')
    parser.add_argument("--entropy_log_scaling", action='store_true',
                        help='using log scaling for entropy loss')
    parser.add_argument("--entropy_ignore_smoothing", action='store_true',
                        help='ignoring entropy for ray for smoothing')
    parser.add_argument("--entropy_end_iter", type=int, default=None,
                        help='end iteratio of entropy')
    parser.add_argument("--entropy_type", type=str, default='log2', choices=['log2', '1-p'],
                        help='choosing type of entropy')
    parser.add_argument("--entropy_acc_threshold", type=float, default=0.1,
                        help='threshold for acc masking')
    parser.add_argument("--computing_entropy_all", action='store_true',
                        help='computing entropy for both seen and unseen ')

    # lambda
    parser.add_argument("--entropy_ray_lambda", type=float, default=1,
                        help='entropy lambda for ray entropy loss')
    parser.add_argument("--entropy_ray_zvals_lambda", type=float, default=1,
                        help='entropy lambda for ray zvals entropy loss')

    #######################################################
    #         Infomation Gain Reduction Loss              #
    #######################################################

    parser.add_argument("--smoothing", action='store_true',
                        help='using information gain reduction loss')
    # choosing between rotating camera pose & near pixel
    parser.add_argument("--smooth_sampling_method", type=str, default='near_pose',
                        help='how to sample the near rays, near_pose: modifying camera pose, near_pixel: sample near pixel',
                        choices=['near_pose', 'near_pixel'])
    # 1) sampling by rotating camera pose
    parser.add_argument("--near_c2w_type", type=str, default='rot_from_origin',
                        help='random augmentation method')
    parser.add_argument("--near_c2w_rot", type=float, default=5,
                        help='random augmentation rotate: degree')
    parser.add_argument("--near_c2w_trans", type=float, default=0.1,
                        help='random augmentation translation')
    # 2) sampling with near pixel
    parser.add_argument("--smooth_pixel_range", type=int,
                        help='the maximum distance between the near ray & the original ray (pixel dimension)')
    # optimizing
    parser.add_argument("--smoothing_lambda", type=float, default=1,
                        help='lambda for smoothing loss')
    parser.add_argument("--smoothing_activation", type=str, default='norm',
                        help='how to make alpha to the distribution')
    parser.add_argument("--smoothing_step_size", type=int, default='5000',
                        help='reducing smoothing every')
    parser.add_argument("--smoothing_rate", type=float, default=1,
                        help='reducing smoothing rate')
    parser.add_argument("--smoothing_end_iter", type=int, default=None,
                        help='when smoothing will be end')

    #######################################################
    #                      Others                         #
    #######################################################

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    # rendering options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--eval_only", action='store_true',
                        help='do not optimize, reload weights and evaluation and logging to wandb')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_test_full", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--ckpt_render_iter", type=int, default=None,
                        help='checkpoint iteration')

    parser.add_argument("--render_test_ray", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_train", action='store_true',
                        help='render the train set instead of render_poses path')
    parser.add_argument("--render_mypath", action='store_true',
                        help='render the test path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--render_pass", action='store_true',
                        help='do not rendering when resume')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--fewshot", type=int, default=0,
                        help='if 0 not using fewshot, else: using fewshot')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--i_wandb", type=int, default=100,
                        help='frequency of logging on wandb(iteration)')
    parser.add_argument("--wandb_group", type=str,
                        help='wandb group name')
    # debug
    parser.add_argument("--debug", action='store_true')

    # new experiment by kangle
    parser.add_argument("--N_iters", type=int, default=200000,
                        help='number of iters')
    parser.add_argument("--alpha_model_path", type=str, default=None,
                        help='predefined alpha model')
    parser.add_argument("--no_coarse", action='store_true',
                        help="Remove coarse network.")
    parser.add_argument("--train_scene", nargs='+', type=int,
                        help='id of scenes used to train')
    parser.add_argument("--test_scene", nargs='+', type=int,
                        help='id of scenes used to test')

    return parser


def train():
    parser = config_parser()
    args = parser.parse_known_args()[0]

    render_first_time = True
    if args.render_pass:
        render_first_time = False

    ########################################
    #              LLFF                    #
    ########################################
    if args.dataset_type == 'llff':
        data_info = jt.load('./data/nerf_llff_data/data_splits.pth')
        datadir_split = args.datadir.split('/')
        if datadir_split[-1] in data_info.keys():
            category = datadir_split[-1]
        if datadir_split[-2] in data_info.keys():
            category = datadir_split[-2]
        full_datadir = os.path.join('./data/nerf_llff_data', category)
        images, poses, bds, render_poses, i_test = load_llff_data(full_datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        logger.info('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            logger.info('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        if args.test_scene is not None:
            i_test = np.array([i for i in args.test_scene])

        if i_test[0] < 0:
            i_test = []

        i_val = i_test

        if args.train_scene is None:
            i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                                (i not in i_test and i not in i_val)])
        else:
            i_train = np.array([i for i in args.train_scene if
                                (i not in i_test and i not in i_val)])

        if args.fewshot > 0:
            i_train = data_info[category][f'{args.fewshot}shot_split'][0]
            if args.train_scene is None:
                np.random.seed(args.fewshot_seed)
                i_train = np.random.choice(i_train, args.fewshot, replace=False)
            else:
                i_train = np.array(args.train_scene)
        logger.info('i_train', i_train)
        logger.info('DEFINING BOUNDS')

        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.
        logger.info('NEAR FAR', near, far)

    ########################################
    #              DTU                     #
    ########################################

    elif args.dataset_type == 'dtu':
        images, poses, hwf, masks = load_dtu_data(args.datadir, args.train_scene, args.maskdir)
        render_poses = poses
        logger.info('Loaded DTU', images.shape, poses.shape, hwf, args.datadir)
        if args.test_scene is not None:
            i_test = np.array([i for i in args.test_scene])

        if i_test[0] < 0:
            i_test = []

        i_val = i_test
        if args.train_scene is None:
            i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                                (i not in i_test and i not in i_val)])
        else:
            i_train = np.array([i for i in args.train_scene if
                                (i not in i_test and i not in i_val)])

        i_test = np.array([i for i in range(len(poses)) if (i not in i_train)])

        near = 0.1
        far = 5.0

    ########################################
    #              Blender                 #
    ########################################

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        logger.info('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        near = 2.
        far = 6.

        if args.fewshot > 0:
            if args.train_scene is None:
                np.random.seed(args.fewshot_seed)
                i_train = np.random.choice(i_train, args.fewshot, replace=False)
            else:
                i_train = np.array(args.train_scene)
            logger.info('i_train', i_train)

        images_mask = images[..., -1]
        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    else:
        logger.info('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])
    elif args.render_train:
        render_poses = np.array(poses[i_train])
    elif args.render_mypath:
        # render_poses = generate_renderpath(np.array(poses[i_test]), focal)
        render_poses = generate_renderpath(np.array(poses[i_test])[3:4], focal, sc=1)

    # Create log dir and copy the config file
    basedir = args.basedir
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


if __name__ == '__main__':
    # disable multi-GPUs before running because of the bug of Jittor
    jt.flags.use_cuda = jt.has_cuda

    # train()
