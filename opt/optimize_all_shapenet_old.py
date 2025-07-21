# Copyright 2021 Alex Yu

# First, install svox2
# Then, python opt.py <path_to>/nerf_synthetic/<scene> -t ckpt/<some_name>
# or use launching script:   sh launch.sh <EXP_NAME> <GPU> <DATA_DIR>
MAIN_DIR = "/om/user/akiruga/svox2/opt"
import sys 
sys.path.append(MAIN_DIR)
import torch
import torch.cuda
import torch.optim
import torch.nn.functional as F
import svox2
import json
import imageio
import os
from os import path
import shutil
import gc
import numpy as np
from pathlib import Path
import math
import argparse
import cv2
import glob
import time
from util.dataset import datasets
from util.util import Timing, get_expon_lr_func, generate_dirs_equirect, viridis_cmap
from util import config_util

from warnings import warn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from typing import NamedTuple, Optional, Union

device = "cuda" if torch.cuda.is_available() else "cpu"


# Create argument namespace object to mimic argparse
class Args:
    def __init__(self):
        pass

def get_default_args():
    """Get default arguments for training"""
    args = Args()
    
    # Set all the default argument values as attributes
    args.base_train_dir = '/om/user/akiruga/svox2/data/ckpts/shapenet_chairs_all_jupyter' # base directory for all chairs
    # args.reso = "[[32, 32, 32]]"
    args.reso = "[[256, 256, 256], [512, 512, 512]]"
    args.upsamp_every = 3 * 12800
    args.init_iters = 0
    args.upsample_density_add = 0.0
    args.basis_type = 'sh'
    args.basis_reso = 32
    args.sh_dim = 9
    args.mlp_posenc_size = 4
    args.mlp_width = 32
    args.background_nlayers = 0
    args.background_reso = 512
    args.n_iters = 10 * 12800
    args.batch_size = 5000

    args.sigma_optim = 'rmsprop'
    args.lr_sigma = 3e1
    args.lr_sigma_final = 5e-2
    args.lr_sigma_decay_steps = 250000
    args.lr_sigma_delay_steps = 15000
    args.lr_sigma_delay_mult = 1e-2

    args.sh_optim = 'rmsprop'
    args.lr_sh = 1e-2
    args.lr_sh_final = 5e-6
    args.lr_sh_decay_steps = 250000
    args.lr_sh_delay_steps = 0
    args.lr_sh_delay_mult = 1e-2

    args.lr_fg_begin_step = 0

    args.bg_optim = 'rmsprop'
    args.lr_sigma_bg = 3e0
    args.lr_sigma_bg_final = 3e-3
    args.lr_sigma_bg_decay_steps = 250000
    args.lr_sigma_bg_delay_steps = 0
    args.lr_sigma_bg_delay_mult = 1e-2

    args.lr_color_bg = 1e-1
    args.lr_color_bg_final = 5e-6
    args.lr_color_bg_decay_steps = 250000
    args.lr_color_bg_delay_steps = 0
    args.lr_color_bg_delay_mult = 1e-2

    args.basis_optim = 'rmsprop'
    args.lr_basis = 1e-6
    args.lr_basis_final = 1e-6
    args.lr_basis_decay_steps = 250000
    args.lr_basis_delay_steps = 0
    args.lr_basis_begin_step = 0
    args.lr_basis_delay_mult = 1e-2

    args.rms_beta = 0.95
    args.print_every = 20
    args.save_every = 5
    args.eval_every = 1

    args.init_sigma = 0.1
    args.init_sigma_bg = 0.1

    args.log_mse_image = True
    args.log_depth_map = True
    args.log_depth_map_use_thresh = None

    args.thresh_type = "weight"
    args.weight_thresh = 0.0005 * 512
    args.density_thresh = 5.0
    args.background_density_thresh = 1.0+1e-9
    args.max_grid_elements = 44_000_000

    args.tune_mode = False
    args.tune_nosave = False

    args.lambda_tv = 1e-5
    args.tv_sparsity = 0.01
    args.tv_logalpha = False
    args.lambda_tv_sh = 1e-3
    args.tv_sh_sparsity = 0.01
    args.lambda_tv_lumisphere = 0.0
    args.tv_lumisphere_sparsity = 0.01
    args.tv_lumisphere_dir_factor = 0.0
    args.tv_decay = 1.0
    args.lambda_l2_sh = 0.0
    args.tv_early_only = 1
    args.tv_contiguous = 1

    args.lambda_sparsity = 0.0
    args.lambda_beta = 0.0

    args.lambda_tv_background_sigma = 1e-2
    args.lambda_tv_background_color = 1e-2
    args.tv_background_sparsity = 0.01

    args.lambda_tv_basis = 0.0

    args.weight_decay_sigma = 1.0
    args.weight_decay_sh = 1.0

    args.lr_decay = True
    args.n_train = None
    args.nosphereinit = False

    # Add common args from config_util
    args.config = None
    args.dataset_type = "shapenet"
    args.scene_scale = None
    args.scale = None
    args.seq_id = 1000
    args.epoch_size = 12800
    args.white_bkgd = True
    args.llffhold = 8
    args.normalize_by_bbox = False
    args.data_bbox_scale = 1.2
    args.cam_scale_factor = 0.95
    args.normalize_by_camera = True
    args.perm = False
    args.step_size = 0.5
    args.sigma_thresh = 1e-8
    args.stop_thresh = 1e-7
    args.background_brightness = 1.0
    args.renderer_backend = 'cuvol'
    args.random_sigma_std = 0.0
    args.random_sigma_std_background = 0.0
    args.near_clip = 0.00
    args.use_spheric_clip = False
    args.enable_random = False
    args.last_sample_opaque = False
    
    return args

def render_video(grid,
                 cameras,
                 out_path,
                 fps: int = 12,
                 crop: float = 1.0):
    """
    Render a simple orbit video of the current grid and dump to MP4.
    Args:
        grid (svox2.SparseGrid): trained / training grid
        cameras (List[svox2.Camera]): list of camera poses
        out_path (str | Path): where to write the mp4
        fps (int): frames per second
        crop (float): 1.0 = full res, <1 crops center
    """
    grid.eval()                        # just for safety
    frames = []
    with torch.no_grad():
        for cam in cameras:
            # Optional center‑crop so you can render faster mid‑training
            w, h = cam.width, cam.height
            if crop < 1.0:
                cam = svox2.Camera(
                    cam.c2w, cam.fx, cam.fy,
                    cam.cx * crop, cam.cy * crop,
                    int(w * crop), int(h * crop),
                    ndc_coeffs=cam.ndc_coeffs
                )
            im = grid.volume_render_image(cam, use_kernel=True)
            im = (im.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            frames.append(im)
    imageio.mimwrite(str(out_path), frames, fps=fps, macro_block_size=8)
    print(f"✔️  Saved preview video → {out_path}")
    grid.train()


def get_dense_density_sh(grid):
    """
    Returns a dense 4D grid with shape (X, Y, Z, C), where:
    - C = 1 + basis_dim * 3
    - Channel 0: density
    - Channels 1: density SH coefficients (flattened)
    """
    X, Y, Z = grid.links.shape
    C = 1 + grid.sh_data.shape[1]  # 1 for density + SH channels
    dense_grid = torch.zeros((X, Y, Z, C), device=grid.links.device)

    mask = grid.links >= 0
    active_indices = grid.links[mask]

    dense_grid_flat = dense_grid.view(-1, C)
    mask_flat = mask.view(-1)

    # Fill density
    dense_grid_flat[mask_flat, 0] = grid.density_data.detach()[active_indices, 0]

    # Fill SH coefficients
    dense_grid_flat[mask_flat, 1:] = grid.sh_data.detach()[active_indices]

    dense_grid = dense_grid_flat.view(X, Y, Z, C)
    return dense_grid


def is_object_already_trained(base_train_dir, object_name):
    """Check if an object has already been trained by looking for its output directory and key files"""
    object_dir = os.path.join(base_train_dir, object_name)
    
    # Check if directory exists and has key files indicating completed training
    if not os.path.exists(object_dir):
        return False
    
    # Check for key files that indicate training completion
    required_files = ['ckpt.npz', 'dense_grid.npz', 'training_metadata.json']
    for required_file in required_files:
        if not os.path.exists(os.path.join(object_dir, required_file)):
            return False
    
    return True


def train_single_object(data_dir, object_id, total_objects, args):
    """Train a model on a single object"""
    
    object_name = os.path.basename(os.path.dirname(data_dir))
    print(f"\n🪑 Training object {object_id}/{total_objects}: {object_name}")
    
    # Create object-specific training directory
    args.train_dir = os.path.join(args.base_train_dir, object_name)
    os.makedirs(args.train_dir, exist_ok=True)
    
    summary_writer = SummaryWriter(args.train_dir)
    
    # Set data directory for this object
    args.data_dir = data_dir
    
    reso_list = json.loads(args.reso)
    reso_id = 0

    with open(path.join(args.train_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    torch.manual_seed(20200823)
    np.random.seed(20200823)

    factor = 1

    # Load dataset for this specific object
    dset = datasets[args.dataset_type](
                   args.data_dir,
                   split="train",
                   device=device,
                   factor=factor,
                   n_images=args.n_train,
                   **config_util.build_data_options(args))

    if args.background_nlayers > 0 and not dset.should_use_background:
        warn('Using a background model for dataset type ' + str(type(dset)) + ' which typically does not use background')

    object_start_time = datetime.now()

    grid = svox2.SparseGrid(reso=reso_list[reso_id],
                            center=dset.scene_center,
                            radius=dset.scene_radius,
                            use_sphere_bound=dset.use_sphere_bound and not args.nosphereinit,
                            basis_dim=args.sh_dim,
                            use_z_order=True,
                            device=device,
                            basis_reso=args.basis_reso,
                            basis_type=svox2.__dict__['BASIS_TYPE_' + args.basis_type.upper()],
                            mlp_posenc_size=args.mlp_posenc_size,
                            mlp_width=args.mlp_width,
                            background_nlayers=args.background_nlayers,
                            background_reso=args.background_reso)

    # DC -> gray; mind the SH scaling!
    grid.sh_data.data[:] = 0.0
    grid.density_data.data[:] = 0.0 if args.lr_fg_begin_step > 0 else args.init_sigma

    if grid.use_background:
        grid.background_data.data[..., -1] = args.init_sigma_bg

    optim_basis_mlp = None

    if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
        grid.reinit_learned_bases(init_type='sh')

    elif grid.basis_type == svox2.BASIS_TYPE_MLP:
        # MLP!
        optim_basis_mlp = torch.optim.Adam(
                        grid.basis_mlp.parameters(),
                        lr=args.lr_basis
                    )

    grid.requires_grad_(True)
    config_util.setup_render_opts(grid.opt, args)
    print('Render options', grid.opt)

    gstep_id_base = 0

    resample_cameras = [
            svox2.Camera(c2w.to(device=device),
                         dset.intrins.get('fx', i),
                         dset.intrins.get('fy', i),
                         dset.intrins.get('cx', i),
                         dset.intrins.get('cy', i),
                         width=dset.get_image_size(i)[1],
                         height=dset.get_image_size(i)[0],
                         ndc_coeffs=dset.ndc_coeffs) for i, c2w in enumerate(dset.c2w)
        ]
    ckpt_path = path.join(args.train_dir, 'ckpt.npz')

    lr_sigma_func = get_expon_lr_func(args.lr_sigma, args.lr_sigma_final, args.lr_sigma_delay_steps,
                                      args.lr_sigma_delay_mult, args.lr_sigma_decay_steps)
    lr_sh_func = get_expon_lr_func(args.lr_sh, args.lr_sh_final, args.lr_sh_delay_steps,
                                   args.lr_sh_delay_mult, args.lr_sh_decay_steps)
    lr_basis_func = get_expon_lr_func(args.lr_basis, args.lr_basis_final, args.lr_basis_delay_steps,
                                   args.lr_basis_delay_mult, args.lr_basis_decay_steps)
    lr_sigma_bg_func = get_expon_lr_func(args.lr_sigma_bg, args.lr_sigma_bg_final, args.lr_sigma_bg_delay_steps,
                                   args.lr_sigma_bg_delay_mult, args.lr_sigma_bg_decay_steps)
    lr_color_bg_func = get_expon_lr_func(args.lr_color_bg, args.lr_color_bg_final, args.lr_color_bg_delay_steps,
                                   args.lr_color_bg_delay_mult, args.lr_color_bg_decay_steps)
    lr_sigma_factor = 1.0
    lr_sh_factor = 1.0
    lr_basis_factor = 1.0

    last_upsamp_step = args.init_iters

    if args.enable_random:
        warn("Randomness is enabled for training (normal for LLFF & scenes with background)")

    epoch_id = -1

    first_vid_path = Path(args.train_dir) / "video_00000.mp4"
    render_video(grid,
                 resample_cameras[:60],   # 60 poses ≈ 5 s @ 12 fps
                 first_vid_path,
                 fps=12, crop=1) 

    # Training loop
    while True:
        dset.shuffle_rays()
        epoch_id += 1
        epoch_size = dset.rays.origins.size(0)
        batches_per_epoch = (epoch_size-1)//args.batch_size+1
        
        def train_step():
            print('Train step')
            pbar = tqdm(enumerate(range(0, epoch_size, args.batch_size)), total=batches_per_epoch)
            stats = {"mse" : 0.0, "psnr" : 0.0, "invsqr_mse" : 0.0}
            for iter_id, batch_begin in pbar:
                gstep_id = iter_id + gstep_id_base
                if args.lr_fg_begin_step > 0 and gstep_id == args.lr_fg_begin_step:
                    grid.density_data.data[:] = args.init_sigma
                lr_sigma = lr_sigma_func(gstep_id) * lr_sigma_factor
                lr_sh = lr_sh_func(gstep_id) * lr_sh_factor
                lr_basis = lr_basis_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
                lr_sigma_bg = lr_sigma_bg_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
                lr_color_bg = lr_color_bg_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
                if not args.lr_decay:
                    lr_sigma = args.lr_sigma * lr_sigma_factor
                    lr_sh = args.lr_sh * lr_sh_factor
                    lr_basis = args.lr_basis * lr_basis_factor

                batch_end = min(batch_begin + args.batch_size, epoch_size)
                batch_origins = dset.rays.origins[batch_begin: batch_end]
                batch_dirs = dset.rays.dirs[batch_begin: batch_end]
                rgb_gt = dset.rays.gt[batch_begin: batch_end]
                rays = svox2.Rays(batch_origins, batch_dirs)

                rgb_pred = grid.volume_render_fused(rays, rgb_gt,
                        beta_loss=args.lambda_beta,
                        sparsity_loss=args.lambda_sparsity,
                        randomize=args.enable_random)

                mse = F.mse_loss(rgb_gt, rgb_pred)

                # Stats
                mse_num : float = mse.detach().item()
                psnr = -10.0 * math.log10(mse_num)
                stats['mse'] += mse_num
                stats['psnr'] += psnr
                stats['invsqr_mse'] += 1.0 / mse_num ** 2

                if (iter_id + 1) % args.print_every == 0:
                    # Print averaged stats
                    pbar.set_description(f'obj {object_id}/{total_objects} epoch {epoch_id} psnr={psnr:.2f}')
                    for stat_name in stats:
                        stat_val = stats[stat_name] / args.print_every
                        summary_writer.add_scalar(stat_name, stat_val, global_step=gstep_id)
                        stats[stat_name] = 0.0

                    summary_writer.add_scalar("lr_sh", lr_sh, global_step=gstep_id)
                    summary_writer.add_scalar("lr_sigma", lr_sigma, global_step=gstep_id)
                    if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                        summary_writer.add_scalar("lr_basis", lr_basis, global_step=gstep_id)
                    if grid.use_background:
                        summary_writer.add_scalar("lr_sigma_bg", lr_sigma_bg, global_step=gstep_id)
                        summary_writer.add_scalar("lr_color_bg", lr_color_bg, global_step=gstep_id)

                    if args.weight_decay_sh < 1.0:
                        grid.sh_data.data *= args.weight_decay_sigma
                    if args.weight_decay_sigma < 1.0:
                        grid.density_data.data *= args.weight_decay_sh

                # Apply TV/Sparsity regularizers
                if args.lambda_tv > 0.0:
                    grid.inplace_tv_grad(grid.density_data.grad,
                            scaling=args.lambda_tv,
                            sparse_frac=args.tv_sparsity,
                            logalpha=args.tv_logalpha,
                            ndc_coeffs=dset.ndc_coeffs,
                            contiguous=args.tv_contiguous)
                if args.lambda_tv_sh > 0.0:
                    grid.inplace_tv_color_grad(grid.sh_data.grad,
                            scaling=args.lambda_tv_sh,
                            sparse_frac=args.tv_sh_sparsity,
                            ndc_coeffs=dset.ndc_coeffs,
                            contiguous=args.tv_contiguous)
                if args.lambda_tv_lumisphere > 0.0:
                    grid.inplace_tv_lumisphere_grad(grid.sh_data.grad,
                            scaling=args.lambda_tv_lumisphere,
                            dir_factor=args.tv_lumisphere_dir_factor,
                            sparse_frac=args.tv_lumisphere_sparsity,
                            ndc_coeffs=dset.ndc_coeffs)
                if args.lambda_l2_sh > 0.0:
                    grid.inplace_l2_color_grad(grid.sh_data.grad,
                            scaling=args.lambda_l2_sh)
                if grid.use_background and (args.lambda_tv_background_sigma > 0.0 or args.lambda_tv_background_color > 0.0):
                    grid.inplace_tv_background_grad(grid.background_data.grad,
                            scaling=args.lambda_tv_background_color,
                            scaling_density=args.lambda_tv_background_sigma,
                            sparse_frac=args.tv_background_sparsity,
                            contiguous=args.tv_contiguous)
                if args.lambda_tv_basis > 0.0:
                    tv_basis = grid.tv_basis()
                    loss_tv_basis = tv_basis * args.lambda_tv_basis
                    loss_tv_basis.backward()

                # Manual SGD/rmsprop step
                if gstep_id >= args.lr_fg_begin_step:
                    grid.optim_density_step(lr_sigma, beta=args.rms_beta, optim=args.sigma_optim)
                    grid.optim_sh_step(lr_sh, beta=args.rms_beta, optim=args.sh_optim)
                if grid.use_background:
                    grid.optim_background_step(lr_sigma_bg, lr_color_bg, beta=args.rms_beta, optim=args.bg_optim)
                if gstep_id >= args.lr_basis_begin_step:
                    if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                        grid.optim_basis_step(lr_basis, beta=args.rms_beta, optim=args.basis_optim)
                    elif grid.basis_type == svox2.BASIS_TYPE_MLP:
                        optim_basis_mlp.step()
                        optim_basis_mlp.zero_grad()

        train_step()
        gc.collect()
        
        # Render video after each epoch
        step_vid_path = Path(args.train_dir) / f"video_{epoch_id:06d}.mp4"
        render_video(grid, resample_cameras[:60], step_vid_path, fps=12, crop=1.0)
        
        gstep_id_base += batches_per_epoch

        # Save periodic checkpoints
        if args.save_every > 0 and (epoch_id) % max(factor, args.save_every) == 0 and not args.tune_mode:
            print('Saving', ckpt_path)
            grid.save(ckpt_path)

        # Upsampling
        if (gstep_id_base - last_upsamp_step) >= args.upsamp_every:
            last_upsamp_step = gstep_id_base
            if reso_id < len(reso_list) - 1:
                print('* Upsampling from', reso_list[reso_id], 'to', reso_list[reso_id + 1])
                if args.tv_early_only > 0:
                    print('turning off TV regularization')
                    args.lambda_tv = 0.0
                    args.lambda_tv_sh = 0.0
                elif args.tv_decay != 1.0:
                    args.lambda_tv *= args.tv_decay
                    args.lambda_tv_sh *= args.tv_decay

                reso_id += 1
                use_sparsify = True
                z_reso = reso_list[reso_id] if isinstance(reso_list[reso_id], int) else reso_list[reso_id][2]
                grid.resample(reso=reso_list[reso_id],
                        sigma_thresh=args.density_thresh,
                        weight_thresh=args.weight_thresh / z_reso if use_sparsify else 0.0,
                        dilate=2,
                        cameras=resample_cameras if args.thresh_type == 'weight' else None,
                        max_elements=args.max_grid_elements)

                if grid.use_background and reso_id <= 1:
                    grid.sparsify_background(args.background_density_thresh)

                if args.upsample_density_add:
                    grid.density_data.data[:] += args.upsample_density_add

            if factor > 1 and reso_id < len(reso_list) - 1:
                print('* Using higher resolution images due to large grid; new factor', factor)
                factor //= 2
                dset.gen_rays(factor=factor)
                dset.shuffle_rays()

        # Check if training is complete
        if gstep_id_base >= args.n_iters:
            print(f'* Final save for object {object_id}/{total_objects}')
            object_stop_time = datetime.now()
            secs = (object_stop_time - object_start_time).total_seconds()
            timings_file = open(os.path.join(args.train_dir, 'time_mins.txt'), 'a')
            timings_file.write(f"{secs / 60}\n")
            
            # Save comprehensive checkpoint with dense grid and original sparse grid
            if not args.tune_nosave:
                # Save original sparse grid
                print('Saving original sparse grid...')
                grid.save(ckpt_path)
                
                # Generate and save dense grid
                print('Generating and saving dense grid...')
                dense_grid = get_dense_density_sh(grid)
                dense_grid_path = path.join(args.train_dir, 'dense_grid.npz')
                np.savez_compressed(dense_grid_path, 
                                   dense_grid=dense_grid.cpu().numpy(),
                                   shape=dense_grid.shape,
                                   center=grid.center.cpu().numpy(),
                                   radius=grid.radius.cpu().numpy())
                
                # Save training metadata
                metadata = {
                    'object_id': object_id,
                    'object_name': object_name,
                    'data_dir': data_dir,
                    'final_resolution': reso_list[reso_id],
                    'training_time_minutes': secs / 60,
                    'final_epoch': epoch_id,
                    'final_step': gstep_id_base
                }
                metadata_path = path.join(args.train_dir, 'training_metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f'✅ Saved comprehensive checkpoint for object {object_id}/{total_objects}:')
                print(f'  - Sparse grid: {ckpt_path}')
                print(f'  - Dense grid: {dense_grid_path}')
                print(f'  - Metadata: {metadata_path}')
            break

    # Final video with all cameras
    final_vid_path = Path(args.train_dir) / f"video_final_{gstep_id_base:06d}.mp4"
    render_video(grid, resample_cameras, final_vid_path, fps=12, crop=1.0)
    
    # Cleanup
    summary_writer.close()
    del grid, dset
    torch.cuda.empty_cache()
    gc.collect()
    
    return args.train_dir


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train SVox2 models on ShapeNet chairs with split processing')
    
    parser.add_argument('--split', type=int, default=0,
                        help='Current split index (0-based)')
    parser.add_argument('--total_splits', type=int, default=1,
                        help='Total number of splits')
    parser.add_argument('--max_objects', type=int, default=None,
                        help='Maximum number of objects to process (for testing)')
    parser.add_argument('--base_data_dir', type=str, 
                        default="/om/user/akiruga/datasets/srn_chairs_alternate_views",
                        help='Base directory containing all object subdirectories')
    parser.add_argument('--base_output_dir', type=str,
                        default='/om/user/akiruga/svox2/data/ckpts/shapenet_chairs_all_jupyter',
                        help='Base output directory for all trained models')
    
    return parser.parse_args()


# Main execution: iterate over split subset of objects
if __name__ == "__main__":
    # Parse command line arguments
    cli_args = parse_args()
    
    # Find all ShapeNet chair object directories
    all_object_dirs = glob.glob(os.path.join(cli_args.base_data_dir, "*/viz"))
    all_object_dirs.sort()  # Ensure consistent ordering across runs
    
    print(f"Found {len(all_object_dirs)} total ShapeNet chair objects")
    
    # Apply max_objects limit if specified (for testing)
    if cli_args.max_objects is not None and isinstance(cli_args.max_objects, int):
        print(f"Using max_objects limit: {cli_args.max_objects}")
        all_object_dirs = all_object_dirs[:cli_args.max_objects]
        print(f"Limited to {len(all_object_dirs)} objects")
    
    # Calculate split boundaries (same logic as the hydra example)
    size_per_split = len(all_object_dirs) // cli_args.total_splits
    remainder = len(all_object_dirs) % cli_args.total_splits
    cur_size = size_per_split
    if cli_args.split == cli_args.total_splits - 1: 
        cur_size += remainder
    
    # Get objects for current split
    start_idx = cli_args.split * size_per_split
    end_idx = start_idx + cur_size
    current_split_dirs = all_object_dirs[start_idx:end_idx]
    
    print(f"Processing split {cli_args.split + 1}/{cli_args.total_splits}")
    print(f"Objects in this split: {len(current_split_dirs)} (indices {start_idx} to {end_idx-1})")
    
    # Filter out already trained objects
    remaining_dirs = []
    skipped_count = 0
    
    for data_dir in current_split_dirs:
        object_name = os.path.basename(os.path.dirname(data_dir))
        if is_object_already_trained(cli_args.base_output_dir, object_name):
            print(f"⏭️  Skipping already trained object: {object_name}")
            skipped_count += 1
        else:
            remaining_dirs.append(data_dir)
    
    print(f"Found {skipped_count} already trained objects in this split")
    print(f"Remaining objects to train: {len(remaining_dirs)}")
    
    if len(remaining_dirs) == 0:
        print("🎉 All objects in this split are already trained!")
        exit(0)
    
    # Get default training arguments and update paths
    args = get_default_args()
    args.base_train_dir = cli_args.base_output_dir
    
    # Train remaining objects
    completed_objects = []
    failed_objects = []
    
    global_start_time = datetime.now()
    
    for i, data_dir in enumerate(remaining_dirs):
        object_name = os.path.basename(os.path.dirname(data_dir))
        try:
            print(f"\n{'='*80}")
            print(f"TRAINING OBJECT {i+1}/{len(remaining_dirs)} (Split {cli_args.split + 1}/{cli_args.total_splits})")
            print(f"Object: {object_name}")
            print(f"{'='*80}")
            
            start_time = time.time()
            object_dir = train_single_object(data_dir, i+1, len(remaining_dirs), args)
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            completed_objects.append((data_dir, object_dir))
            print(f"✅ Finished {object_name} in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            print(f"❌ Failed to train object {object_name}: {e}")
            failed_objects.append((data_dir, str(e)))
            continue
    
    # Final summary for this split
    global_end_time = datetime.now()
    total_time = (global_end_time - global_start_time).total_seconds()
    
    print(f"\n{'='*80}")
    print(f"SPLIT {cli_args.split + 1}/{cli_args.total_splits} SUMMARY")
    print(f"{'='*80}")
    print(f"📋 Total objects in split: {len(current_split_dirs)}")
    print(f"⏭️  Already trained (skipped): {skipped_count}")
    print(f"🎯 Attempted to train: {len(remaining_dirs)}")
    print(f"✅ Successfully trained: {len(completed_objects)}")
    print(f"❌ Failed: {len(failed_objects)}")
    print(f"⏱️  Split training time: {total_time/3600:.2f} hours")
    
    # Save split-specific summary
    split_summary = {
        'split': cli_args.split,
        'total_splits': cli_args.total_splits,
        'total_objects_in_split': len(current_split_dirs),
        'already_trained_skipped': skipped_count,
        'attempted_to_train': len(remaining_dirs),
        'completed': len(completed_objects),
        'failed': len(failed_objects),
        'split_time_hours': total_time / 3600,
        'completed_objects': [{'data_dir': d, 'output_dir': o} for d, o in completed_objects],
        'failed_objects': [{'data_dir': d, 'error': e} for d, e in failed_objects]
    }
    
    os.makedirs(args.base_train_dir, exist_ok=True)
    split_summary_path = path.join(args.base_train_dir, f'split_{cli_args.split:03d}_summary.json')
    with open(split_summary_path, 'w') as f:
        json.dump(split_summary, f, indent=2)
    
    print(f"📋 Split summary saved to: {split_summary_path}")
    print(f"🎉 Split {cli_args.split + 1}/{cli_args.total_splits} complete!")