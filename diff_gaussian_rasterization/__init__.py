#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    means2D_densify,
    shift_factors,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        means2D_densify,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings.viewmatrix,
        raster_settings.campos,
        raster_settings.projmatrix,
        shift_factors,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        means2D_densify,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        world_view,
        camera_center,
        full_proj,
        shift_factors,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            shift_factors,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            world_view,
            full_proj,
            raster_settings.intrinsic,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            camera_center,
            raster_settings.prefiltered,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, weights, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, weights, radii, geomBuffer, binningBuffer, imgBuffer, means2Dx, means2Dy = _C.rasterize_gaussians(*args)

        mean2D = torch.cat((means2Dx, means2Dy), dim=-1)
        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.camera_center = camera_center
        ctx.full_proj = full_proj
        ctx.world_view = world_view
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, weights)
        return color, radii, depth, weights, mean2D

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth, grad_out_weight, grad_mean2D):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        camera_center = ctx.camera_center
        full_proj = ctx.full_proj
        world_view = ctx.world_view
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, weights = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D,
                radii,
                colors_precomp,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                #raster_settings.viewmatrix,
                #raster_settings.projmatrix,
                world_view,
                full_proj,
                raster_settings.intrinsic,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.image_height,
                raster_settings.image_width,
                grad_out_color,
                grad_out_depth,
                grad_out_weight,
                sh,
                raster_settings.sh_degree,
                #raster_settings.campos,
                camera_center,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                weights,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_means2D_densify, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_camera_center, grad_full_proj, grad_world_view, grad_shift_factors, covariance = _C.rasterize_gaussians_backward(*args)

        print("in __init__ gradient: ")
        print(grad_means3D[898411])
        # remember to set three gradient to 0 if recovering original 3dgs
        #grad_camera_center = torch.zeros(3).cuda()
        #grad_full_proj = torch.zeros(4, 4).cuda()
        #grad_world_view = torch.zeros(4, 4).cuda()

        # reshape grad_full_proj from 16 to 4x4
        grad_full_proj = grad_full_proj[:, None].reshape(4, 4)
        grad_world_view = grad_world_view[:, None].reshape(4, 4)
        # print the gradient scale of three chains
        #if raster_settings.debug_iter in [10000, 20000, 30000, 40000]:
        #    print(grad_camera_center)
        #    print(grad_world_view)
        #    print(grad_full_proj)

        #grad_world_view[0][0] = 1.
        #if raster_settings.debug_iter in [1001, 1002]:
        #    print("*"*20)
        #    grad_camera_center = torch.zeros(3).cuda()
        #    print(grad_camera_center)
        #    print(grad_full_proj)
        #    print(grad_world_view)
        #    print("*"*20)

        #print(covariance.max())
        #print(covariance.min())
        #print(sum((covariance != 0) == True))
        #import pdb;pdb.set_trace()

        grads = (
            grad_means3D,
            grad_means2D,
            grad_means2D_densify,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_world_view,
            grad_camera_center,
            grad_full_proj,
            grad_shift_factors,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    intrinsic : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    debug_iter : int

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)

        return visible

    def forward(self, means3D, means2D, means2D_densify, shift_factors, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            means2D_densify,
            shift_factors,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings
        )

def compute_relocation(opacity_old, scale_old, N, binoms, n_max):
    new_opacity, new_scale = _C.compute_relocation(opacity_old, scale_old, N.int(), binoms, n_max)
    return new_opacity, new_scale
