/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/utils.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& shift_factors,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const torch::Tensor& intrinsic,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_alpha = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor means2Dx = torch::full({P, 1}, 0, means3D.options());
  torch::Tensor means2Dy = torch::full({P, 1}, 0, means3D.options());
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		intrinsic.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		shift_factors.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_depth.contiguous().data<float>(),
		out_alpha.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		means2Dx.contiguous().data<float>(),
		means2Dy.contiguous().data<float>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, out_depth, out_alpha, radii, geomBuffer, binningBuffer, imgBuffer, means2Dx, means2Dy);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const torch::Tensor& intrinsic,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dout_depth,
    const torch::Tensor& dL_dout_alpha,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const torch::Tensor& alphas,
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D_densify = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_ddepths = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor covariance = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dprojmatrix = torch::zeros({P, 16}, means3D.options());
  torch::Tensor dL_dviewmatrix = torch::zeros({P, 16}, means3D.options());
  torch::Tensor dL_dcampos = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dshift = torch::zeros({P, 3}, means3D.options());
  
  //float data[] = { 1, 2, 3,
  //               4, 5, 6 };
  //torch::Tensor test = torch::from_blob(data, {2, 3});
  //float weight[] = {1, 2};
  //torch::Tensor w = torch::from_blob(weight, {2});
  //std::cout << test;
  //std::cout << test.sizes();
  //std::cout << test.mean(1);
  //std::cout << test.sum(1);
  //std::cout << test.sum(0);
  //std::cout << test.sum();
  //std::cout << w / w.sum(0);
  //std::cout << "*****************************************************\n" << std::endl;
  //std::cout << "DO NOT tried to access the contiguous().data<>(), probably because the pointer is on cpu but our original vairble is create on GPU\n" << std::endl;
  //std:: cout << means3D.options() << std::endl;
  //std:: cout << campos.contiguous() << std::endl;
  //std::cout << "abc" << std::endl;
  //std:: cout << campos.contiguous().data<float>() << std::endl;
  //std:: cout << intrinsic << std::endl;
  //std::cout << "*****************************************************\n" << std::endl;
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  alphas.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  intrinsic.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  image_height,
	  image_width,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_depth.contiguous().data<float>(),
	  dL_dout_alpha.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
      dL_dmeans2D_densify.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  covariance.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_ddepths.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  dL_dprojmatrix.contiguous().data<float>(),
	  dL_dviewmatrix.contiguous().data<float>(),
	  dL_dcampos.contiguous().data<float>(),
	  dL_dshift.contiguous().data<float>(),
	  debug);

  }

  std::cout << "+++++++++" << std::endl;
  std::cout << "accumulate" << std::endl;
  std::cout << (dL_dmeans3D[898411]) << std::endl;
  std::cout << "+++++++++" << std::endl;
  //std::cout << "*****************************************************\n" << std::endl;
  // check which points are not processed by cuda kernel
  //std::cout << "*****************************************************\n" << std::endl;
  //std::cout << (dL_dcampos[0][0].item<bool>()) << std::endl;
  //std::cout << (!dL_dcampos[0][0].item<bool>()) << std::endl;
  //std::cout << (dL_dcampos[0][0].item<int>() != int(1)) << std::endl;
  //for (int i=0; i < 13477; i++) {
  //    if (!dL_dcampos[i][0].item<bool>()) {
  //        std::cout << dL_dcampos[i] << std::endl;
  //        std::cout << "pos:" << i << std::endl;
  //    }
  //}
  //std::cout << "*****************************************************\n" << std::endl;

  //std::cout << (covariance / covariance.sum()).sizes();
  //std::cout << (covariance / covariance.sum()).sum();
  //std::cout << (dL_dviewmatrix)[0];
  //std::cout << (covariance / covariance.sum())[0];
  //std::cout << (dL_dviewmatrix * (covariance / covariance.sum()))[0];
  //return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations, (dL_dcampos * (covariance / covariance.sum())).sum(0), (dL_dprojmatrix * (covariance / covariance.sum())).sum(0), (dL_dviewmatrix * (covariance / covariance.sum())).sum(0), covariance + 1e-4f);
  return std::make_tuple(dL_dmeans2D, dL_dmeans2D_densify, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations, dL_dcampos.mean(0), dL_dprojmatrix.mean(0), dL_dviewmatrix.mean(0), dL_dshift.mean(0), covariance + 1e-4f);
  //return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations, (dL_dcampos * ((1e-2f / (covariance + 1e-4f)) / (1e-2f / (covariance + 1e-4f)).sum())).sum(0), (dL_dprojmatrix * ((1e-2f / (covariance + 1e-4f)) / (1e-2f / (covariance + 1e-4f)).sum())).sum(0), (dL_dviewmatrix * ((1e-2f / (covariance + 1e-4f)) / (1e-2f / (covariance + 1e-4f)).sum())).sum(0), covariance);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}

std::tuple<torch::Tensor, torch::Tensor> ComputeRelocationCUDA(
	torch::Tensor& opacity_old,
	torch::Tensor& scale_old,
	torch::Tensor& N,
	torch::Tensor& binoms,
	const int n_max)
{
	const int P = opacity_old.size(0);

	torch::Tensor final_opacity = torch::full({P}, 0, opacity_old.options().dtype(torch::kFloat32));
	torch::Tensor final_scale = torch::full({3 * P}, 0, scale_old.options().dtype(torch::kFloat32));

	if(P != 0)
	{
		UTILS::ComputeRelocation(P,
			opacity_old.contiguous().data<float>(),
			scale_old.contiguous().data<float>(),
			N.contiguous().data<int>(),
			binoms.contiguous().data<float>(),
			n_max,
			final_opacity.contiguous().data<float>(),
			final_scale.contiguous().data<float>());
	}

	return std::make_tuple(final_opacity, final_scale);

}
