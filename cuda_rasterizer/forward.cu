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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped, const float* viewmatrix, float entrance_pupil_shift)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::mat4 w2c = glm::mat4(
		viewmatrix[0], viewmatrix[4], viewmatrix[8], viewmatrix[12],
		viewmatrix[1], viewmatrix[5], viewmatrix[9], viewmatrix[13],
		viewmatrix[2], viewmatrix[6], viewmatrix[10], viewmatrix[14],
		viewmatrix[3], viewmatrix[7], viewmatrix[11], viewmatrix[15]);
    glm::mat4 c2w = glm::inverse(w2c);
    glm::vec4 pos_world = glm::vec4(pos[0], pos[1], pos[2], 1.);
    glm::vec4 pos_camera = pos_world * w2c;
    pos_camera[2] += entrance_pupil_shift;
    glm::vec4 pos_world_shift = pos_camera * c2w;
    pos[0] = pos_world_shift[0];
    pos[1] = pos_world_shift[1];
    pos[2] = pos_world_shift[2];
	glm::vec3 dir = pos - campos;

    // check the length function and dir xyz
    //if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //    printf("*********************************\n");
    //    printf("glm::length function");
    //    printf("%f\n", glm::length(dir));
    //    printf("%f\n", dir.x);
    //    printf("%f\n", dir.y);
    //    printf("%f\n", dir.z);
    //    printf("%d\n", deg);
    //    printf("*********************************\n");
    //}
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix, float entrance_pupil_shift)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);
    t.z += entrance_pupil_shift;

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);
	//glm::mat3 J = glm::mat3(
    //    focal_x / t.z, 0.0f, 0,
	//	0.0f, focal_y / t.z, 0,
	//	-(focal_x * t.x) / (t.z * t.z), -(focal_y * t.y) / (t.z * t.z), 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);
	//glm::mat3 W = glm::mat3(
	//	viewmatrix[0], viewmatrix[1], viewmatrix[2],
	//	viewmatrix[4], viewmatrix[5], viewmatrix[6],
	//	viewmatrix[8], viewmatrix[9], viewmatrix[10]);

    //  the order of rotation part from viewmatrix is the inversion of original viewmatrix
    //if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //    printf("*********************************\n");
    //    printf("viewmatrix\n");
    //    for (int i = 0; i < 3; i++) {
    //        for (int j = 0; j < 3; j++) {
    //            if (j < 2) 
    //                printf("%f ", W[i][j]);
    //            else
    //                printf("%f\n", W[i][j]);
    //        }
    //    }
    //    printf("*********************************\n");
    //}

	glm::mat3 T = W * J;
	//glm::mat3 T = J * W;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;
	//glm::mat3 cov = T * Vrk * glm::transpose(T);

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Bilinear Interpolation for displacement
__device__ float2 bilinearInterpolateKernel(int x, int y, const int res_u, const float* u_distortion, const float* v_distortion, float xp, float yp)
{
    // Assuming u and v are arrays containing the u and v displacements for the corners
    // u[0], v[0] for (x, y)
    // u[1], v[1] for (x+1, y)
    // u[2], v[2] for (x, y+1)
    // u[3], v[3] for (x+1, y+1)

    // Compute bilinear coefficients
    float A = (x + 1 - xp) * (y + 1 - yp);
    float B = (xp - x) * (y + 1 - yp);
    float C = (x + 1 - xp) * (yp - y);
    float D = (xp - x) * (yp - y);

    // Interpolate u
    float up = A * u_distortion[x + y * res_u] + B * u_distortion[x + 1 + y * res_u] + C * u_distortion[x + (y + 1) * res_u] + D * u_distortion[x + 1 + (y + 1) * res_u];

    // Interpolate v
    float vp = A * v_distortion[x + y * res_u] + B * v_distortion[x + 1 + y * res_u] + C * v_distortion[x + (y + 1) * res_u] + D * v_distortion[x + 1 + (y + 1) * res_u];

    // check interpolation weights
    //if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //    printf("*********************************\n");
    //    printf("%d\n", x);
    //    printf("%d\n", y);
    //    printf("%f\n", xp);
    //    printf("%f\n", yp);
    //    printf("%f\n", A);
    //    printf("%f\n", B);
    //    printf("%f\n", C);
    //    printf("%f\n", D);
    //    printf("%f\n", A + B + C + D);
    //    printf("%f\n", up);
    //    printf("%f\n", vp);
    //    printf("*********************************\n");
    //}

	return {up, vp};
}

// Bilinear Interpolation for control points
__device__ float2 bilinearInterpolateControlPoints(int x, int y, const int res_u, const float* control_points, float xp, float yp)
{
    // Assuming u and v are arrays containing the u and v displacements for the corners
    // u[0], v[0] for (x, y)
    // u[1], v[1] for (x+1, y)
    // u[2], v[2] for (x, y+1)
    // u[3], v[3] for (x+1, y+1)

    // Compute bilinear coefficients
    float A = (x + 1 - xp) * (y + 1 - yp);
    float B = (xp - x) * (y + 1 - yp);
    float C = (x + 1 - xp) * (yp - y);
    float D = (xp - x) * (yp - y);

    // Interpolate u
    float up = A * control_points[2 * (x + y * res_u) + 0] + B * control_points[2 * (x + 1 + y * res_u) + 0] + C * control_points[2 * (x + (y + 1) * res_u) + 0] + D * control_points[2 * (x + 1 + (y + 1) * res_u) + 0];

    // Interpolate v
    float vp = A * control_points[2 * (x + y * res_u) + 1] + B * control_points[2 * (x + 1 + y * res_u) + 1] + C * control_points[2 * (x + (y + 1) * res_u) + 1] + D * control_points[2 * (x + 1 + (y + 1) * res_u) + 1];

	return {up, vp};
}


// Omnidirectional camera distortion
__device__ float3 omnidirectionalDistortion_OPENCV(float2 ab, float z, const float* affine_coeff, const float* poly_coeff) {
    float inv_r    = 1 / sqrt(ab.x * ab.x + ab.y * ab.y);
    float theta    = atan(sqrt(ab.x * ab.x + ab.y * ab.y));
    float theta2   = theta * theta;
    float theta4   = theta2 * theta2;
    float theta6   = theta4 * theta2;
    float theta8   = theta4 * theta4;
    float rho      = theta * (1 + poly_coeff[0] * theta2 + poly_coeff[1] * theta4 + poly_coeff[2] * theta6 + poly_coeff[3] * theta8);
    //float rho      = theta * (1 + poly_coeff[0] * theta2 * 0 + poly_coeff[1] * theta4 * 0 + poly_coeff[2] * theta6 * 0 + poly_coeff[3] * theta8 * 0);

    //float e  = affine_coeff[1];
    //float d  = affine_coeff[2];
    //float c  = affine_coeff[3];
    //float cx = affine_coeff[4];
    //float cy = affine_coeff[5];

    //float dist_x =     ab.x * rho + e * ab.y * rho + cx;
    //float dist_y = d * ab.x * rho + c * ab.y * rho + cy;
    
    return {rho * inv_r * ab.x * z, rho * inv_r * ab.y * z, z};
}

// Omnidirectional camera distortion
__device__ float3 omnidirectionalDistortion(float2 ab, float z, const float* affine_coeff, const float* poly_coeff) {
    float inv_norm = 1 / sqrt(ab.x * ab.x + ab.y * ab.y);
    float theta    = atan(sqrt(ab.x * ab.x + ab.y * ab.y));
    float rho      = poly_coeff[0] + poly_coeff[1] * theta + poly_coeff[2] * theta * theta;

    float e  = affine_coeff[1];
    float d  = affine_coeff[2];
    float c  = affine_coeff[3];
    float cx = affine_coeff[4];
    float cy = affine_coeff[5];

    float dist_x =     ab.x * rho + e * ab.y * rho + cx;
    float dist_y = d * ab.x * rho + c * ab.y * rho + cy;
    
    //if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //    printf("*********************************\n");
    //    printf("%f\n", e);
    //    printf("%f\n", d);
    //    printf("%f\n", c);
    //    printf("%f\n", cx);
    //    printf("%f\n", cy);
    //    printf("*********************************\n");
    //}
    return {dist_x * z, dist_y * z, z};
    //return {ab.x * z, ab.y * z, z};
}

// Apply neuralens
__device__ float3 applyNeuralens(float2 ab, float z, int res_u, int res_v, const float* u_distortion, const float* v_distortion) {
    int u_idx = int((ab.x + 1) * (res_u / 2));
    int v_idx = int((ab.y + 1) * (res_v / 2));
    float2 uv_displacement;

    if (u_idx > 0 && u_idx < (res_u - 1) && v_idx > 0 && v_idx < (res_v - 1)) {
        uv_displacement = bilinearInterpolateKernel(u_idx, v_idx, res_u, u_distortion, v_distortion, (ab.x + 1) * (res_u / 2), (ab.y + 1) * (res_v / 2));
    }

    ab.x = ab.x + uv_displacement.x;
    ab.y = ab.y + uv_displacement.y;

    return {ab.x * z, ab.y * z, z};
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* intrinsic,
	const float* shift_factors,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* means2Dx,
	float* means2Dy,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);

    // calculate entrance pupil shift based on pts under camera coordinate instead of pixel space (should not use projmatrix)
	//float3 t = transformPoint4x3(p_orig, projmatrix);
    //float2 ab = {t.x / t.z, t.y / t.z};
    // maybe projmatrix is better... before intrinsic is to small to observe
    float2 ab = {p_hom.x / p_hom.w, p_hom.y / p_hom.w};
    float radius = sqrt(ab.x * ab.x + ab.y * ab.y);
    float theta = atan(radius);
    float theta2 = theta * theta;
    float theta3 = theta2 * theta;
    float theta5 = theta2 * theta3;
    float theta7 = theta2 * theta5;
    //float D_n = 1 - shift_factors[0] * theta3 - shift_factors[1] * theta5 - shift_factors[2] * theta7;
    //float D_n_prime = -3 * shift_factors[0] * theta2 - 5 * shift_factors[1] * theta2 * theta2 - 7 * shift_factors[2] * theta5 * theta;
    //float up = radius / D_n - tan(theta);
    //float down = (-radius * D_n_prime) / (D_n * D_n) - 1 / (cos(theta) * cos(theta));
    //if (threadIdx.x == 100 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //    for (int i=0; i<3; i++) {
    //        printf("%f\n", theta);
    //        theta = theta - up / down;
    //        printf("%f\n", up/down);
    //        printf("%f\n", theta);
    //        printf("++++++++++++\n");
    //        theta2 = theta * theta;
    //        theta3 = theta2 * theta;
    //        theta5 = theta2 * theta3;
    //        theta7 = theta2 * theta5;
    //        D_n = 1 - shift_factors[0] * theta3 - shift_factors[1] * theta5 - shift_factors[2] * theta7;
    //        D_n_prime = -3 * shift_factors[0] * theta2 - 5 * shift_factors[1] * theta2 * theta2 - 7 * shift_factors[2] * theta5 * theta;
    //        up = radius / D_n - tan(theta);
    //        down = (-radius * D_n_prime) / (D_n * D_n) - 1 / (cos(theta) * cos(theta));
    //    }
    //    printf("*******************\n");
    //}

    float entrance_pupil_shift = shift_factors[0] * theta3 + shift_factors[1] * theta5 + shift_factors[2] * theta7;

    p_hom.w += entrance_pupil_shift;

    // implement only for llff
    //float2 ab = {p_w2c.x / p_w2c.z, p_w2c.y / p_w2c.z};
    //float x2 = ab.x * ab.x;
    //float y2 = ab.y * ab.y;
    //float r2 = x2 + y2;
    //if (r2 < 2) {
    //    // room
    //    //float wtf = -0.5882877287895939 * r2 + 0.40174065051468477 * r2 * r2;
    //    // fern
    //    float wtf = -0.8794442879124985 * r2 + 1.0081900208012675 * r2 * r2;
    //    p_w2c = {ab.x * (1 + wtf) * p_w2c.z, ab.y * (1 + wtf) * p_w2c.z, p_w2c.z};
    //}



    // forward of control points
    //---------------------------------------------------------------//
    //float2 ab = {p_w2c.x / p_w2c.z, p_w2c.y / p_w2c.z};
    //float ab_u = ((ab.x + boundary_original_points[0]) / (2. * boundary_original_points[0])) * (res_control_points_u - 1);
    //float ab_v = ((ab.y + boundary_original_points[1]) / (2. * boundary_original_points[1])) * (res_control_points_v - 1);
    //int u_idx_ = int(ab_u);
    //int v_idx_ = int(ab_v);

    //float2 mapping_point;
    //if (ab.x > -boundary_original_points[0] && ab.x < boundary_original_points[0] && ab.y > -boundary_original_points[1] && ab.y < boundary_original_points[1]) {
    //    mapping_point = bilinearInterpolateControlPoints(u_idx_, v_idx_, res_control_points_u, control_points, ab_u, ab_v);
    //    p_w2c = {mapping_point.x * p_w2c.z, mapping_point.y * p_w2c.z, p_w2c.z};

    //    //// check llff
    //    //float x2 = ab.x * ab.x;
    //    //float y2 = ab.y * ab.y;
    //    //float r2 = x2 + y2;
    //    //float wtf = -0.6952201430141931 * r2 + 0.5931150189831125 * r2 * r2;
    //    //float3 test2_tmp = {ab.x * (1 + wtf) * p_w2c.z, ab.y * (1 + wtf) * p_w2c.z, p_w2c.z};
    //    //// check fisheye dataset
    //    ////float3 test2_tmp = omnidirectionalDistortion_OPENCV(ab, p_w2c.z, affine_coeff, poly_coeff);
    //    //float2 test2 = {test2_tmp.x / test2_tmp.z, test2_tmp.y / test2_tmp.z};
    //    //float2 test1 = bilinearInterpolateControlPoints(u_idx_, v_idx_, res_control_points_u, control_points, ab_u, ab_v);
    //    //if (threadIdx.x < 1000 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //    //    //printf("*********************************\n");
    //    //    //printf("%f\n", ab.x - mapping_point.x);
    //    //    //printf("%f\n", ab.y - mapping_point.y);
    //    //    //printf("*********************************\n");
    //    //    printf("%f\n", test1.x - test2.x);
    //    //    printf("%f\n", test1.y - test2.y);
    //    //    //printf("*********************************\n");
    //    //}
    //}
    //else {
    //    p_w2c = omnidirectionalDistortion_OPENCV(ab, p_w2c.z, affine_coeff, poly_coeff);
    //}
    //---------------------------------------------------------------//

    // forward of omnidirectional camera model
    //---------------------------------------------------------------//
    // implementation of omnidirectional camera model distortion
    // originally, gs first apply intrinsic (K) to p_w2c and use p_proj to get [x'/z', y'/z']
    // in omnidirectional camera, we first get [x/z, y/z] and then apply intrisic
    // Both are the same!
    //float2 ab = {p_w2c.x / p_w2c.z, p_w2c.y / p_w2c.z};
    //p_w2c = omnidirectionalDistortion_OPENCV(ab, p_w2c.z, affine_coeff, poly_coeff);
    //---------------------------------------------------------------//


    // forward of radial table
    //---------------------------------------------------------------//
    //float theta = atan(sqrt(ab.x * ab.x + ab.y * ab.y));
    //int x1 = int((theta / 1.57079632679) * 1000);
    //int x_2 = int((theta / 1.57079632679) * 1000) + 1;
    //float x = (theta / 1.57079632679) * 1000;
    //if (x1 < 1000 && x1 >= 0 && x_2 < 1000 && x_2 >= 0) {
    //    float y1 = radial[x1];
    //    float y_2 = radial[x_2];
    //    p_w2c.x = p_w2c.x * (y1 + ((x - x1) * (y_2 - y1)) / (x_2 - x1));
    //    p_w2c.y = p_w2c.y * (y1 + ((x - x1) * (y_2 - y1)) / (x_2 - x1));
    //}
    //---------------------------------------------------------------//
    

	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
    
    //
    // A test for x/z and y/z before or after intrinsic
    //float3 test = {p_w2c.x / p_w2c.z, p_w2c.y / p_w2c.z, 1};
	//float4 test_hom = transformPoint4x4(test, intrinsic);
    //if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //    printf("*********************************\n");
    //    printf("%f\n", test_hom.x);
    //    printf("%f\n", test_hom.y);
    //    printf("%f\n", test_hom.z);
    //    printf("%f\n", test_hom.w);
    //    printf("*********************************\n");
    //}
    //if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //    printf("*********************************\n");
    //    printf("%f\n", p_proj.x);
    //    printf("%f\n", p_proj.y);
    //    printf("%f\n", p_proj.z);
    //    printf("*********************************\n");
    //}

    // backward of grid
    //---------------------------------------------------------------//
    // pay a special attention to u_distortion index
    // u_distortion[u, v] represents the displacement at (u, v)
    // the index should be v * W + u
    //int u_idx = int((p_proj.x + 1) * (res_u / 2));
    //int v_idx = int((p_proj.y + 1) * (res_v / 2));
    //float2 uv_displacement;
    //float2 uv_radial;
    //
    //if (u_idx > 0 && u_idx < (res_u - 1) && v_idx > 0 && v_idx < (res_v - 1)) {
    //    uv_displacement = bilinearInterpolateKernel(u_idx, v_idx, res_u, u_distortion, v_distortion, (p_proj.x + 1) * (res_u / 2), (p_proj.y + 1) * (res_v / 2));
    //    uv_radial = bilinearInterpolateKernel(u_idx, v_idx, res_u, u_radial, v_radial, (p_proj.x + 1) * (res_u / 2), (p_proj.y + 1) * (res_v / 2));
    //}
    //
    //p_proj.x = p_proj.x * uv_radial.x + uv_displacement.x;
    //p_proj.y = p_proj.y * uv_radial.y + uv_displacement.y;

    // check bilinear interpolation position
    //if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //    printf("*********************************\n");
    //    printf("%f\n", p_proj.x);
    //    printf("%f\n", p_proj.y);
    //    printf("%f\n", (p_proj.x + 1) * 100);
    //    printf("%f\n", (p_proj.y + 1) * 100);
    //    printf("%d\n", u_idx);
    //    printf("%d\n", v_idx);
    //    printf("%d\n", res_u);
    //    printf("%d\n", res_v);
    //    printf("%f\n", u_distortion[0]);
    //    printf("%f\n", u_distortion[1]);
    //    printf("%f\n", u_distortion[u_idx + v_idx * res_u]);
    //    printf("%d\n", (u_idx + v_idx * res_u));
    //    printf("*********************************\n");
    //}
    //---------------------------------------------------------------//


	// forward of 8 params distortion
    //---------------------------------------------------------------//
    // Apply 2D distortion to p_proj
    //float k1 = distortion_params[0];
    //float k2 = distortion_params[1];
    //float k3 = distortion_params[2];
    //float k4 = distortion_params[3];
    //float k5 = distortion_params[4];
    //float k6 = distortion_params[5];
    //float p1 = distortion_params[6];
    //float p2 = distortion_params[7];
    //
    //float x2 = p_proj.x * p_proj.x;
    //float y2 = p_proj.y * p_proj.y;
    //float r2 = x2 + y2;
    //float _2xy = float(2) * p_proj.x * p_proj.y;

    //// The forward distortion fails if the points are too far away on the image plain
    //if (r2 < 2){
    //    float radial_u = float(1) + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    //    float radial_v = float(1) + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2;
    //    float radial   = (radial_u / radial_v);

    //    float tangentialX = p1 * _2xy + p2 * (r2 + float(2) * x2);
    //    float tangentialY = p1 * (r2 + float(2) * y2) + p2 * _2xy;

    //    p_proj.x = p_proj.x * radial + tangentialX;
    //    p_proj.y = p_proj.y * radial + tangentialY;
    //}
    //---------------------------------------------------------------//

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix, entrance_pupil_shift);
	//float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped, viewmatrix, entrance_pupil_shift);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	//depths[idx] = p_view.z;
    depths[idx] = sqrt((p_orig.x - (*cam_pos).x) * (p_orig.x - (*cam_pos).x) + (p_orig.y - (*cam_pos).y) * (p_orig.y - (*cam_pos).y) + (p_orig.z - (*cam_pos).z) * (p_orig.z - (*cam_pos).z));

	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	means2Dx[idx] = p_hom.z;
	means2Dy[idx] = p_hom.w;
	//means2Dx[idx] = ab.x - mapping_point.x;
	//means2Dy[idx] = ab.y - mapping_point.y;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
    const float* __restrict__ depth,
	float* __restrict__ out_depth,
	float* __restrict__ out_alpha)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
    __shared__ float collected_depth[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
    //float D = 15.0f;  // Median Depth. wtf? why 15 according to splaTAM
    float D = 0.0f;  // Average Depth
    float weight = 0.0f; // Weight used for integral (i.e., alpha * T)

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
            collected_depth[block.thread_rank()] = depth[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

            // Accumulate weights
            weight += alpha * T;

            // Mean depth:
            D += collected_depth[j] * alpha * T;

            // find median depth
            //if (T > 0.5f && test_T < 0.5)
			//{
			//    float dep = collected_depth[j];
			//	D = dep;
			//}

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
        out_depth[pix_id] = D;
        out_alpha[pix_id] = weight;
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
    const float* depth,
    float* out_depth,
    float* out_alpha)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
        depth,
        out_depth,
        out_alpha);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* intrinsic,
	const glm::vec3* cam_pos,
	const float* shift_factors,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* means2Dx,
	float* means2Dy,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		intrinsic,
        shift_factors,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		means2Dx,
		means2Dy,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}
