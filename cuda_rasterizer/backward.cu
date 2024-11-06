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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>

namespace cg = cooperative_groups;

template <typename T>
__device__ void checkType(T var) {
    printf("Unknown type\n");
}

template <>
__device__ void checkType<int>(int var) {
    printf("Integer type\n");
}

template <>
__device__ void checkType<float>(float var) {
    printf("Float type\n");
}

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs, glm::vec3* dL_dcampos, glm::vec3* dL_dshift)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));
    //if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //    printf("*********************************\n");
    //    printf("derivative of rgb\n");
    //    printf("%d\n", deg);
    //    printf("%f\n", dL_dRGB.x);
    //    printf("%f\n", dRGBdx.x);
    //    printf("*********************************\n");
    //}

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
    if (idx == 898411) {
        printf("++++++++++++\n");
        printf("computeColorFromSH\n");
        printf("%f\n", dL_dmean.x);
        printf("%f\n", dL_dmean.y);
        printf("%f\n", dL_dmean.z);
        printf("++++++++++++\n");
    }
    
    // The derivative of campos is accidentally the reverse direction of the 3d mean derivative.
    // Moving gaussian left == moving cam right
    //dL_dcampos[idx].x += 1;
	dL_dcampos[idx] += glm::vec3(-dL_dmean.x, -dL_dmean.y, -dL_dmean.z);

}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* proj_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov,
	float* dL_dviewmatrix,
	glm::vec3* dL_dshift,
    float* dL_ddepths,
	const float* intrinsic)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z; 
    // return t.x if -limx < txtz < limx
    // return -limx if txtz < -limx
    // return limx if txtz > limx
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(
        h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);
	//glm::mat3 J = glm::mat3(
    //    h_x / t.z, 0.0f, 0,
	//	0.0f, h_y / t.z, 0,
	//	-(h_x * t.x) / (t.z * t.z), -(h_y * t.y) / (t.z * t.z), 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);
	//glm::mat3 W = glm::mat3(
	//	view_matrix[0], view_matrix[1], view_matrix[2],
	//	view_matrix[4], view_matrix[5], view_matrix[6],
	//	view_matrix[8], view_matrix[9], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;
	//glm::mat3 T = J * W;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;
	//glm::mat3 cov2D = T * Vrk * glm::transpose(T);

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;
    float dL_dT20 = 0.;
    float dL_dT21 = 0.;
    float dL_dT22 = 0.;

	// Gradients of loss w.r.t. displacement_xyz
    // only has z'
    //dL_ddisplacement_p_w2c[4 * idx + 2] += dL_dT00 * intrinsic[0] * view_matrix[0] * (-1.) * displacement_p_w2c[4 * idx + 2]; 
    //dL_ddisplacement_p_w2c[4 * idx + 2] += dL_dT01 * intrinsic[0] * view_matrix[1] * (-1.) * displacement_p_w2c[4 * idx + 2]; 
    //dL_ddisplacement_p_w2c[4 * idx + 2] += dL_dT02 * intrinsic[0] * view_matrix[2] * (-1.) * displacement_p_w2c[4 * idx + 2]; 
    //dL_ddisplacement_p_w2c[4 * idx + 2] += dL_dT10 * intrinsic[5] * view_matrix[4] * (-1.) * displacement_p_w2c[4 * idx + 2]; 
    //dL_ddisplacement_p_w2c[4 * idx + 2] += dL_dT11 * intrinsic[5] * view_matrix[5] * (-1.) * displacement_p_w2c[4 * idx + 2]; 
    //dL_ddisplacement_p_w2c[4 * idx + 2] += dL_dT12 * intrinsic[5] * view_matrix[6] * (-1.) * displacement_p_w2c[4 * idx + 2]; 


	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

    // For entrance pupil shift gradient
	float4 p_hom = transformPoint4x4(mean, proj_matrix);
    float2 ab = {p_hom.x / p_hom.w, p_hom.y / p_hom.w};
    float theta = atan(sqrt(ab.x * ab.x + ab.y * ab.y));
    float theta2 = theta * theta;
    float theta3 = theta2 * theta;
    float theta5 = theta2 * theta3;
    float theta7 = theta2 * theta5;

	dL_dshift[idx] += glm::vec3(dL_dtz * theta3, dL_dtz * theta5, dL_dtz * theta7);
	

    // Gradients of loss w.r.t. viewmatrix
    // T = W * J
    //float dL_dv3 = 0.;
    //float dL_dv7 = 0.;
    //float dL_dv8 = 0.;
    //float dL_dv9 = 0.;
    //float dL_dv11 = 0.;
    //float dL_dv12 = 0.;
    //float dL_dv13 = 0.;
    //float dL_dv15 = 0.;
    dL_dviewmatrix[0] = h_x * tz * dL_dT00;
    dL_dviewmatrix[1] = h_x * tz * dL_dT01;
    dL_dviewmatrix[4] = h_y * tz * dL_dT10;
    dL_dviewmatrix[5] = h_y * tz * dL_dT11;
    dL_dviewmatrix[2] = -(view_matrix[0] * dL_dT00 + view_matrix[1] * dL_dT01) * mean.x * h_x * tz2 - (view_matrix[4] * dL_dT10 + view_matrix[5] * dL_dT11 + view_matrix[6] * dL_dT12) * mean.x * h_y * tz2 + (tz - view_matrix[2] * mean.x) * h_x * tz2 * dL_dT02;
    dL_dviewmatrix[6] = -(view_matrix[0] * dL_dT00 + view_matrix[1] * dL_dT01 + view_matrix[2] * dL_dT02) * mean.y * h_x * tz2 - (view_matrix[4] * dL_dT10 + view_matrix[5] * dL_dT11) * mean.y * h_y * tz2 + (tz - view_matrix[6] * mean.y) * h_y * tz2 * dL_dT12;
    dL_dviewmatrix[10] = -((view_matrix[0] * dL_dT00 + view_matrix[1] * dL_dT01 + view_matrix[2] * dL_dT02) * h_x + (view_matrix[4] * dL_dT10 + view_matrix[5] * dL_dT11 + view_matrix[6] * dL_dT12) * h_y) * mean.z * tz2;
    dL_dviewmatrix[14] = -((view_matrix[0] * dL_dT00 + view_matrix[1] * dL_dT01 + view_matrix[2] * dL_dT02) * h_x + (view_matrix[4] * dL_dT10 + view_matrix[5] * dL_dT11 + view_matrix[6] * dL_dT12) * h_y) * tz2;

    dL_dviewmatrix[2] += mean.x * dL_ddepths[idx];
    dL_dviewmatrix[6] += mean.y * dL_ddepths[idx];
    dL_dviewmatrix[10] += mean.z * dL_ddepths[idx];
    dL_dviewmatrix[14] += dL_ddepths[idx];
    
	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
    if (idx == 898411) {
        printf("++++++++++++\n");
        printf("computeCov2DCUDA\n");
        printf("%f\n", dL_dmean.x);
        printf("%f\n", dL_dmean.y);
        printf("%f\n", dL_dmean.z);
        printf("++++++++++++\n");
    }
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}


// Weights of four corners used in Bilinear Interpolation
__device__ float4 bilinearInterpolateWeights(int x, int y, float xp, float yp)
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

	return {A, B, C, D};
}


// Omnidirectional camera distortion
__device__ float3 omnidirectionalDistortion_back(float2 ab, float z, const float* affine_coeff, const float* poly_coeff) {
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
    //return {ab.x * z, ab.y * z, z};
}


// Bilinear Interpolation for displacement
__device__ float2 bilinearInterpolateKernel_back(int x, int y, const int res_u, const float* u_distortion, const float* v_distortion, float xp, float yp)
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

// Apply neuralens
__device__ float3 applyNeuralens_back(float2 ab, float z, int res_u, int res_v, const float* u_distortion, const float* v_distortion) {
    int u_idx = int((ab.x + 1) * (res_u / 2));
    int v_idx = int((ab.y + 1) * (res_v / 2));
    float2 uv_displacement;

    if (u_idx > 0 && u_idx < (res_u - 1) && v_idx > 0 && v_idx < (res_v - 1)) {
        uv_displacement = bilinearInterpolateKernel_back(u_idx, v_idx, res_u, u_distortion, v_distortion, (ab.x + 1) * (res_u / 2), (ab.y + 1) * (res_v / 2));
    }

    ab.x = ab.x + uv_displacement.x;
    ab.y = ab.y + uv_displacement.y;

    return {ab.x * z, ab.y * z, z};
}


// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* view,
	const float* proj,
	const float* intrinsic,
    const int image_height, int image_width,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_ddepths,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
    float* dL_dprojmatrix,
    glm::vec3* dL_dcampos,
    glm::vec3* dL_dshift)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

    //if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //    printf("*********************************\n");
    //    printf("number of points:");
    //    printf("%d\n", image_width);
    //    printf("%d\n", image_height);
    //    printf("*********************************\n");
    //}


    // one thread doesn't work.... shit...
    // count the number of P
    //dL_dcampos[idx].x += 1;
    //if (idx >= 4484 && idx <= 4487) {
    //    printf("*********************************\n");
    //    printf("into %d\n", idx);
    //    printf("%d\n", threadIdx.x);
    //    printf("%d\n", threadIdx.y);
    //    printf("%d\n", threadIdx.z);
    //    printf("%d\n", blockIdx.x);
    //    printf("%d\n", blockIdx.y);
    //    printf("%d\n", blockIdx.z);
    //    printf("*********************************\n");
    //}

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);
	float3 p_proj = { m_hom.x * m_w, m_hom.y * m_w, m_hom.z * m_w };

    float2 ab = {m_hom.x / m_hom.w, m_hom.y / m_hom.w};
    float theta = atan(sqrt(ab.x * ab.x + ab.y * ab.y));
    float theta2 = theta * theta;
    float theta3 = theta2 * theta;
    float theta5 = theta2 * theta3;
    float theta7 = theta2 * theta5;

    // Backward of control points
    //---------------------------------------------------------------//
    //float2 ab = {m_w2c.x / m_w2c.z, m_w2c.y / m_w2c.z};
    //float ab_u = ((ab.x + boundary_original_points[0]) / (2. * boundary_original_points[0])) * (res_control_points_u - 1);
    //float ab_v = ((ab.y + boundary_original_points[1]) / (2. * boundary_original_points[1])) * (res_control_points_v - 1);
    //int u_idx = int(ab_u);
    //int v_idx = int(ab_v);
    //float4 ABCD;

    //if (ab.x > -boundary_original_points[0] && ab.x < boundary_original_points[0] && ab.y > -boundary_original_points[1] && ab.y < boundary_original_points[1]) {
    //    ABCD = bilinearInterpolateWeights(u_idx, v_idx, ab_u, ab_v);

    //    //float magnitute_x = dL_dmean2D[idx].x * (image_width / 2) * intrinsic[0];
    //    //float magnitute_y = dL_dmean2D[idx].y * (image_height / 2) * intrinsic[5];
    //    //float magnitute   = sqrt(magnitute_x * magnitute_x + magnitute_y * magnitute_y);

    //    //float sign = (ab.x * dL_dmean2D[idx].x + ab.y * dL_dmean2D[idx].y) > 0 ? 1: -1;

    //    //float cos_theta = ab.x / sqrt(ab.x * ab.x + ab.y * ab.y);
    //    //float sin_theta = ab.y / sqrt(ab.x * ab.x + ab.y * ab.y);

    //    //dL_dcontrol_points[2 * (u_idx + v_idx * res_control_points_u) + 0]           += magnitute * sign * cos_theta * ABCD.x;
    //    //dL_dcontrol_points[2 * (u_idx + 1 + v_idx * res_control_points_u) + 0]       += magnitute * sign * cos_theta * ABCD.y;
    //    //dL_dcontrol_points[2 * (u_idx + (v_idx + 1) * res_control_points_u) + 0]     += magnitute * sign * cos_theta * ABCD.z;
    //    //dL_dcontrol_points[2 * (u_idx + 1 + (v_idx + 1) * res_control_points_u) + 0] += magnitute * sign * cos_theta * ABCD.w;
    //    //dL_dcontrol_points[2 * (u_idx + v_idx * res_control_points_u) + 1]           += magnitute * sign * sin_theta * ABCD.x;
    //    //dL_dcontrol_points[2 * (u_idx + 1 + v_idx * res_control_points_u) + 1]       += magnitute * sign * sin_theta * ABCD.y;
    //    //dL_dcontrol_points[2 * (u_idx + (v_idx + 1) * res_control_points_u) + 1]     += magnitute * sign * sin_theta * ABCD.z;
    //    //dL_dcontrol_points[2 * (u_idx + 1 + (v_idx + 1) * res_control_points_u) + 1] += magnitute * sign * sin_theta * ABCD.w;

    //    dL_dcontrol_points[2 * (u_idx + v_idx * res_control_points_u) + 0]           += dL_dmean2D[idx].x * (image_width / 2) * intrinsic[0] * ABCD.x;
    //    dL_dcontrol_points[2 * (u_idx + 1 + v_idx * res_control_points_u) + 0]       += dL_dmean2D[idx].x * (image_width / 2) * intrinsic[0] * ABCD.y;
    //    dL_dcontrol_points[2 * (u_idx + (v_idx + 1) * res_control_points_u) + 0]     += dL_dmean2D[idx].x * (image_width / 2) * intrinsic[0] * ABCD.z;
    //    dL_dcontrol_points[2 * (u_idx + 1 + (v_idx + 1) * res_control_points_u) + 0] += dL_dmean2D[idx].x * (image_width / 2) * intrinsic[0] * ABCD.w;
    //    dL_dcontrol_points[2 * (u_idx + v_idx * res_control_points_u) + 1]           += dL_dmean2D[idx].y * (image_height / 2) * intrinsic[5] * ABCD.x;
    //    dL_dcontrol_points[2 * (u_idx + 1 + v_idx * res_control_points_u) + 1]       += dL_dmean2D[idx].y * (image_height / 2) * intrinsic[5] * ABCD.y;
    //    dL_dcontrol_points[2 * (u_idx + (v_idx + 1) * res_control_points_u) + 1]     += dL_dmean2D[idx].y * (image_height / 2) * intrinsic[5] * ABCD.z;
    //    dL_dcontrol_points[2 * (u_idx + 1 + (v_idx + 1) * res_control_points_u) + 1] += dL_dmean2D[idx].y * (image_height / 2) * intrinsic[5] * ABCD.w;
    //}
    //---------------------------------------------------------------//


    // backward of neuralens
    //---------------------------------------------------------------//
    //int u_idx = int((ab.x + 1) * (res_u / 2));
    //int v_idx = int((ab.y + 1) * (res_v / 2));
    //float4 ABCD;

    //if (u_idx > 0 && u_idx < (res_u - 1) && v_idx > 0 && v_idx < (res_v - 1)) {
    //    ABCD = bilinearInterpolateWeights(u_idx, v_idx, (ab.x + 1) * (res_u / 2), (ab.y + 1) * (res_v / 2));
    //    // Partial derivative of loss w.r.t. u_distortion and v_distortion
    //    // Gradient in u direction
    //    dL_du_distortion[u_idx + v_idx * res_u]           += dL_dmean2D[idx].x * (image_width / 2) * intrinsic[0] * ABCD.x;
    //    dL_du_distortion[u_idx + 1 + v_idx * res_u]       += dL_dmean2D[idx].x * (image_width / 2) * intrinsic[0] * ABCD.y;
    //    dL_du_distortion[u_idx + (v_idx + 1) * res_u]     += dL_dmean2D[idx].x * (image_width / 2) * intrinsic[0] * ABCD.z;
    //    dL_du_distortion[u_idx + 1 + (v_idx + 1) * res_u] += dL_dmean2D[idx].x * (image_width / 2) * intrinsic[0] * ABCD.w;

    //    // Gradient in v direction
    //    dL_dv_distortion[u_idx + v_idx * res_u]           += dL_dmean2D[idx].y * (image_height / 2) * intrinsic[5] *  ABCD.x;
    //    dL_dv_distortion[u_idx + 1 + v_idx * res_u]       += dL_dmean2D[idx].y * (image_height / 2) * intrinsic[5] *  ABCD.y;
    //    dL_dv_distortion[u_idx + (v_idx + 1) * res_u]     += dL_dmean2D[idx].y * (image_height / 2) * intrinsic[5] *  ABCD.z;
    //    dL_dv_distortion[u_idx + 1 + (v_idx + 1) * res_u] += dL_dmean2D[idx].y * (image_height / 2) * intrinsic[5] *  ABCD.w;
    //}
    //---------------------------------------------------------------//

    // backward of omnidirectional camera model
    //---------------------------------------------------------------//
    //float inv_r  = 1 / sqrt(p_proj.x * p_proj.x + p_proj.y * p_proj.y);
    //float theta  = atan(sqrt(p_proj.x * p_proj.x + p_proj.y * p_proj.y));
    //float theta2 = theta * theta;
    //float theta3 = theta * theta2;
    //float theta5 = theta3 * theta2;
    //float theta7 = theta5 * theta2;
    //float theta9 = theta7 * theta2;

    //if ((p_proj.x * p_proj.x + p_proj.y * p_proj.y) < 4){
    //    dL_dpoly[4 * idx + 0] = (dL_dmean2D[idx].x * (image_width / 2) * m_w * p_proj.x * inv_r * theta3 + dL_dmean2D[idx].y * (image_height / 2) * m_w * p_proj.y * inv_r * theta3);
    //    dL_dpoly[4 * idx + 1] = (dL_dmean2D[idx].x * (image_width / 2) * m_w * p_proj.x * inv_r * theta5 + dL_dmean2D[idx].y * (image_height / 2) * m_w * p_proj.y * inv_r * theta5);
    //    dL_dpoly[4 * idx + 2] = (dL_dmean2D[idx].x * (image_width / 2) * m_w * p_proj.x * inv_r * theta7 + dL_dmean2D[idx].y * (image_height / 2) * m_w * p_proj.y * inv_r * theta7);
    //    dL_dpoly[4 * idx + 3] = (dL_dmean2D[idx].x * (image_width / 2) * m_w * p_proj.x * inv_r * theta9 + dL_dmean2D[idx].y * (image_height / 2) * m_w * p_proj.y * inv_r * theta9);
    //}
    //---------------------------------------------------------------//


    // backward of distortion table
    //---------------------------------------------------------------//
    //int left_x  = int((theta / 1.57079632679) * 1000);
    //int right_x = int((theta / 1.57079632679) * 1000) + 1;
    //float middle_x = (theta / 1.57079632679) * 1000;

    //if (right_x < 1000 && right_x >= 0 && left_x < 1000 && left_x >= 0) {
    //    dL_dradial[left_x] += dL_dmean2D[idx].x * (image_width / 2) * m_w * m_w2c.x * (right_x - middle_x) / (right_x - left_x);
    //    dL_dradial[left_x] += dL_dmean2D[idx].y * (image_height / 2) * m_w * m_w2c.y * (right_x - middle_x) / (right_x - left_x);


    //    dL_dradial[right_x] += dL_dmean2D[idx].x * (image_width / 2) * m_w * m_w2c.x * (middle_x - left_x) / (right_x - left_x);
    //    dL_dradial[right_x] += dL_dmean2D[idx].y * (image_height / 2) * m_w * m_w2c.y * (middle_x - left_x) / (right_x - left_x);
    //}
    //---------------------------------------------------------------//


    // backward of grid
    //---------------------------------------------------------------//
    //int u_idx = int((p_proj.x + 1) * (res_u / 2));
    //int v_idx = int((p_proj.y + 1) * (res_v / 2));
    //float4 ABCD;

    //if (u_idx > 0 && u_idx < (res_u - 1) && v_idx > 0 && v_idx < (res_v - 1)) {
    //    ABCD = bilinearInterpolateWeights(u_idx, v_idx, (p_proj.x + 1) * (res_u / 2), (p_proj.y + 1) * (res_v / 2));

    //    //// check ABCD
    //    //if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //    //    printf("*********************************\n");
    //    //    printf("%f\n", ABCD.x);
    //    //    printf("%f\n", ABCD.y);
    //    //    printf("%f\n", ABCD.z);
    //    //    printf("%f\n", ABCD.w);
    //    //    printf("*********************************\n");
    //    //}

    //    // Partial derivative of loss w.r.t. u_distortion and v_distortion
    //    // Gradient in u direction
    //    dL_du_distortion[u_idx + v_idx * res_u]           += dL_dmean2D[idx].x * (image_width / 2) * ABCD.x;
    //    dL_du_distortion[u_idx + 1 + v_idx * res_u]       += dL_dmean2D[idx].x * (image_width / 2) * ABCD.y;
    //    dL_du_distortion[u_idx + (v_idx + 1) * res_u]     += dL_dmean2D[idx].x * (image_width / 2) * ABCD.z;
    //    dL_du_distortion[u_idx + 1 + (v_idx + 1) * res_u] += dL_dmean2D[idx].x * (image_width / 2) * ABCD.w;

    //    // Gradient in v direction
    //    dL_dv_distortion[u_idx + v_idx * res_u]           += dL_dmean2D[idx].y * (image_height / 2) * ABCD.x;
    //    dL_dv_distortion[u_idx + 1 + v_idx * res_u]       += dL_dmean2D[idx].y * (image_height / 2) * ABCD.y;
    //    dL_dv_distortion[u_idx + (v_idx + 1) * res_u]     += dL_dmean2D[idx].y * (image_height / 2) * ABCD.z;
    //    dL_dv_distortion[u_idx + 1 + (v_idx + 1) * res_u] += dL_dmean2D[idx].y * (image_height / 2) * ABCD.w;

    //    // Partial derivative of loss w.r.t. u_radial and v_radial
    //    // Gradient in u direction
    //    dL_du_radial[u_idx + v_idx * res_u]           += dL_dmean2D[idx].x * (image_width / 2) * ABCD.x * p_proj.x;
    //    dL_du_radial[u_idx + 1 + v_idx * res_u]       += dL_dmean2D[idx].x * (image_width / 2) * ABCD.y * p_proj.x;
    //    dL_du_radial[u_idx + (v_idx + 1) * res_u]     += dL_dmean2D[idx].x * (image_width / 2) * ABCD.z * p_proj.x;
    //    dL_du_radial[u_idx + 1 + (v_idx + 1) * res_u] += dL_dmean2D[idx].x * (image_width / 2) * ABCD.w * p_proj.x;

    //    // Gradient in v direction
    //    dL_dv_radial[u_idx + v_idx * res_u]           += dL_dmean2D[idx].y * (image_height / 2) * ABCD.x * p_proj.y;
    //    dL_dv_radial[u_idx + 1 + v_idx * res_u]       += dL_dmean2D[idx].y * (image_height / 2) * ABCD.y * p_proj.y;
    //    dL_dv_radial[u_idx + (v_idx + 1) * res_u]     += dL_dmean2D[idx].y * (image_height / 2) * ABCD.z * p_proj.y;
    //    dL_dv_radial[u_idx + 1 + (v_idx + 1) * res_u] += dL_dmean2D[idx].y * (image_height / 2) * ABCD.w * p_proj.y;
    //}
    //---------------------------------------------------------------//

	// Compute loss gradient w.r.t. 8 distortion parameters using dL_dmean2D
    //---------------------------------------------------------------//
    //float k1 = distortion_params[0];
    //float k2 = distortion_params[1];
    //float k3 = distortion_params[2];
    //float k4 = distortion_params[3];
    //float k5 = distortion_params[4];
    //float k6 = distortion_params[5];
    //float p1 = distortion_params[6];
    //float p2 = distortion_params[7];
    //
    //float x2       = p_proj.x * p_proj.x;
    //float y2       = p_proj.y * p_proj.y;
    //float r2       = x2 + y2;
    //float _2xy     = float(2) * p_proj.x * p_proj.y;
    //float radial_u = float(1) + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    //float radial_v = float(1) + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2;
    //// The forward distortion fails if the points are too far away on the image plain
    //if (r2 < 2){
    //    // gradient from p_proj.x
    //    dL_ddistortion_params[8 * idx]     += dL_dmean2D[idx].x * (image_width / 2) * (p_proj.x * r2 ) / radial_v;
    //    dL_ddistortion_params[8 * idx + 1] += dL_dmean2D[idx].x * (image_width / 2) * (p_proj.x * r2 * r2) / radial_v;
    //    dL_ddistortion_params[8 * idx + 2] += dL_dmean2D[idx].x * (image_width / 2) * (p_proj.x * r2 * r2 * r2) / radial_v;
    //    dL_ddistortion_params[8 * idx + 3] += dL_dmean2D[idx].x * (image_width / 2) * (p_proj.x * radial_u * (-1.) * r2) / (radial_v * radial_v);
    //    dL_ddistortion_params[8 * idx + 4] += dL_dmean2D[idx].x * (image_width / 2) * (p_proj.x * radial_u * (-1.) * r2 * r2) / (radial_v * radial_v);
    //    dL_ddistortion_params[8 * idx + 5] += dL_dmean2D[idx].x * (image_width / 2) * (p_proj.x * radial_u * (-1.) * r2 * r2 * r2) / (radial_v * radial_v);
    //    dL_ddistortion_params[8 * idx + 6] += dL_dmean2D[idx].x * (image_width / 2) * _2xy;
    //    dL_ddistortion_params[8 * idx + 7] += dL_dmean2D[idx].x * (image_width / 2) * (float(2) * x2 + r2);
    //    // gradient from p_proj.y
    //    dL_ddistortion_params[8 * idx]     += dL_dmean2D[idx].y * (image_height /2) * (p_proj.y * r2 ) / radial_v;
    //    dL_ddistortion_params[8 * idx + 1] += dL_dmean2D[idx].y * (image_height /2) * (p_proj.y * r2 * r2) / radial_v;
    //    dL_ddistortion_params[8 * idx + 2] += dL_dmean2D[idx].y * (image_height /2) * (p_proj.y * r2 * r2 * r2) / radial_v;
    //    dL_ddistortion_params[8 * idx + 3] += dL_dmean2D[idx].y * (image_height /2) * (p_proj.y * radial_u * (-1.) * r2) / (radial_v * radial_v);
    //    dL_ddistortion_params[8 * idx + 4] += dL_dmean2D[idx].y * (image_height /2) * (p_proj.y * radial_u * (-1.) * r2 * r2) / (radial_v * radial_v);
    //    dL_ddistortion_params[8 * idx + 5] += dL_dmean2D[idx].y * (image_height /2) * (p_proj.y * radial_u * (-1.) * r2 * r2 * r2) / (radial_v * radial_v);
    //    dL_ddistortion_params[8 * idx + 6] += dL_dmean2D[idx].y * (image_height /2) * (float(2) * y2 + r2);
    //    dL_ddistortion_params[8 * idx + 7] += dL_dmean2D[idx].y * (image_height /2) * _2xy;
    //}
    //---------------------------------------------------------------//

	// Compute loss gradient w.r.t. 3D means under camera coordinate with displacement due to gradients of 2D means
    //dL_ddisplacement_p_w2c[4 * idx] += dL_dmean2D[idx].x * (image_width / 2) * m_w * intrinsic[0]; 
    //dL_ddisplacement_p_w2c[4 * idx + 1] += dL_dmean2D[idx].y * (image_height / 2) * m_w * intrinsic[5]; 
    //dL_ddisplacement_p_w2c[4 * idx + 2] += dL_dmean2D[idx].x * (image_width / 2) * (-1.) * m_w * m_w * m_hom.x + dL_dmean2D[idx].y * (image_height / 2) * (-1.) * m_w * m_w * m_hom.y; 

    // check intrinsic order
    //if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //    printf("*********************************\n");
    //    for (int i = 0; i < 16; i++) {
    //        printf("%f\n", intrinsic[i]);
    //    }
    //    printf("*********************************\n");
    //}



    // backward of projection matrix
    //---------------------------------------------------------------//
    float scaling_radial = 1.;
	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = scaling_radial * (image_width / 2) * (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + scaling_radial * (image_height / 2) * (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = scaling_radial * (image_width / 2) * (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + scaling_radial * (image_height / 2) * (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = scaling_radial * (image_width / 2) * (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + scaling_radial * (image_height / 2) * (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;
    // Original 3dgs implementation without height and width as the constants
	//dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	//dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	//dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

    //Compute the loss gradient w.r.t entrance shift 
    dL_dshift[idx] += glm::vec3(
            dL_dmean2D[idx].x * (image_width / 2) * m_hom.x * (-1. / (m_w * m_w)) * theta3 + dL_dmean2D[idx].y * (image_height / 2) * m_hom.y * (-1. / (m_w * m_w)) * theta3,
            dL_dmean2D[idx].x * (image_width / 2) * m_hom.x * (-1. / (m_w * m_w)) * theta3 + dL_dmean2D[idx].y * (image_height / 2) * m_hom.y * (-1. / (m_w * m_w)) * theta5,
            dL_dmean2D[idx].x * (image_width / 2) * m_hom.x * (-1. / (m_w * m_w)) * theta3 + dL_dmean2D[idx].y * (image_height / 2) * m_hom.y * (-1. / (m_w * m_w)) * theta7
            );

    //Compute the loss gradient w.r.t projection matrix 
    // the graident flow back from ux (screen), p0, p4, p8, p12
    //dL_dprojmatrix[16 * idx + 0] = m.x * dL_dmean2D[idx].x;
    //dL_dprojmatrix[16 * idx + 4] = m.y * dL_dmean2D[idx].x;
    //dL_dprojmatrix[16 * idx + 8] = m.z * dL_dmean2D[idx].x;
    //dL_dprojmatrix[16 * idx + 12] = dL_dmean2D[idx].x;
    dL_dprojmatrix[16 * idx + 0]  = scaling_radial * m_w * m.x * dL_dmean2D[idx].x;
    dL_dprojmatrix[16 * idx + 4]  = scaling_radial * m_w * m.y * dL_dmean2D[idx].x;
    dL_dprojmatrix[16 * idx + 8]  = scaling_radial * m_w * m.z * dL_dmean2D[idx].x;
    dL_dprojmatrix[16 * idx + 12] = scaling_radial * m_w * dL_dmean2D[idx].x;
    // the graident flow back from uy (screen), p1, p5, p9, p13
    //dL_dprojmatrix[16 * idx + 1] = m.x * dL_dmean2D[idx].y;
    //dL_dprojmatrix[16 * idx + 5] = m.y * dL_dmean2D[idx].y;
    //dL_dprojmatrix[16 * idx + 9] = m.z * dL_dmean2D[idx].y;
    //dL_dprojmatrix[16 * idx + 13] = dL_dmean2D[idx].y;
    dL_dprojmatrix[16 * idx + 1]  = scaling_radial * m_w * m.x * dL_dmean2D[idx].y;
    dL_dprojmatrix[16 * idx + 5]  = scaling_radial * m_w * m.y * dL_dmean2D[idx].y;
    dL_dprojmatrix[16 * idx + 9]  = scaling_radial * m_w * m.z * dL_dmean2D[idx].y;
    dL_dprojmatrix[16 * idx + 13] = scaling_radial * m_w * dL_dmean2D[idx].y;
    // the graident flow back from both ux and uy (screen), p3, p7, p11, p15
    //dL_dprojmatrix[16 * idx + 3] = (m_w * (-1.) * m_hom.x * m.x * dL_dmean2D[idx].x + m_w * (-1.) * m_hom.y * m.x * dL_dmean2D[idx].y);
    //dL_dprojmatrix[16 * idx + 7] = (m_w * (-1.) * m_hom.x * m.y * dL_dmean2D[idx].x + m_w * (-1.) * m_hom.y * m.y * dL_dmean2D[idx].y);
    //dL_dprojmatrix[16 * idx + 11] = (m_w * (-1.) * m_hom.x * m.z * dL_dmean2D[idx].x + m_w * (-1.) * m_hom.y * m.z * dL_dmean2D[idx].y);
    //dL_dprojmatrix[16 * idx + 15] = (m_w * (-1.) * m_hom.x * dL_dmean2D[idx].x + m_w * (-1.) * m_hom.y * dL_dmean2D[idx].y);
    dL_dprojmatrix[16 * idx + 3]  = scaling_radial * (m_w * m_w * (-1.) * m_hom.x * m.x * dL_dmean2D[idx].x + m_w * m_w * (-1.) * m_hom.y * m.x * dL_dmean2D[idx].y);
    dL_dprojmatrix[16 * idx + 7]  = scaling_radial * (m_w * m_w * (-1.) * m_hom.x * m.y * dL_dmean2D[idx].x + m_w * m_w * (-1.) * m_hom.y * m.y * dL_dmean2D[idx].y);
    dL_dprojmatrix[16 * idx + 11] = scaling_radial * (m_w * m_w * (-1.) * m_hom.x * m.z * dL_dmean2D[idx].x + m_w * m_w * (-1.) * m_hom.y * m.z * dL_dmean2D[idx].y);
    dL_dprojmatrix[16 * idx + 15] = scaling_radial * (m_w * m_w * (-1.) * m_hom.x * dL_dmean2D[idx].x + m_w * m_w * (-1.) * m_hom.y * dL_dmean2D[idx].y);
    // p2, p6, p10, p14 have the identical gradient to p3 7 11 15
    //dL_dprojmatrix[16 * idx + 2] = (m_w * m_w * (-1.) * m_hom.x * m.x * dL_dmean2D[idx].x + m_w * m_w * (-1.) * m_hom.y * m.x * dL_dmean2D[idx].y);
    //dL_dprojmatrix[16 * idx + 6] = (m_w * m_w * (-1.) * m_hom.x * m.y * dL_dmean2D[idx].x + m_w * m_w * (-1.) * m_hom.y * m.y * dL_dmean2D[idx].y);
    //dL_dprojmatrix[16 * idx + 10] = (m_w * m_w * (-1.) * m_hom.x * m.z * dL_dmean2D[idx].x + m_w * m_w * (-1.) * m_hom.y * m.z * dL_dmean2D[idx].y);
    //dL_dprojmatrix[16 * idx + 14] = (m_w * m_w * (-1.) * m_hom.x * dL_dmean2D[idx].x + m_w * m_w * (-1.) * m_hom.y * dL_dmean2D[idx].y);
    // check the gradient
    // also an example to printf value
    //const glm::mat4* test = (glm::mat4*)dL_dprojmatrix;
    //if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //    printf("*********************************\n");
    //    printf("index is:");
    //    printf("%d\n", idx);
    //    for (int i = 0; i < 4; i++) {
    //        for (int j = 0; j < 4; j++) {
    //            printf("%f\n", test[idx][i][j]);
    //        }
    //    }
    //    printf("*********************************\n");
    //}
    //---------------------------------------------------------------//


	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;
    if (idx == 898411) {
        printf("++++++++++++\n");
        printf("preprocessCUDA1\n");
        printf("%f\n", dL_dmean.x);
        printf("%f\n", dL_dmean.y);
        printf("%f\n", dL_dmean.z);
        printf("++++++++++++\n");
    }

    // the w must be equal to 1 for view^T * [x,y,z,1]
	float3 m_view = transformPoint4x3(m, view);

	// Compute loss gradient w.r.t. 3D means due to gradients of depth
	// from rendering procedure
	//glm::vec3 dL_dmean2;
	//float mul3 = view[2] * m.x + view[6] * m.y + view[10] * m.z + view[14];
	//dL_dmean2.x = (view[2] - view[3] * mul3) * dL_ddepths[idx];
	//dL_dmean2.y = (view[6] - view[7] * mul3) * dL_ddepths[idx];
	//dL_dmean2.z = (view[10] - view[11] * mul3) * dL_ddepths[idx];
	//// That's the third part of the mean gradient.
	//dL_dmeans[idx] += dL_dmean2;
	dL_dmeans[idx].x += view[2] * dL_ddepths[idx];
	dL_dmeans[idx].y += view[6] * dL_ddepths[idx];
	dL_dmeans[idx].z += view[10] * dL_ddepths[idx];
    if (idx == 898411) {
        printf("++++++++++++\n");
        printf("preprocessCUDA2\n");
        printf("%f\n", view[2] * dL_ddepths[idx]);
        printf("%f\n", view[6] * dL_ddepths[idx]);
        printf("%f\n", view[10] * dL_ddepths[idx]);
        printf("++++++++++++\n");
    }


	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh, dL_dcampos, dL_dshift);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ alphas,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dpixel_depths,
	const float* __restrict__ dL_dalphas,
	float3* __restrict__ dL_dmean2D,
	float3* __restrict__ dL_dmean2D_densify,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ covariance,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_ddepths)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	float accum_depth_rec = 0;
	float dL_dpixel_depth;
    float accum_alpha_rec = 0;
	float dL_dweight_alpha;
    if (inside) {
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		dL_dpixel_depth = dL_dpixel_depths[pix_id];
		dL_dweight_alpha = dL_dalphas[pix_id];
    }

	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_depth = 0;

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
            const float2 xy = collected_xy[j];
            const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
            const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;
			const float dpixel_depth_ddepth = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}

			// Propagate gradients from pixel depth to opacity
			const float c_d = collected_depths[j];
			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = c_d;
			dL_dalpha += (c_d - accum_depth_rec) * dL_dpixel_depth;
			atomicAdd(&(dL_ddepths[global_id]), dpixel_depth_ddepth * dL_dpixel_depth);

            // Propagate gradients from pixel alpha (weights_sum) to opacity
			accum_alpha_rec = last_alpha + (1.f - last_alpha) * accum_alpha_rec;
			dL_dalpha += (1 - accum_alpha_rec) * dL_dweight_alpha; //- (alpha - accum_alpha_rec) * dL_dalpha;

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// Update gradients for densification
			atomicAdd(&dL_dmean2D_densify[global_id].x, fabsf(dL_dG * dG_ddelx * ddelx_dx));
			atomicAdd(&dL_dmean2D_densify[global_id].y, fabsf(dL_dG * dG_ddely * ddely_dy));

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Store the covariance of each gaussian based on conic2D
			atomicAdd(&(covariance[global_id]), (con_o.x + con_o.z));
			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
        // check 2x2 matrix will have one 0. unsigned value
        //if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
		//	const int global_id = collected_id[0];
        //    printf("************************\n");
        //    printf("%f\n",dL_dconic2D[global_id].x);
        //    printf("%f\n",dL_dconic2D[global_id].y);
        //    printf("%f\n",dL_dconic2D[global_id].z);
        //    printf("%f\n",dL_dconic2D[global_id].w);
        //    printf("************************\n");
        //}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float* intrinsic,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
    const int image_height, int image_width,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_ddepths,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dprojmatrix,
	float* dL_dviewmatrix,
	float* dL_dcampos,
    float* dL_dshift)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		projmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D,
		dL_dviewmatrix,
        (glm::vec3*)dL_dshift,
        dL_ddepths,
        intrinsic);


	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
    //std::cout << "*****************************************************" << std::endl;
    //std::cout << "abcd" << std::endl;
    //std:: cout << campos << std::endl;
    //std::cout << "*****************************************************" << std::endl;
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
        viewmatrix,
		projmatrix,
		intrinsic,
        image_height, image_width,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_ddepths,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
        dL_drot,
		dL_dprojmatrix,
		(glm::vec3*)dL_dcampos,
		(glm::vec3*)dL_dshift);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* depths,
	const float* alphas,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_dpixel_depths,
	const float* dL_dalphas,
    float3* dL_dmean2D,
	float3* dL_dmean2D_densify,
	float4* dL_dconic2D,
	float* covariance,
	float* dL_dopacity,
	float* dL_dcolors,
    float* dL_ddepths)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		depths,
        alphas,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dpixel_depths,
		dL_dalphas,
		dL_dmean2D,
		dL_dmean2D_densify,
		dL_dconic2D,
		covariance,
		dL_dopacity,
		dL_dcolors,
		dL_ddepths
		);
}
