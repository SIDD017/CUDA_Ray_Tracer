#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <iostream>
#include <fstream>

#include "essentials.h"
#include "color.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)

const char* ppm_filename = "Render.ppm";

/* Check if CUDA API call generated an error. Reset and exit if true. */
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";

		cudaDeviceReset();
		exit(99);
	}
}

/* Based on the value of the y component in the normalized direction vector of the ray, calculate 
the final color by interpolating between white and color(0.5f, 0.7f, 1.0f). */
__device__ color ray_color(const ray& r, hittable **world, curandState *rand_state)
{
	ray curr_ray = r;
	color curr_attenuation(1.0f, 1.0f, 1.0f);
	for (int i = 0; i < 50; i++) {
		hit_record rec;
		if ((*world)->hit(curr_ray, 0.001f, FLT_MAX, rec)) {
			ray scattered;
			color attenuation;
			if (rec.mtr->scatter(curr_ray, rec, attenuation, scattered, rand_state)) {
				curr_attenuation *= attenuation;
				curr_ray = scattered;
			}
			else {
				return color(0.0f, 0.0f, 0.0f);
			}
		}
		else {
			vec3 unit_direction = unit_length(curr_ray.direction());
			float t = 0.5f * (unit_direction.y() + 1.0f);
			vec3 c = (1.0f - t) * color(1.0f, 1.0f, 1.0f) + t * color(0.5f, 0.7f, 1.0f);
			return curr_attenuation * c;
		}
	}
	return vec3(0.0f, 0.0f, 0.0f);
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	/* Early return if the current thread is not mapped to any pixel in the final render. */
	if ((i >= max_x) || (j >= max_y)) {
		return;
	}

	int pixel_index = j * max_x + i;
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int num_samples, camera **cam, hittable** world, curandState *rand_state)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	/* Early return if the current thread is not mapped to any pixel in the final render. */
	if ((i >= max_x) || (j >= max_y)) {
		return;
	}

	int pixel_index = j * max_x + i;
	curandState local_rand_state = rand_state[pixel_index];
	color pixel_color(0.0f, 0.0f, 0.0f);
	for (int k = 0; k < num_samples; k++) {
		float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
		ray r = (*cam)->get_ray(u, v, local_rand_state);
		pixel_color += ray_color(r, world, &local_rand_state);
	}
	rand_state[pixel_index] = local_rand_state;
	fb[pixel_index] = pixel_color / float(num_samples);
}

__global__ void rand_init(curandState* rand_state)
{
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curand_init(1984, 0, 0, rand_state);
	}
}

#define RND (curand_uniform(&local_rand_state))

__device__ void random_scene(hittable **d_list, hittable **d_world, material **d_material, curandState *rand_state)
{
	curandState local_rand_state = *rand_state;
	
	/* Ground material and sphere. */
	*(d_material) = new lambertian(color(0.5f, 0.5f, 0.5f));
	*(d_list) = new sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f, *(d_material));

	/* 3 large spheres. One lambertian, one metal and one dielectric. */
	*(d_material + 1) = new lambertian(color(0.4f, 0.2f, 0.1f));
	*(d_material + 2) = new dielectric(1.5f);
	*(d_material + 3) = new metal(color(0.7f, 0.6f, 0.5f), 0.0f);
	*(d_list + 1) = new sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f, *(d_material + 1));
	*(d_list + 2) = new sphere(vec3(0.0f, 1.0f, 0.0f), 1.0f, *(d_material + 2));
	*(d_list + 3) = new sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f, *(d_material + 3));

	int i = 4;
	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			float choose_mat = RND;
			point3 center(a + RND, 0.2f, b + RND);

			if (choose_mat < 0.8f) {
				/* Diffuse. */
				color albedo(RND * RND, RND * RND, RND * RND);
				*(d_material + i) = new lambertian(albedo);
				vec3 center2 = center + vec3(0.0f, RND * 0.5f, 0.0f);
				*(d_list + i) = new moving_sphere(center, center2, 0.0f, 1.0f, 0.2f, *(d_material + i));
				i++;
			}
			else if (choose_mat < 0.95f) {
				/* Metal. */
				color albedo(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND));
				float fuzz = 0.5f * (1.0f + RND);
				*(d_material + i) = new metal(albedo, fuzz);
				*(d_list + i) = new sphere(center, 0.2f, *(d_material + i));
				i++;
			}
			else {
				/* Glass. */
				*(d_material + i) = new dielectric(1.5f);
				*(d_list + i) = new sphere(center, 0.2f, *(d_material + i));
				i++;
			}
		}
	}

}

__global__ void create_world(hittable** d_list, hittable** d_world, camera** d_camera, material** d_material, curandState *rand_state, int num_of_objs)
{
	if (threadIdx.x == 0 && threadIdx.x == 0) {

		point3 lookfrom(13.0f, 2.0f, 3.0f);
		point3 lookat(0.0f, 0.0f, 0.0f);
		vec3 vup(0.0f, 1.0f, 0.0f);
		float dist_to_focus = 10.0f;
		float aperture = 0.1f;

		random_scene(d_list, d_world, d_material, rand_state);
		*(d_world) = new hittable_list(d_list, num_of_objs);
		*(d_camera) = new camera(lookfrom, lookat, vup, 20.0f, (float)(16.0f / 9.0f), aperture, dist_to_focus, 0.0f, 1.0f);
	}
}

__global__ void free_world(hittable** d_list, hittable** d_world, camera **d_camera, material **d_material)
{
	for (int i = 0; i < (22 * 22 + 1 + 3); i++) {
		delete* (d_list + i);
		delete* (d_material + i);
	}
	
	delete* (d_world);
	delete* (d_camera);
}

int main(void)
{
	/* Image size. */
	const int nx = 400, ny = 225;
	const int num_pixels = nx * ny;
	const int num_of_samples = 32;

	/* Thread size for dividing work on GPU. */
	int tx = 8, ty = 8;

	/* CUDA random state objects for anti-aliasing in each pixel. */
	curandState* d_rand_state;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
	curandState* d_objs_rand_state;
	checkCudaErrors(cudaMalloc((void**)&d_objs_rand_state, 1 * sizeof(curandState)));

	/* Initialize d_objs_rand_state fro world creation. */
	rand_init<<<1, 1 >>>(d_objs_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	/* NOTE: The d_ prefix here is to denote device only data (GPU only data). */

	int num_of_objs = 22 * 22 + 1 + 3;
	/* List of materials inlcuded in our scene. */
	material** d_material;
	checkCudaErrors(cudaMalloc((void **)&d_material, num_of_objs * sizeof(material*)));

	/* List of objects that are hittable in our scene. */
	hittable** d_list;
	checkCudaErrors(cudaMalloc((void**)&d_list, num_of_objs * sizeof(hittable *)));
	hittable** d_world;
	checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
	camera** d_camera;
	checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
	create_world<<<1, 1 >>> (d_list, d_world, d_camera, d_material, d_objs_rand_state, num_of_objs);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	/* Size of frame buffer in Unified memory to hold final pixel values. */
	size_t fb_size = num_pixels * sizeof(vec3);

	/* Allocate Unified memory for framebuffer */
	vec3* fb;
	checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

	/* Number of required blocks. */
	dim3 blocks(nx/tx+1, ny/ty+1);
	/* Number of threads per block. */
	dim3 threads(tx, ty);

	render_init<<<blocks, threads >>>(nx, ny, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	render<<<blocks, threads>>> (fb, nx, ny, num_of_samples, d_camera, d_world, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	/* Use fstream to write final pixel RGB values to the output ppm file. */
	std::ofstream out_ppm;
	out_ppm.open(ppm_filename);

	std::cout << "Writing to output file\n";
	/* Write the final pixel values from the buffer in Unified memory to the ppm output file. */
	out_ppm << "P3\n" << nx << " " << ny << "\n255\n";
	for (int j = ny - 1; j >= 0; j--) {
		for (int i = 0; i < nx; i++) {
			size_t pixel_index = j * nx + i;
			color pixel_color(fb[pixel_index].x(), fb[pixel_index].y(), fb[pixel_index].z());
			write_color(out_ppm, pixel_color);
		}
	}

	std::cout << "Done writing to output file\n";

	/* Cleanup before terminating application. */
	free_world<<<1, 1 >>>(d_list, d_world, d_camera, d_material);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(fb));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(d_objs_rand_state));
	checkCudaErrors(cudaFree(d_camera));

	out_ppm.close();

	cudaDeviceReset();

	return 0;
}