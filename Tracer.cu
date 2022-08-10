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

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state)
{
	while (true) {
		float rand_x = (curand_uniform(local_rand_state) * 2) - 1;
		float rand_y = (curand_uniform(local_rand_state) * 2) - 1;
		float rand_z = (curand_uniform(local_rand_state) * 2) - 1;
		vec3 p = vec3(rand_x, rand_y, rand_z);
		if (p.length_squared() >= 1.0f) {
			continue;
		}
		return p;
	}
}

__device__ vec3 random_in_hemisphere(const vec3& normal, curandState* local_rand_state) {
	vec3 in_unit_sphere = random_in_unit_sphere(local_rand_state);
	/* In the same hemisphere as the normal */
	if (dot(in_unit_sphere, normal) > 0.0f) {
		return in_unit_sphere;
	}
	else {
		return -in_unit_sphere;
	}
}

/* Based on the value of the y component in the normalized direction vector of the ray, calculate 
the final color by interpolating between white and color(0.5f, 0.7f, 1.0f). */
__device__ color ray_color(const ray& r, hittable **world, curandState *rand_state)
{
	ray curr_ray = r;
	float curr_attenuation = 1.0f;
	for (int i = 0; i < 50; i++) {
		hit_record rec;
		if ((*world)->hit(curr_ray, 0.001f, FLT_MAX, rec)) {
			vec3 target = rec.p + rec.normal + random_in_hemisphere(rec.normal, rand_state);
			curr_attenuation *= 0.5f;
			curr_ray = ray(rec.p, target - rec.p);
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

__global__ void create_world(hittable **d_list, hittable **d_world, camera **d_camera)
{
	if (threadIdx.x == 0 && threadIdx.x == 0) {
		*(d_list) = new sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f);
		*(d_list + 1) = new sphere(vec3(0.0f, -100.5f, -1.0f), 100.0f);
		*(d_world) = new hittable_list(d_list, 2);
		*(d_camera) = new camera();
	}
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
		ray r = (*cam)->get_ray(u, v);
		pixel_color += ray_color(r, world, &local_rand_state);
	}
	rand_state[pixel_index] = local_rand_state;
	fb[pixel_index] = pixel_color / float(num_samples);
}

__global__ void free_world(hittable** d_list, hittable** d_world, camera **d_camera)
{
	delete* (d_list);
	delete* (d_list + 1);
	delete* (d_world);
	delete* (d_camera);
}

int main(void)
{
	/* Image size. */
	const int nx = 1200, ny = 600;
	const int num_pixels = nx * ny;
	const int num_of_samples = 32;

	/* Thread size for dividing work on GPU. */
	int tx = 8, ty = 8;

	/* CUDA random state objects for anti-aliasing in each pixel. */
	curandState* d_rand_state;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));

	/* NOTE: The d_ prefix here is to denote device only data (GPU only data). */

	/* List of objects that are hittable in our scene. */
	hittable** d_list;
	checkCudaErrors(cudaMalloc((void**)&d_list, 2*sizeof(hittable *)));
	hittable** d_world;
	checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
	camera** d_camera;
	checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
	create_world<<<1, 1 >>> (d_list, d_world, d_camera);
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
	free_world<<<1, 1 >>>(d_list, d_world, d_camera);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(fb));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(d_camera));

	out_ppm.close();

	cudaDeviceReset();

	return 0;
}