#include "cuda_runtime.h"
#include <device_launch_parameters.h>

#include <iostream>
#include <fstream>

#include "vec3.h"
#include "color.h"
#include "ray.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)

const char* ppm_filename = "Render.ppm";

/* Check if CUDA API call generated an error. Reset and exit if true. */
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << "at" <<
			file << ":" << line << " '" << func << "' \n";

		cudaDeviceReset();
		exit(99);
	}
}

__device__ float hit_sphere(const point3& center, float radius, const ray& r)
{
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = 2.0f * dot(oc, r.direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - 4.0f * a * c;
	if (discriminant < 0.0f) {
		return -1.0f;
	}
	else {
		return (-b - sqrt(discriminant)) / (2.0f * a);
	}
}

/* Based on the value of the y component in the normalized direction vector of the ray, calculate 
the final color by interpolating between white and color(0.5f, 0.7f, 1.0f). */
__device__ color ray_color(const ray& r)
{
	float t = hit_sphere(point3(0.0f, 0.0f, -1.0f), 0.5f, r);
	if (t > 0.0f) {
		vec3 N = unit_length(r.at(t) - vec3(0.0f, 0.0f, -1.0f));
		return 0.5f * color(N.x() + 1, N.y() + 1, N.z() + 1);
	}
	const vec3 unit_direction = unit_length(r.direction());
	t = 0.5f * (unit_direction.y() + 1.0f);
	return (1.0f - t) * color(1.0f, 1.0f, 1.0f) + t * color(0.5f, 0.7f, 1.0f);
}

__global__ void render(vec3 *fb, int max_x, int max_y, vec3 lower_left_corner, vec3 vertical, vec3 horizontal, vec3 origin)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	/* Early return if the current thread is not mapped to any pixel in the final render. */
	if ((i >= max_x) || (j >= max_y)) {
		return;
	}

	int pixel_index = j * max_x + i;
	float u = float(i) / float(max_x);
	float v = float(j) / float(max_y);
	ray r(origin, lower_left_corner + u * horizontal + v * vertical);
	fb[pixel_index] = ray_color(r);
}

int main(void)
{
	/* Image size. */
	const int nx = 1200, ny = 600;
	const int num_pixels = nx * ny;

	/* Camera properties. */
	const float viewport_height = 2.0f;
	const float viewport_width = (nx / ny) * viewport_height;
	const float focal_length = 1.0f;

	/* Viewport properties. */
	const vec3 origin = point3(0.0f, 0.0f, 0.0f);
	const vec3 horizontal = vec3(viewport_width, 0.0f, 0.0f);
	const vec3 vertical = vec3(0.0f, viewport_height, 0.0f);
	const vec3 lower_left_corner = origin - (horizontal / 2.0f) - (vertical / 2.0f) - vec3(0.0f, 0.0f, focal_length);

	/* Size of frame buffer in Unified memory to hold final pixel values. */
	size_t fb_size = num_pixels * sizeof(vec3);

	/* Allocate Unified memory for framebuffer */
	vec3* fb;
	checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

	/* Thread size for dividing work on GPU. */
	int tx = 8, ty = 8;

	/* Number of required blocks. */
	dim3 blocks(nx/tx+1, ny/ty+1);
	/* Number of threads per block. */
	dim3 threads(tx, ty);

	render<<<blocks, threads>>> (fb, nx, ny, lower_left_corner, vertical, horizontal, origin); 
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
	checkCudaErrors(cudaFree(fb));

	out_ppm.close();

	return 0;
}