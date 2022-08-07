#include "cuda_runtime.h"
#include <device_launch_parameters.h>

#include <iostream>
#include <fstream>

#include "vec3.h"
#include "color.h"

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

__global__ void render(float *fb, int max_x, int max_y)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	/* Early return if the current thread is not mapped to any pixel in the final render. */
	if ((i >= max_x) || (j >= max_y)) {
		return;
	}

	int pixel_index = j * max_x * 3 + i * 3;
	fb[pixel_index + 0] = float(i) / max_x;
	fb[pixel_index + 1] = float(j) / max_y;
	fb[pixel_index + 2] = 0.2;
}

int main(void)
{
	/* Image size. */
	int nx = 1280, ny = 720;
	int num_pixels = nx * ny;

	/* Size of frame buffer in Unified memory to hold final pixel values. */
	size_t fb_size = 3 * num_pixels * sizeof(float);

	/* Allocate Unified memory for framebuffer */
	float* fb;
	checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

	/* Thread size for dividing work on GPU. */
	int tx = 8, ty = 8;

	/* Number of required blocks. */
	dim3 blocks(nx/tx+1, ny/ty+1);
	/* Number of threads per block. */
	dim3 threads(tx, ty);

	render<<<blocks, threads>>> (fb, nx, ny);
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
			size_t pixel_index = j * 3 * nx + i * 3;
			color pixel_color(fb[pixel_index + 0], fb[pixel_index + 1], fb[pixel_index + 2]);
			write_color(out_ppm, pixel_color);
		}
	}

	std::cout << "Done writing to output file\n";
	checkCudaErrors(cudaFree(fb));

	out_ppm.close();

	return 0;
}