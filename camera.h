#ifndef CAMERA_H
#define CAMERA_H

#include <cuda_runtime.h>

#include "vec3.h"
#include "ray.h"

class camera
{
public:
	__device__ camera()
	{
		float aspect_ratio = 16.0 / 9.0;
		float viewport_height = 2.0f;
		float viewport_width = aspect_ratio * viewport_height;
		float focal_length = 1.0f;

		origin = point3(0.0f, 0.0f, 0.0f);
		horizontal = vec3(viewport_width, 0.0f, 0.0f);
		vertical = vec3(0.0f, viewport_height, 0.0f);
		lower_left_corner = origin - vec3(0.0f, 0.0f, focal_length) - (horizontal / 2.0f) - (vertical / 2.0f);
	}

	__device__ ray get_ray(float u, float v)
	{
		return ray(origin, lower_left_corner + (u * horizontal) + (v * vertical) - origin);
	}

private:
	point3 origin;
	point3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
};

#endif