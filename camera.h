#ifndef CAMERA_H
#define CAMERA_H

#include <cuda_runtime.h>

#include "vec3.h"
#include "ray.h"

class camera
{
public:
	__device__ camera(point3 lookfrom, point3 lookat, vec3 vup, float vfov, float aspect_ratio)
	{
		float theta = degrees_to_radians(vfov);
		float h = tan(theta / 2.0f);
		float viewport_height = 2.0f * h;
		float viewport_width = aspect_ratio * viewport_height;

		vec3 w = unit_length(lookfrom - lookat);
		vec3 u = unit_length(cross(vup, w));
		vec3 v = cross(w, u);

		float focal_length = 1.0f;

		origin = lookfrom;
		horizontal = viewport_width * u;
		vertical = viewport_height * v;
		lower_left_corner = origin - w - (horizontal / 2.0f) - (vertical / 2.0f);

	}

	__device__ ray get_ray(float s, float t)
	{
		return ray(origin, lower_left_corner + s * horizontal + t * vertical - origin);
	}

private:
	point3 origin;
	point3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
};

#endif