#ifndef CAMERA_H
#define CAMERA_H

#include <cuda_runtime.h>

#include "vec3.h"
#include "ray.h"

__device__ vec3 random_in_unit_disk(curandState rand_state)
{
	while (true) {
		vec3 p = vec3((curand_uniform(&rand_state) * 2) - 1, (curand_uniform(&rand_state) * 2) - 1, 0.0f);
		if (p.length_squared() >= 1.0f) {
			continue;
		}
		return p;
	}
}

class camera
{
public:
	__device__ camera(point3 lookfrom, point3 lookat, vec3 vup, float vfov, float aspect_ratio, float aperture, float focus_dist)
	{
		float theta = degrees_to_radians(vfov);
		float h = tan(theta / 2.0f);
		float viewport_height = 2.0f * h;
		float viewport_width = aspect_ratio * viewport_height;

		w = unit_length(lookfrom - lookat);
		u = unit_length(cross(vup, w));
		v = cross(w, u);

		float focal_length = 1.0f;

		origin = lookfrom;
		horizontal = focus_dist * viewport_width * u;
		vertical = focus_dist * viewport_height * v;
		lower_left_corner = origin - (horizontal / 2.0f) - (vertical / 2.0f) - focus_dist * w;

		lens_radius = aperture / 2.0f;
	}

	__device__ ray get_ray(float s, float t, curandState rand_state)
	{
		vec3 rd = lens_radius * random_in_unit_disk(rand_state);
		vec3 offset = u * rd.x() + v * rd.y();

		return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
	}

private:
	point3 origin;
	point3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	float lens_radius;
};

#endif