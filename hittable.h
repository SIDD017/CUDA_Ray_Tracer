#ifndef HITTABLE_H
#define HITTABLE_H

#include <cuda_runtime.h>

#include "ray.h"

class material;

/* For a given ray, this struct hold all the information related to the object the ray hits. */
struct hit_record 
{
	float t;
	vec3 normal;
	point3 p;
	material* mtr;
	bool front_face;

	__device__ inline void set_face_normal(const ray& r, const vec3& outward_normal) {
		front_face = dot(r.direction(), outward_normal) < 0.0f;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class hittable
{
public:
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif