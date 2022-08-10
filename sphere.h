#ifndef SPHERE_H
#define SPHERE_H

#include <cuda_runtime.h>

#include <math.h>

#include "hittable.h"
#include "vec3.h"

class sphere : public hittable
{
public:
	__device__ sphere() {}
	__device__ sphere(point3 cen, float r, material *mtr)
	{
		center = cen;
		radius = r;
		mtr_ptr = mtr;
	}

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;


	point3 center;
	float radius;
	material *mtr_ptr;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
	vec3 oc = r.origin() - center;
	float a = r.direction().length_squared();
	float half_b = dot(oc, r.direction());
	float c = oc.length_squared() - radius * radius;

	float discriminant = half_b * half_b - a * c;
	if (discriminant < 0.0f) {
		return false;
	}
	float sqrtd = sqrt(discriminant);

	float root = (-half_b - sqrtd) / a;
	if (root < t_min || t_max < root) {
		root = (-half_b + sqrtd) / a;
		if (root < t_min || t_max < root) {
			return false;
		}
	}

	rec.t = root;
	rec.p = r.at(rec.t);
	vec3 outward_normal = (rec.p - center) / radius;
	rec.set_face_normal(r, outward_normal);
	rec.mtr = mtr_ptr;

	return true;
}

#endif