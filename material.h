#ifndef MATERIAL_H
#define MATERIAL_H

#include <cuda_runtime.h>

#include "essentials.h"

struct hit_record;

class material
{
public:
	__device__ virtual bool scatter(const ray &r_in, const hit_record& rec, color &attenuation, ray &scattered, curandState *rand_state) const = 0;
};


__device__ vec3 random_in_unit_sphere(curandState* local_rand_state)
{
	/* When using #random_in_unit_sphere() instead of #random_in_hemisphere(), the result might
	become NaN or infinite due to the random vector pointing opposite to the normal. To avoid
	this, check for a cases where the result is close to 0.0f and just change the value to the
	normal vector instead. */
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

class lambertian : public material
{
public:
	__device__ lambertian(const color& a)
	{
		albedo = a;
	}

	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *rand_state) const override
	{
		vec3 scatter_direction = rec.normal + random_in_hemisphere(rec.normal, rand_state);
		scattered = ray(rec.p, scatter_direction, r_in.time());
		attenuation = albedo;
		return true;
	}

	color albedo;
};

class metal : public material
{
public:
	__device__ metal(const color& a, float f)
	{
		albedo = a;
		fuzz = (f < 1.0f) ? f : 1.0f;
	}

	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *rand_state) const override
	{
		vec3 reflected = reflect(unit_length(r_in.direction()), rec.normal);
		scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(rand_state), r_in.time());
		attenuation = albedo;
		return (dot(scattered.direction(), rec.normal) > 0);
	}

	color albedo;
	float fuzz;
};

class dielectric : public material
{
public:
	__device__ dielectric(float index_of_refraction)
	{
		ir = index_of_refraction;
	}

	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* rand_state) const override
	{
		attenuation = color(1.0f, 1.0f, 1.0f);
		float refraction_ratio = rec.front_face ? (1.0f / ir) : ir;

		vec3 unit_direction = unit_length(r_in.direction());
		float cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0f);
		float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

		bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
		vec3 direction;

		if (cannot_refract || reflectance(cos_theta, refraction_ratio) > curand_uniform(rand_state)) {
			direction = reflect(unit_direction, rec.normal);
		}
		else {
			direction = refract(unit_direction, rec.normal, refraction_ratio);
		}

		scattered = ray(rec.p, direction, r_in.time());
		return true;
	}

	double ir;

private:
	__device__ static float reflectance(float cosine, float ref_index)
	{
		float r0 = (1 - ref_index) / (1 + ref_index);
		r0 = r0 * r0;
		return r0 + (1 - r0) * powf((1 - cosine), 5.0f);
	}
};

#endif