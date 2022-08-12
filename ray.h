#ifndef RAYH
#define RAYH

#include <cuda_runtime.h>

#include "vec3.h"

class ray
{
public:
	__device__ ray() {}
	__device__ ray(const point3& origin, const vec3& direction, float time = 0.0f)
	{
		dir = direction;
		orig = origin;
		tm = time;
	}

	__device__ vec3 direction() const { return dir; }
	__device__ point3 origin() const { return orig; }
	__device__ float time() const { return tm; }

	__device__ point3 at(float t) const
	{
		return (orig + (t * dir));
	}

	point3 orig;
	vec3 dir;
	float tm;
};

#endif