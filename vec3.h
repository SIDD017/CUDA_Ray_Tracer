#ifndef VEC3H
#define VEC3H

#include <cuda_runtime.h>

class vec3
{
public:
	__host__ __device__ vec3() { e[0] = 0.0f; e[1] = 0.0f; e[2] = 0.0f; }
	__host__ __device__ vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2;}

	__host__ __device__ inline float x() const { return e[0]; }
	__host__ __device__ inline float y() const { return e[1]; }
	__host__ __device__ inline float z() const { return e[2]; }

	__host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
	__host__ __device__ inline float operator[](int i) const { return e[i]; }
	__host__ __device__ inline float& operator[](int i) { return e[i]; }

	__host__ __device__ inline vec3& operator+=(const vec3& v) 
	{
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];
		return *this;
	}

	__host__ __device__ inline vec3& operator-=(const vec3& v)
	{
		e[0] -= v.e[0];
		e[1] -= v.e[1];
		e[2] -= v.e[2];
		return *this;
	}

	__host__ __device__ inline vec3& operator*=(const float t)
	{
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;
		return *this;
	}

	__host__ __device__ inline vec3& operator*=(const vec3& v)
	{
		e[0] *= v.e[0];
		e[1] *= v.e[1];
		e[2] *= v.e[2];
		return *this;
	}

	__host__ __device__ inline vec3& operator/=(const float t)
	{
		e[0] /= t;
		e[1] /= t;
		e[2] /= t;
		return *this;
	}

	__host__ __device__ inline float length() const
	{
		return sqrt(length_squared());
	}

	__host__ __device__ inline float length_squared() const
	{
		return ((e[0] * e[0]) + (e[1] * e[1]) + (e[2] * e[2]));
	}

	float e[3];
};

/* NOTE:
* The functions defined in the class definition require access to the class veriable e[3].
* The following functions are independent of the class variables and use the operands specified 
* in the function parameters, hence they are defined outside the class definition. */

inline std::ostream& operator<<(std::ostream& out, const vec3& v)
{
	return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v)
{
	return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v)
{
	return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v)
{
	return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const float t, const vec3& v)
{
	return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

/* Looks similar to the earlier definition of * operator buT this one handles cases when the 
order of operands is reversed .i.e when it is (v * t) instead of (t * v) */
__host__ __device__ inline vec3 operator*(const vec3& v, const float t)
{
	return t * v;
}

__host__ __device__ inline vec3 operator/(const vec3& v, const float t)
{
	return (1 / t) * v;
}

__host__ __device__ inline float dot(const vec3& u, const vec3& v)
{
	return ((u.e[0] * v.e[0]) + 
			(u.e[1] * v.e[1]) + 
			(u.e[2] * v.e[2]));
}

/* Use the matrix/determinant form to calculate the cross product. */
__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v)
{
	return vec3((u.e[1] * v.e[2]) - (u.e[2] * v.e[1]),
				(u.e[2] * v.e[0]) - (u.e[0] * v.e[2]),
				(u.e[0] * v.e[1]) - (u.e[1] * v.e[0]));
}

__host__ __device__ vec3 unit_length(vec3& v)
{
	return (v / v.length());
}

__host__ __device__ vec3 reflect(const vec3& v, const vec3& n)
{
	return v - 2 * dot(v, n) * n;
}

__host__ __device__ vec3 refract(const vec3 &uv, const vec3& n, float etai_over_etat)
{
	float cos_theta = fmin(dot(-uv, n), 1.0f);
	vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
	vec3 r_out_parallel = -sqrt(fabs(1.0f - r_out_perp.length_squared())) * n;
	return r_out_perp + r_out_parallel;
}

/* Aliases for vec3. */
using point3 = vec3;
using color = vec3;

#endif