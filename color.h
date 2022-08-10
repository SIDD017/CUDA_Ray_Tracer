#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"
#include <fstream>

inline float clamp(float x, float min, float max)
{
	if (x < min) {
		return min;
	}
	if (x > max) {
		return max;
	}
	return x;
}

void write_color(std::ofstream &out, color pixel_color)
{
	float r = sqrt(pixel_color.x());
	float g = sqrt(pixel_color.y());
	float b = sqrt(pixel_color.z());

	out << static_cast<int>(255.99f * r) << ' '
		<< static_cast<int>(255.99f * g) << ' '
		<< static_cast<int>(255.99f * b) << '\n';
}

#endif