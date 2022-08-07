#ifndef COLORH
#define COLORH

#include "vec3.h"
#include <fstream>

void write_color(std::ofstream &out, color pixel_color)
{
	out << static_cast<int>(255.99 * pixel_color.x()) << ' '
		<< static_cast<int>(255.99 * pixel_color.y()) << ' '
		<< static_cast<int>(255.99 * pixel_color.z()) << '\n';
}

#endif