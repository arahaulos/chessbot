#pragma once

#include <string>
#include <stdint.h>

struct bmp_image_utility {
    static void save_pixels(std::string path, uint8_t *pixels, int width, int height, int bpp);
};
