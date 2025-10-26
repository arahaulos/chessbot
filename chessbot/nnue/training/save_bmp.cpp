#include <iostream>
#include <fstream>
#include "save_bmp.hpp"
#include <cstring>


struct __attribute__((packed)) bmp_fileheader {
	int8_t signature[2];
	uint32_t filesize;
	uint32_t reserved;
	uint32_t dataoffset;
};

struct __attribute__((packed)) bmp_infoheader
{
	uint32_t size;
	uint32_t width;
	uint32_t height;
	uint16_t planes;
	uint16_t bit_count;
	uint32_t compression;
	uint32_t image_size;
	uint32_t x_pixels_per_m;
	uint32_t y_pixels_per_m;
	uint32_t colors_used;
	uint32_t color_important;
};


void bmp_image_utility::save_pixels(std::string path, uint8_t *pixels, int width, int height, int bpp)
{
    bmp_fileheader fhdr;
    bmp_infoheader ihdr;

    std::memset((void*)&fhdr, 0, sizeof(bmp_fileheader));
    std::memset((void*)&ihdr, 0, sizeof(bmp_infoheader));

    fhdr.signature[0] = 'B';
    fhdr.signature[1] = 'M';
    fhdr.filesize = (bpp/8)*width*height + sizeof(bmp_fileheader) + sizeof(ihdr);
    fhdr.dataoffset = sizeof(bmp_fileheader) + sizeof(ihdr);

    ihdr.bit_count = bpp;
    ihdr.compression = 0;
    ihdr.image_size = (bpp/8)*width*height;
    ihdr.height = height;
    ihdr.width = width;
    ihdr.y_pixels_per_m = height;
    ihdr.x_pixels_per_m = width;
    ihdr.size = sizeof(ihdr);
    ihdr.planes = 1;

    ihdr.colors_used = 0;
    ihdr.color_important = 0;

    std::ofstream file(path, std::ios::binary);
    if (file.is_open()) {

        file.write((char*)&fhdr, sizeof(bmp_fileheader));
        file.write((char*)&ihdr, sizeof(bmp_infoheader));
        file.write((char*)pixels, width*height*(bpp/8));

        file.close();
    }
}
