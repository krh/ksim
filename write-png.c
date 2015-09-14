#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <error.h>
#include <errno.h>

#include <libpng16/png.h>

#include "write-png.h"

static void
convert_to_bytes(png_structp png, png_row_infop row_info, png_bytep data)
{
   for (uint32_t i = 0; i < row_info->rowbytes; i += 4) {
      uint8_t *b = &data[i];
      uint32_t pixel;

      memcpy (&pixel, b, sizeof (uint32_t));
      b[0] = (pixel & 0xff0000) >> 16;
      b[1] = (pixel & 0x00ff00) >>  8;
      b[2] = (pixel & 0x0000ff) >>  0;
      b[3] = 0xff;
   }
}

void
write_png(const char *path, int32_t width, int32_t height, int32_t stride, void *pixels)
{
   FILE *f = NULL;
   png_structp png_writer = NULL;
   png_infop png_info = NULL;

   uint8_t *rows[height];

   for (int32_t y = 0; y < height; y++)
	   rows[y] = pixels + y * stride;

   f = fopen(path, "wb");
   if (f == NULL)
	   error(-1, errno, "failed to open file for writing: %s", path);

   png_writer = png_create_write_struct(PNG_LIBPNG_VER_STRING,
					NULL, NULL, NULL);
   if (png_writer == NULL)
	   error(-1, 0, "failed to create png writer");

   png_info = png_create_info_struct(png_writer);
   if (png_info == NULL)
	   error(-1, 0, "failed to create png writer info");

   png_init_io(png_writer, f);
   png_set_IHDR(png_writer, png_info,
		width, height,
		8, PNG_COLOR_TYPE_RGBA,
		PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
		PNG_FILTER_TYPE_DEFAULT);
   png_write_info(png_writer, png_info);
   png_set_rows(png_writer, png_info, rows);
   png_set_write_user_transform_fn(png_writer, convert_to_bytes);
   png_write_png(png_writer, png_info, PNG_TRANSFORM_IDENTITY, NULL);

   png_destroy_write_struct(&png_writer, &png_info);

   fclose(f);
}
