/*
 * Copyright © 2015 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "ksim.h"

#define V 1
#define SRGB 2

static const struct format_info {
	uint32_t size;
	uint32_t channels;
	uint32_t block_size;
	uint32_t caps;
} formats[] = {
	[SF_R32G32B32A32_FLOAT]			= { .size = 16, .channels = 4, .block_size = 1, .caps = V },
	[SF_R32G32B32A32_SINT]			= { .size = 16, .channels = 4, .block_size = 1, .caps = V },
	[SF_R32G32B32A32_UINT]			= { .size = 16, .channels = 4, .block_size = 1, .caps = V },
	[SF_R32G32B32A32_UNORM]			= { .size = 16, .channels = 4, .block_size = 1, .caps = V },
	[SF_R32G32B32A32_SNORM]			= { .size = 16, .channels = 4, .block_size = 1, .caps = V },
	[SF_R64G64_FLOAT]			= { .size = 16, .channels = 2, .block_size = 1, .caps = V },
	[SF_R32G32B32X32_FLOAT]			= { .size = 16, .channels = 4, .block_size = 1, .caps = V },
	[SF_R32G32B32A32_SSCALED]		= { .size = 16, .channels = 4, .block_size = 1, .caps = V },
	[SF_R32G32B32A32_USCALED]		= { .size = 16, .channels = 4, .block_size = 1, .caps = V },
	[SF_R32G32B32A32_SFIXED]		= { .size = 16, .channels = 4, .block_size = 1, .caps = V },
	[SF_R64G64_PASSTHRU]			= { .size = 16, .channels = 2, .block_size = 1, .caps = V },
	[SF_R32G32B32_FLOAT]			= { .size = 12, .channels = 3, .block_size = 1, .caps = V },
	[SF_R32G32B32_SINT]			= { .size = 12, .channels = 3, .block_size = 1, .caps = V },
	[SF_R32G32B32_UINT]			= { .size = 12, .channels = 3, .block_size = 1, .caps = V },
	[SF_R32G32B32_UNORM]			= { .size = 12, .channels = 3, .block_size = 1, .caps = V },
	[SF_R32G32B32_SNORM]			= { .size = 12, .channels = 3, .block_size = 1, .caps = V },
	[SF_R32G32B32_SSCALED]			= { .size = 12, .channels = 3, .block_size = 1, .caps = V },
	[SF_R32G32B32_USCALED]			= { .size = 12, .channels = 3, .block_size = 1, .caps = V },
	[SF_R32G32B32_SFIXED]			= { .size = 12, .channels = 3, .block_size = 1, .caps = V },
	[SF_R16G16B16A16_UNORM]			= { .size =  8, .channels = 4, .block_size = 1, .caps = V },
	[SF_R16G16B16A16_SNORM]			= { .size =  8, .channels = 4, .block_size = 1, .caps = V },
	[SF_R16G16B16A16_SINT]			= { .size =  8, .channels = 4, .block_size = 1, .caps = V },
	[SF_R16G16B16A16_UINT]			= { .size =  8, .channels = 4, .block_size = 1, .caps = V },
	[SF_R16G16B16A16_FLOAT]			= { .size =  8, .channels = 4, .block_size = 1, .caps = V },
	[SF_R32G32_FLOAT]			= { .size =  8, .channels = 2, .block_size = 1, .caps = V },
	[SF_R32G32_SINT]			= { .size =  8, .channels = 2, .block_size = 1, .caps = V },
	[SF_R32G32_UINT]			= { .size =  8, .channels = 2, .block_size = 1, .caps = V },
	[SF_R32_FLOAT_X8X24_TYPELESS]		= { .size =  8, .channels = 3, .block_size = 1, .caps = 0 },
	[SF_X32_TYPELESS_G8X24_UINT]		= { .size =  8, .channels = 3, .block_size = 1, .caps = 0 },
	[SF_L32A32_FLOAT]			= { .size =  8, .channels = 2, .block_size = 1, .caps = 0 },
	[SF_R32G32_UNORM]			= { .size =  8, .channels = 2, .block_size = 1, .caps = V },
	[SF_R32G32_SNORM]			= { .size =  8, .channels = 2, .block_size = 1, .caps = V },
	[SF_R64_FLOAT]				= { .size =  8, .channels = 1, .block_size = 1, .caps = V },
	[SF_R16G16B16X16_UNORM]			= { .size =  8, .channels = 4, .block_size = 1, .caps = V },
	[SF_R16G16B16X16_FLOAT]			= { .size =  8, .channels = 4, .block_size = 1, .caps = V },
	[SF_A32X32_FLOAT]			= { .size =  8, .channels = 2, .block_size = 1, .caps = 0 },
	[SF_L32X32_FLOAT]			= { .size =  8, .channels = 2, .block_size = 1, .caps = 0 },
	[SF_I32X32_FLOAT]			= { .size =  8, .channels = 2, .block_size = 1, .caps = 0 },
	[SF_R16G16B16A16_SSCALED]		= { .size =  8, .channels = 4, .block_size = 1, .caps = V },
	[SF_R16G16B16A16_USCALED]		= { .size =  8, .channels = 4, .block_size = 1, .caps = V },
	[SF_R32G32_SSCALED]			= { .size =  8, .channels = 2, .block_size = 1, .caps = V },
	[SF_R32G32_USCALED]			= { .size =  8, .channels = 2, .block_size = 1, .caps = V },
	[SF_R32G32_SFIXED]			= { .size =  8, .channels = 2, .block_size = 1, .caps = V },
	[SF_R64_PASSTHRU]			= { .size =  8, .channels = 1, .block_size = 1, .caps = V },
	[SF_B8G8R8A8_UNORM]			= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_B8G8R8A8_UNORM_SRGB]		= { .size =  4, .channels = 4, .block_size = 1, .caps = V | SRGB },
	[SF_R10G10B10A2_UNORM]			= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_R10G10B10A2_UNORM_SRGB]		= { .size =  4, .channels = 4, .block_size = 1, .caps = V | SRGB },
	[SF_R10G10B10A2_UINT]			= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_R10G10B10_SNORM_A2_UNORM]		= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_R8G8B8A8_UNORM]			= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_R8G8B8A8_UNORM_SRGB]		= { .size =  4, .channels = 4, .block_size = 1, .caps = V | SRGB },
	[SF_R8G8B8A8_SNORM]			= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_R8G8B8A8_SINT]			= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_R8G8B8A8_UINT]			= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_R16G16_UNORM]			= { .size =  4, .channels = 2, .block_size = 1, .caps = V },
	[SF_R16G16_SNORM]			= { .size =  4, .channels = 2, .block_size = 1, .caps = V },
	[SF_R16G16_SINT]			= { .size =  4, .channels = 2, .block_size = 1, .caps = V },
	[SF_R16G16_UINT]			= { .size =  4, .channels = 2, .block_size = 1, .caps = V },
	[SF_R16G16_FLOAT]			= { .size =  4, .channels = 2, .block_size = 1, .caps = V },
	[SF_B10G10R10A2_UNORM]			= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_B10G10R10A2_UNORM_SRGB]		= { .size =  4, .channels = 4, .block_size = 1, .caps = V | SRGB },
	[SF_R11G11B10_FLOAT]			= { .size =  4, .channels = 3, .block_size = 1, .caps = V },
	[SF_R32_SINT]				= { .size =  4, .channels = 1, .block_size = 1, .caps = V },
	[SF_R32_UINT]				= { .size =  4, .channels = 1, .block_size = 1, .caps = V },
	[SF_R32_FLOAT]				= { .size =  4, .channels = 1, .block_size = 1, .caps = V },
	[SF_R24_UNORM_X8_TYPELESS]		= { .size =  4, .channels = 2, .block_size = 1, .caps = V },
	[SF_X24_TYPELESS_G8_UINT]		= { .size =  4, .channels = 2, .block_size = 1, .caps = V },
	[SF_L32_UNORM]				= { .size =  4, .channels = 1, .block_size = 1, .caps = 0 },
	[SF_A32_UNORM]				= { .size =  4, .channels = 1, .block_size = 1, .caps = 0 },
	[SF_L16A16_UNORM]			= { .size =  4, .channels = 2, .block_size = 1, .caps = 0 },
	[SF_I24X8_UNORM]			= { .size =  4, .channels = 2, .block_size = 1, .caps = 0 },
	[SF_L24X8_UNORM]			= { .size =  4, .channels = 2, .block_size = 1, .caps = 0 },
	[SF_A24X8_UNORM]			= { .size =  4, .channels = 2, .block_size = 1, .caps = 0 },
	[SF_I32_FLOAT]				= { .size =  4, .channels = 1, .block_size = 1, .caps = 0 },
	[SF_L32_FLOAT]				= { .size =  4, .channels = 1, .block_size = 1, .caps = 0 },
	[SF_A32_FLOAT]				= { .size =  4, .channels = 1, .block_size = 1, .caps = 0 },
	[SF_X8B8_UNORM_G8R8_SNORM]		= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_A8X8_UNORM_G8R8_SNORM]		= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_B8X8_UNORM_G8R8_SNORM]		= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_B8G8R8X8_UNORM]			= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_B8G8R8X8_UNORM_SRGB]		= { .size =  4, .channels = 4, .block_size = 1, .caps = V | SRGB },
	[SF_R8G8B8X8_UNORM]			= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_R8G8B8X8_UNORM_SRGB]		= { .size =  4, .channels = 4, .block_size = 1, .caps = V | SRGB },
	[SF_R9G9B9E5_SHAREDEXP]			= { .size =  4, .channels = 3, .block_size = 1, .caps = V },
	[SF_B10G10R10X2_UNORM]			= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_L16A16_FLOAT]			= { .size =  4, .channels = 2, .block_size = 1, .caps = V },
	[SF_R32_UNORM]				= { .size =  4, .channels = 1, .block_size = 1, .caps = V },
	[SF_R32_SNORM]				= { .size =  4, .channels = 1, .block_size = 1, .caps = V },
	[SF_R10G10B10X2_USCALED]		= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_R8G8B8A8_SSCALED]			= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_R8G8B8A8_USCALED]			= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_R16G16_SSCALED]			= { .size =  4, .channels = 2, .block_size = 1, .caps = V },
	[SF_R16G16_USCALED]			= { .size =  4, .channels = 2, .block_size = 1, .caps = V },
	[SF_R32_SSCALED]			= { .size =  4, .channels = 1, .block_size = 1, .caps = V },
	[SF_R32_USCALED]			= { .size =  4, .channels = 1, .block_size = 1, .caps = V },
	[SF_B5G6R5_UNORM]			= { .size =  2, .channels = 3, .block_size = 1, .caps = V },
	[SF_B5G6R5_UNORM_SRGB]			= { .size =  2, .channels = 3, .block_size = 1, .caps = V | SRGB },
	[SF_B5G5R5A1_UNORM]			= { .size =  2, .channels = 4, .block_size = 1, .caps = V },
	[SF_B5G5R5A1_UNORM_SRGB]		= { .size =  2, .channels = 4, .block_size = 1, .caps = V | SRGB },
	[SF_B4G4R4A4_UNORM]			= { .size =  2, .channels = 4, .block_size = 1, .caps = V },
	[SF_B4G4R4A4_UNORM_SRGB]		= { .size =  2, .channels = 4, .block_size = 1, .caps = V | SRGB },
	[SF_R8G8_UNORM]				= { .size =  2, .channels = 2, .block_size = 1, .caps = V },
	[SF_R8G8_SNORM]				= { .size =  2, .channels = 2, .block_size = 1, .caps = V },
	[SF_R8G8_SINT]				= { .size =  2, .channels = 2, .block_size = 1, .caps = V },
	[SF_R8G8_UINT]				= { .size =  2, .channels = 2, .block_size = 1, .caps = V },
	[SF_R16_UNORM]				= { .size =  2, .channels = 1, .block_size = 1, .caps = V },
	[SF_R16_SNORM]				= { .size =  2, .channels = 1, .block_size = 1, .caps = V },
	[SF_R16_SINT]				= { .size =  2, .channels = 1, .block_size = 1, .caps = V },
	[SF_R16_UINT]				= { .size =  2, .channels = 1, .block_size = 1, .caps = V },
	[SF_R16_FLOAT]				= { .size =  2, .channels = 1, .block_size = 1, .caps = V },
	[SF_A8P8_UNORM_PALETTE0]		= { .size =  2, .channels = 2, .block_size = 1, .caps = V },
	[SF_A8P8_UNORM_PALETTE1]		= { .size =  2, .channels = 2, .block_size = 1, .caps = V },
	[SF_I16_UNORM]				= { .size =  2, .channels = 1, .block_size = 1, .caps = V },
	[SF_L16_UNORM]				= { .size =  2, .channels = 1, .block_size = 1, .caps = V },
	[SF_A16_UNORM]				= { .size =  2, .channels = 1, .block_size = 1, .caps = V },
	[SF_L8A8_UNORM]				= { .size =  2, .channels = 2, .block_size = 1, .caps = V },
	[SF_I16_FLOAT]				= { .size =  2, .channels = 1, .block_size = 1, .caps = V },
	[SF_L16_FLOAT]				= { .size =  2, .channels = 1, .block_size = 1, .caps = V },
	[SF_A16_FLOAT]				= { .size =  2, .channels = 1, .block_size = 1, .caps = V },
	[SF_L8A8_UNORM_SRGB]			= { .size =  2, .channels = 2, .block_size = 1, .caps = V | SRGB },
	[SF_R5G5_SNORM_B6_UNORM]		= { .size =  2, .channels = 3, .block_size = 1, .caps = V },
	[SF_B5G5R5X1_UNORM]			= { .size =  2, .channels = 4, .block_size = 1, .caps = V },
	[SF_B5G5R5X1_UNORM_SRGB]		= { .size =  2, .channels = 4, .block_size = 1, .caps = V | SRGB },
	[SF_R8G8_SSCALED]			= { .size =  2, .channels = 2, .block_size = 1, .caps = V },
	[SF_R8G8_USCALED]			= { .size =  2, .channels = 2, .block_size = 1, .caps = V },
	[SF_R16_SSCALED]			= { .size =  2, .channels = 1, .block_size = 1, .caps = V },
	[SF_R16_USCALED]			= { .size =  2, .channels = 1, .block_size = 1, .caps = V },
	[SF_P8A8_UNORM_PALETTE0]		= { .size =  2, .channels = 2, .block_size = 1, .caps = V },
	[SF_P8A8_UNORM_PALETTE1]		= { .size =  2, .channels = 2, .block_size = 1, .caps = V },
	[SF_A1B5G5R5_UNORM]			= { .size =  2, .channels = 4, .block_size = 1, .caps = V },
	[SF_A4B4G4R4_UNORM]			= { .size =  2, .channels = 4, .block_size = 1, .caps = V },
	[SF_L8A8_UINT]				= { .size =  2, .channels = 2, .block_size = 1, .caps = V },
	[SF_L8A8_SINT]				= { .size =  2, .channels = 2, .block_size = 1, .caps = V },
	[SF_R8_UNORM]				= { .size =  1, .channels = 1, .block_size = 1, .caps = V },
	[SF_R8_SNORM]				= { .size =  1, .channels = 1, .block_size = 1, .caps = V },
	[SF_R8_SINT]				= { .size =  1, .channels = 1, .block_size = 1, .caps = V },
	[SF_R8_UINT]				= { .size =  1, .channels = 1, .block_size = 1, .caps = V },
	[SF_A8_UNORM]				= { .size =  1, .channels = 1, .block_size = 1, .caps = V },
	[SF_I8_UNORM]				= { .size =  1, .channels = 1, .block_size = 1, .caps = V },
	[SF_L8_UNORM]				= { .size =  1, .channels = 1, .block_size = 1, .caps = V },
	[SF_P4A4_UNORM_PALETTE0]		= { .size =  1, .channels = 2, .block_size = 1, .caps = V },
	[SF_A4P4_UNORM_PALETTE0]		= { .size =  1, .channels = 2, .block_size = 1, .caps = V },
	[SF_R8_SSCALED]				= { .size =  1, .channels = 1, .block_size = 1, .caps = V },
	[SF_R8_USCALED]				= { .size =  1, .channels = 1, .block_size = 1, .caps = V },
	[SF_P8_UNORM_PALETTE0]			= { .size =  1, .channels = 1, .block_size = 1, .caps = V },
	[SF_L8_UNORM_SRGB]			= { .size =  1, .channels = 1, .block_size = 1, .caps = V | SRGB },
	[SF_P8_UNORM_PALETTE1]			= { .size =  1, .channels = 1, .block_size = 1, .caps = V },
	[SF_P4A4_UNORM_PALETTE1]		= { .size =  1, .channels = 2, .block_size = 1, .caps = V },
	[SF_A4P4_UNORM_PALETTE1]		= { .size =  1, .channels = 2, .block_size = 1, .caps = V },
	[SF_Y8_UNORM]				= { .size =  1, .channels = 1, .block_size = 1, .caps = V },
	[SF_L8_UINT]				= { .size =  1, .channels = 1, .block_size = 1, .caps = V },
	[SF_L8_SINT]				= { .size =  1, .channels = 1, .block_size = 1, .caps = V },
	[SF_I8_UINT]				= { .size =  1, .channels = 1, .block_size = 1, .caps = V },
	[SF_I8_SINT]				= { .size =  1, .channels = 1, .block_size = 1, .caps = V },
	[SF_DXT1_RGB_SRGB]			= { .size =  1, .channels = 3, .block_size = 1, .caps = V | SRGB },
	[SF_R1_UNORM]				= { .size =  1, .channels = 1, .block_size = 1, .caps = V },
	[SF_YCRCB_NORMAL]			= { .size =  0, .channels = 3, .block_size = 1, .caps = 0 },
	[SF_YCRCB_SWAPUVY]			= { .size =  0, .channels = 3, .block_size = 1, .caps = 0 },
	[SF_P2_UNORM_PALETTE0]			= { .size =  0, .channels = 1, .block_size = 1, .caps = 0 },
	[SF_P2_UNORM_PALETTE1]			= { .size =  0, .channels = 1, .block_size = 1, .caps = 0 },
	[SF_BC1_UNORM]				= { .size =  0, .channels = 3, .block_size = 4, .caps = 0 },
	[SF_BC2_UNORM]				= { .size =  0, .channels = 3, .block_size = 4, .caps = 0 },
	[SF_BC3_UNORM]				= { .size =  0, .channels = 3, .block_size = 4, .caps = 0 },
	[SF_BC4_UNORM]				= { .size =  0, .channels = 3, .block_size = 4, .caps = 0 },
	[SF_BC5_UNORM]				= { .size =  0, .channels = 3, .block_size = 4, .caps = 0 },
	[SF_BC1_UNORM_SRGB]			= { .size =  0, .channels = 3, .block_size = 4, .caps = 0 | SRGB },
	[SF_BC2_UNORM_SRGB]			= { .size =  0, .channels = 3, .block_size = 4, .caps = 0 | SRGB },
	[SF_BC3_UNORM_SRGB]			= { .size =  0, .channels = 3, .block_size = 4, .caps = 0 | SRGB },
	[SF_MONO8]				= { .size =  0, .channels = 1, .block_size = 1, .caps = 0 },
	[SF_YCRCB_SWAPUV]			= { .size =  0, .channels = 3, .block_size = 1, .caps = 0 },
	[SF_YCRCB_SWAPY]			= { .size =  0, .channels = 3, .block_size = 1, .caps = 0 },
	[SF_DXT1_RGB]				= { .size =  0, .channels = 3, .block_size = 1, .caps = 0 },
	[SF_FXT1]				= { .size =  0, .channels = 3, .block_size = 1, .caps = 0 },
	[SF_R8G8B8_UNORM]			= { .size =  3, .channels = 3, .block_size = 1, .caps = V },
	[SF_R8G8B8_SNORM]			= { .size =  3, .channels = 3, .block_size = 1, .caps = V },
	[SF_R8G8B8_SSCALED]			= { .size =  3, .channels = 3, .block_size = 1, .caps = V },
	[SF_R8G8B8_USCALED]			= { .size =  3, .channels = 3, .block_size = 1, .caps = V },
	[SF_R64G64B64A64_FLOAT]			= { .size = 32, .channels = 4, .block_size = 1, .caps = V },
	[SF_R64G64B64_FLOAT]			= { .size = 24, .channels = 3, .block_size = 1, .caps = V },
	[SF_BC4_SNORM]				= { .size =  0, .channels = 3, .block_size = 4, .caps = 0 },
	[SF_BC5_SNORM]				= { .size =  0, .channels = 3, .block_size = 4, .caps = 0 },
	[SF_R16G16B16_FLOAT]			= { .size =  6, .channels = 3, .block_size = 1, .caps = V },
	[SF_R16G16B16_UNORM]			= { .size =  6, .channels = 3, .block_size = 1, .caps = V },
	[SF_R16G16B16_SNORM]			= { .size =  6, .channels = 3, .block_size = 1, .caps = V },
	[SF_R16G16B16_SSCALED]			= { .size =  6, .channels = 3, .block_size = 1, .caps = V },
	[SF_R16G16B16_USCALED]			= { .size =  6, .channels = 3, .block_size = 1, .caps = V },
	[SF_BC6H_SF16]				= { .size =  6, .channels = 3, .block_size = 4, .caps = 0 },
	[SF_BC7_UNORM]				= { .size =  0, .channels = 3, .block_size = 4, .caps = 0 },
	[SF_BC7_UNORM_SRGB]			= { .size =  0, .channels = 3, .block_size = 4, .caps = 0 | SRGB },
	[SF_BC6H_UF16]				= { .size =  0, .channels = 3, .block_size = 4, .caps = 0 },
	[SF_PLANAR_420_8]			= { .size =  0, .channels = 3, .block_size = 1, .caps = 0 },
	[SF_R8G8B8_UNORM_SRGB]			= { .size =  3, .channels = 3, .block_size = 1, .caps = V | SRGB },
	[SF_ETC1_RGB8]				= { .size =  0, .channels = 3, .block_size = 1, .caps = 0 },
	[SF_ETC2_RGB8]				= { .size =  0, .channels = 3, .block_size = 1, .caps = 0 },
	[SF_EAC_R11]				= { .size =  0, .channels = 3, .block_size = 1, .caps = 0 },
	[SF_EAC_RG11]				= { .size =  0, .channels = 3, .block_size = 1, .caps = 0 },
	[SF_EAC_SIGNED_R11]			= { .size =  0, .channels = 3, .block_size = 1, .caps = 0 },
	[SF_EAC_SIGNED_RG11]			= { .size =  0, .channels = 3, .block_size = 1, .caps = 0 },
	[SF_ETC2_SRGB8]				= { .size =  0, .channels = 3, .block_size = 1, .caps = 0 | SRGB },
	[SF_R16G16B16_UINT]			= { .size =  6, .channels = 3, .block_size = 1, .caps = V },
	[SF_R16G16B16_SINT]			= { .size =  6, .channels = 3, .block_size = 1, .caps = V },
	[SF_R32_SFIXED]				= { .size =  4, .channels = 1, .block_size = 1, .caps = V },
	[SF_R10G10B10A2_SNORM]			= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_R10G10B10A2_USCALED]		= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_R10G10B10A2_SSCALED]		= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_R10G10B10A2_SINT]			= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_B10G10R10A2_SNORM]			= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_B10G10R10A2_USCALED]		= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_B10G10R10A2_SSCALED]		= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_B10G10R10A2_UINT]			= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_B10G10R10A2_SINT]			= { .size =  4, .channels = 4, .block_size = 1, .caps = V },
	[SF_R64G64B64A64_PASSTHRU]		= { .size = 32, .channels = 4, .block_size = 1, .caps = V },
	[SF_R64G64B64_PASSTHRU]			= { .size = 24, .channels = 4, .block_size = 1, .caps = V },
	[SF_ETC2_RGB8_PTA]			= { .size =  0, .channels = 3, .block_size = 1, .caps = 0 },
	[SF_ETC2_SRGB8_PTA]			= { .size =  0, .channels = 3, .block_size = 1, .caps = 0 | SRGB },
	[SF_ETC2_EAC_RGBA8]			= { .size =  0, .channels = 4, .block_size = 1, .caps = 0 },
	[SF_ETC2_EAC_SRGB8_A8]			= { .size =  0, .channels = 4, .block_size = 1, .caps = 0 | SRGB },
	[SF_R8G8B8_UINT]			= { .size =  3, .channels = 3, .block_size = 1, .caps = V },
	[SF_R8G8B8_SINT]			= { .size =  3, .channels = 3, .block_size = 1, .caps = V },
	[SF_RAW]				= { .size =  0, .channels = 4, .block_size = 1, .caps = 0 },
};

bool
valid_vertex_format(uint32_t format)
{
	ksim_assert(format <= SF_RAW);

	return formats[format].caps & V;
}

bool
srgb_format(uint32_t format)
{
	ksim_assert(format <= SF_RAW);

	return formats[format].caps & SRGB;
}

uint32_t
format_size(uint32_t format)
{
	ksim_assert(format <= SF_RAW);

	return formats[format].size;
}

uint32_t
format_channels(uint32_t format)
{
	ksim_assert(format <= SF_RAW);

	return formats[format].channels;
}

uint32_t
format_block_size(uint32_t format)
{
	ksim_assert(format <= SF_RAW);

	return formats[format].block_size;
}

static const struct format_info depth_formats[] = {
	[D32_FLOAT]				= { .size = 4 },
	[D24_UNORM_X8_UINT]			= { .size = 4 },
	[D16_UNORM]				= { .size = 2 },
};

uint32_t
depth_format_size(uint32_t format)
{
	ksim_assert(format <= D16_UNORM);

	return depth_formats[format].size;
}
