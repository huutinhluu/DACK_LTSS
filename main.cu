#include <stdio.h>
#include <stdint.h>
#include "seam_carving/seam_carving.cuh"

using namespace std;

int main(int argc, char **argv)
{

	int width, height, newWidth;
	uchar3 *rgbPic;
	dim3 blockSize(32, 32);

	// Check user's input
	check_input(argc, argv, width, height, rgbPic, newWidth, blockSize);

	// HOST
	uchar3 *out_host = (uchar3 *)malloc(width * height * sizeof(uchar3));
	resize_image_host(rgbPic, width, height, newWidth, out_host);

	// DEVICE
	uchar3 *out_device_1 = (uchar3 *)malloc(width * height * sizeof(uchar3));
	resize_image_device(rgbPic, width, height, newWidth, out_device_1, blockSize, 1);

	uchar3 *out_device_2 = (uchar3 *)malloc(width * height * sizeof(uchar3));
	resize_image_device(rgbPic, width, height, newWidth, out_device_2, blockSize, 2);

	// Compute error
	print_error((char *)"Error between device result and host result: ", out_host, out_device_1, width, height);

	// Write 2 results to files
	write_pnm(out_host, newWidth, height, width, concat_str(argv[2], "_host.pnm"));
	write_pnm(out_device_1, newWidth, height, width, concat_str(argv[2], "_device1.pnm"));
	write_pnm(out_device_1, newWidth, height, width, concat_str(argv[2], "_device2.pnm"));

	// Free memories
	free(rgbPic);
	free(out_host);
	free(out_device_1);
}
