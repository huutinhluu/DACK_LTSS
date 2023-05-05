#ifndef SEAM_CARVING_CUH
#define SEAM_CARVING_CUH

#include "../libraries/pnm.cuh"

/**
 * @brief This function is responsible for checking the input parameters provided by the user and preparing the data for image resizing using CUDA.
 *
 * @param argc the number of arguments provided by the user on the command line
 * @param argv an array of strings containing the command line arguments
 * @param width a reference to an integer that will store the width of the input image
 * @param height a reference to an integer that will store the height of the input image
 * @param rgbImage a pointer to a uchar3 array that will store the RGB values of the input image
 * @param newWidth a reference to an integer that will store the desired width of the output image
 * @param blockSize a reference to a dim3 struct that will store the block size used for CUDA kernel execution
 */
void check_input(int argc, char **argv, int &width, int &height, uchar3 *&rgbImage, int &newWidth, dim3 &blockSize);

/**
 * @brief The purpose of this kernel is to calculate the energy map of an image.
 * The energy map is a matrix that represents the amount of change in color intensity in the image.
 * It is typically used in image processing algorithms such as seam carving,
 * which is used to resize images without distorting important features.
 *
 * @param inPixels  a pointer to the input image pixels
 * @param width the width of the input image
 * @param height the height of the input image
 * @param energyMap a pointer to the output energy map
 * @return __global__ This function will be executed on the GPU
 */
__global__ void calculate_energy_map_device(uint8_t *inPixels, int width, int height, int *energyMap);

/**
 * @brief This function finds the seam path with the minimum energy in the given minimum energy map.
 *  The seam path is the sequence of pixels that will be removed from the image to reduce its width by one pixel.
 *
 * @param minimumEnergy an integer array containing the minimum energy values for each pixel in the image.
 * @param seamPath an integer array that will store the seam path for the image.
 * @param width the width of the image in pixels.
 * @param height the height of the image in pixels.
 */
void find_seam(int *minimumEnergy, int *seamPath, int width, int height);

/**
 * @brief The purpose of this kernel is to remove the least important seam from an image by shifting the remaining pixels to fill the gap left by the removed seam.
 *
 * @param seamPath an array that contains the column indices of the least important pixels in each row
 * @param outPixels a pointer to the output image pixels
 * @param grayPixels a pointer to the grayscale version of the input image pixels
 * @param energyMap a pointer to the energy map of the input image
 * @param width the width of the input image
 * @return __global__ This function will be executed on the GPU
 */
__global__ void remove_seam_device(int *seamPath, uchar3 *outPixels, uint8_t *grayPixels, int *energyMap, int width);

/**
 * @brief This function is a CUDA kernel that calculates the minimal energy path from a given starting row to the last row in an image
 *
 * @param energy A pointer to the energy map array
 * @param minimalEnergy  A pointer to an array that will store the minimal energy value of a path from a given pixel to the bottom row of the image.
 * @param width The width of the input image.
 * @param height The height of the input image.
 * @param fromRow The row of the image at which to start computing minimal energy values.
 * @return __global__ This function will be executed on the GPU
 */
__global__ void calculate_minimal_energy_path_device_from_top(int *energy, int *minimalEnergy, int width, int height, int fromRow);

/**
 * @brief This function is a CUDA kernel that calculates the minimal energy path from button to top
 *
 * @param energy A pointer to the energy map array
 * @param minimalEnergy  A pointer to an array that will store the minimal energy value of a path from a given pixel to the bottom row of the image.
 * @param width The width of the input image.
 * @param height The height of the input image.
 * @param fromRow The row of the image at which to start computing minimal energy values.
 * @return __global__ This function will be executed on the GPU
 */
__global__ void calculate_minimal_energy_path_from_botton_device(int *energy, int *minimalEnergy, int width, int height, int fromRow);

/**
 * @brief This is a function that performs image resizing on the GPU using seam carving technique.
 *
 * @param inPixels pointer to the input image data, stored as an array of uchar3 structs.
 * @param width width of the input image.
 * @param height height of the input image.
 * @param newWidth the target width of the output image.
 * @param outPixels pointer to the output image data, stored as an array of uchar3 structs.
 * @param blockSize a dim3 struct specifying the dimensions of the GPU thread block.
 */
void resize_image_device(uchar3 *inPixels, int width, int height, int newWidth, uchar3 *outPixels, dim3 blockSize, int calculateMethod);

/**
 * @brief This function calculates the energy of a pixel in an image by convolving the image with the Sobel operator.
 *  The Sobel operator is a filter that detects edges in an image.
 *
 * @param grayPixels A pointer to the array of gray-scale pixel values for the image.
 * @param row The row index of the pixel for which we want to calculate the energy.
 * @param col The column index of the pixel for which we want to calculate the energy.
 * @param width The width of the image in pixels.
 * @param height The height of the image in pixels.
 * @return int The pixel energy value
 */
int get_pixel_energy(uint8_t *grayPixels, int row, int col, int width, int height);

/**
 * @brief This function computes the minimal energy path from each pixel in the top row to the bottom row of an image.
 *
 * @param energy a pointer to an array of integers representing the energy values of each pixel in the image.
 * @param minimalEnergy a pointer to an array of integers that will store the minimal energy path values from each pixel in the top row to the bottom row.
 * @param width an integer representing the width of the image.
 * @param height an integer representing the height of the image.
 */
void calculate_minimal_energy_path_host(int *energy, int *minimalEnergy, int width, int height);

/**
 * @brief This is a function that performs image resizing on the host using seam carving technique.
 *
 * @param inPixels pointer to the input image data, stored as an array of uchar3 structs.
 * @param width width of the input image.
 * @param height height of the input image.
 * @param newWidth the target width of the output image.
 * @param outPixels pointer to the output image data, stored as an array of uchar3 structs.
 */
void resize_image_host(uchar3 *inPixels, int width, int height, int newWidth, uchar3 *outPixels);

#endif // SEAM_CARVING_CUH