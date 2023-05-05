#include "helper.cuh"

void convert_rgb_to_gray_host(uchar3 *rgb_image, int width, int height, uint8_t *gray_image)
{
    for (int r = 0; r < height; ++r)
        for (int c = 0; c < width; ++c)
        {
            int i = r * width + c;
            gray_image[i] = 0.299f * rgb_image[i].x + 0.587f * rgb_image[i].y + 0.114f * rgb_image[i].z;
        }
}

__global__ void convert_rgb_to_gray_device(uchar3 *rgb_image, int width, int height, uint8_t *gray_image)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < height && c < width)
    {
        int i = r * width + c;
        gray_image[i] = 0.299f * rgb_image[i].x + 0.587f * rgb_image[i].y + 0.114f * rgb_image[i].z;
    }
}

void print_device_info()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("_____________GPU info_____________\n");
    printf("|Name:                   %s|\n", devProv.name);
    printf("|Compute capability:          %d.%d|\n", devProv.major, devProv.minor);
    printf("|Num SMs:                      %d|\n", devProv.multiProcessorCount);
    printf("|Max num threads per SM:     %d|\n", devProv.maxThreadsPerMultiProcessor);
    printf("|Max num warps per SM:         %d|\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("|GMEM:           %zu byte|\n", devProv.totalGlobalMem);
    printf("|SMEM per SM:          %zu byte|\n", devProv.sharedMemPerMultiprocessor);
    printf("|SMEM per block:       %zu byte|\n", devProv.sharedMemPerBlock);
    printf("|________________________________|\n");
}

float computeError(uchar3 *a1, uchar3 *a2, int n)
{
    float err = 0;
    for (int i = 0; i < n; i++)
    {
        err += abs((int)a1[i].x - (int)a2[i].x);
        err += abs((int)a1[i].y - (int)a2[i].y);
        err += abs((int)a1[i].z - (int)a2[i].z);
    }
    err /= (n * 3);
    return err;
}

void print_error(char *msg, uchar3 *in1, uchar3 *in2, int width, int height)
{
    float err = computeError(in1, in2, width * height);
    printf("%s: %f\n", msg, err);
}

char *concat_str(const char *s1, const char *s2)
{
    char *result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}