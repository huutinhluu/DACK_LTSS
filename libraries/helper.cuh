#ifndef HELPER_CUH
#define HELPER_CUH
#include <stdio.h>
#include <stdint.h>

#define CHECK(call)                                                \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    }

/**
 * Time counter
 */
struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }

    void printTime(char *s)
    {
        printf("Processing time of %s: %f ms\n\n", s, Elapsed());
    }
};

__global__ void convert_rgb_to_gray_device(uchar3 *rgb_image, int width, int height, uint8_t *gray_image);
void convert_rgb_to_gray_host(uchar3 *rgb_image, int width, int height, uint8_t *gray_image);
void print_device_info();
void print_error(char *msg, uchar3 *in1, uchar3 *in2, int width, int height);
char *concat_str(const char *s1, const char *s2);
#endif // HELPER_CUH