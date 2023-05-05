#ifndef PNM_CUH
#define PNM_CUH
#include "helper.cuh"

void read_pnm(char *file_name, int &width, int &height, uchar3 *&pixels);
void write_pnm(uchar3 *pixels, int width, int height, int original_width, char *file_name);
#endif // PNM_CUH