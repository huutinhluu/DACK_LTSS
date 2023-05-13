#include "seam_carving.cuh"

int xSobel[3][3] =
    {{1, 0, -1},
     {2, 0, -2},
     {1, 0, -1}};
int ySobel[3][3] =
    {{1, 2, 1},
     {0, 0, 0},
     {-1, -2, -1}};

__constant__ int d_xSobel[9] =
    {1, 0, -1,
     2, 0, -2,
     1, 0, -1};
__constant__ int d_ySobel[9] =
    {1, 2, 1,
     0, 0, 0,
     -1, -2, -1};

int WIDTH;
__device__ int d_WIDTH;

const int filterWidth = 3;

void check_input(int argc, char **argv, int &width, int &height, uchar3 *&rgbImage, int &newWidth, dim3 &blockSize)
{
    if (argc != 4 && argc != 6)
    {
        printf("The number of arguments is invalid\n");
        exit(EXIT_FAILURE);
    }

    // Read file
    read_pnm(argv[1], width, height, rgbImage);
    printf("Image size (width x height): %i x %i\n\n", width, height);

    WIDTH = width;
    CHECK(cudaMemcpyToSymbol(d_WIDTH, &width, sizeof(int)));

    // Check user's desired width
    newWidth = atoi(argv[3]);

    if (newWidth <= 0 || newWidth >= width)
    {
        printf("Your desired width must between 0 & current picture's width!\n");
        exit(EXIT_FAILURE);
    }

    // Block size
    if (argc == 6)
    {
        blockSize.x = atoi(argv[4]);
        blockSize.y = atoi(argv[5]);
    }

    // Check GPU is working or not
    print_device_info();
}

__global__ void calculate_energy_map_device(uint8_t *inPixels, int width, int height, int *energyMap)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int s_width = blockDim.x + filterWidth - 1;
    int s_height = blockDim.y + filterWidth - 1;

    // Each block loads data from GMEM to SMEM
    extern __shared__ uint8_t s_inPixels[];

    // Load data from global memory (input image) to shared memory (s_inPixels) for this block
    // Each thread loads one pixel from global memory and writes it to shared memory
    // The shared memory buffer includes a border of filterWidth-1 pixels to allow threads to access neighboring pixels
    // Note that we load more pixels into shared memory than we actually need to compute the energy map,
    // in order to allow threads to access neighboring pixels without having to access global memory repeatedly
    int readRow = row - filterWidth / 2, readCol, tmpRow, tmpCol;
    int firstReadCol = col - filterWidth / 2;
    int virtualRow, virtualCol;

    for (virtualRow = threadIdx.y; virtualRow < s_height; readRow += blockDim.y, virtualRow += blockDim.y)
    {
        tmpRow = readRow;

        readRow = min(max(readRow, 0), height - 1); // 0 <= readCol <= height-1

        readCol = firstReadCol;
        virtualCol = threadIdx.x;

        for (; virtualCol < s_width; readCol += blockDim.x, virtualCol += blockDim.x)
        {
            tmpCol = readCol;

            readCol = min(max(readCol, 0), width - 1); // 0 <= readCol <= width-1

            s_inPixels[virtualRow * s_width + virtualCol] = inPixels[readRow * d_WIDTH + readCol];
            readCol = tmpCol;
        }
        readRow = tmpRow;
    }

    // Ensure all threads in this block have finished loading data into shared memory
    __syncthreads();

    // Compute the energy map for this thread's pixel based on the shared memory data
    // Each thread computes the energy for one pixel using the Sobel filter, which requires neighboring pixels
    // in shared memory to be accessible without having to access global memory repeatedly
    int x_kernel = 0, y_kernel = 0;
    for (int i = 0; i < filterWidth; ++i)
    {
        for (int j = 0; j < filterWidth; ++j)
        {
            uint8_t closest = s_inPixels[(threadIdx.y + i) * s_width + threadIdx.x + j];
            int filterIdx = i * filterWidth + j;
            x_kernel += closest * d_xSobel[filterIdx];
            y_kernel += closest * d_ySobel[filterIdx];
        }
    }

    // Each thread writes result from SMEM to GMEM
    if (col < width && row < height)
        energyMap[row * d_WIDTH + col] = abs(x_kernel) + abs(y_kernel);
}

__global__ void remove_seam_device(int *seamPath, uchar3 *out_pixels, uint8_t *grayPixels, int *energyMap, int width)
{
    int row = blockIdx.x;
    int baseIdx = row * d_WIDTH;
    for (int i = seamPath[row]; i < width - 1; ++i)
    {
        out_pixels[baseIdx + i] = out_pixels[baseIdx + i + 1];
        grayPixels[baseIdx + i] = grayPixels[baseIdx + i + 1];
        energyMap[baseIdx + i] = energyMap[baseIdx + i + 1];
    }
}

void find_seam(int *minimumEnergy, int *seamPath, int width, int height)
{
    int minCol = 0, r = height - 1;

    for (int c = 1; c < width; ++c)
        if (minimumEnergy[r * WIDTH + c] < minimumEnergy[r * WIDTH + minCol])
            minCol = c;

    for (; r >= 0; --r)
    {
        seamPath[r] = minCol;
        if (r > 0)
        {
            int aboveIdx = (r - 1) * WIDTH + minCol;
            int min = minimumEnergy[aboveIdx], minColCpy = minCol;

            if (minColCpy > 0 && minimumEnergy[aboveIdx - 1] < min)
            {
                min = minimumEnergy[aboveIdx - 1];
                minCol = minColCpy - 1;
            }
            if (minColCpy < width - 1 && minimumEnergy[aboveIdx + 1] < min)
            {
                minCol = minColCpy + 1;
            }
        }
    }
}

__global__ void calculate_minimal_energy_path_device_from_top(int *energy, int *minimalEnergy, int width, int height, int fromRow)
{
    size_t halfBlock = blockDim.x / 2;

    // For example, let's assume that blockDim.x is equal to 256,
    // halfBlock is equal to 128,
    // blockIdx.x is equal to 1,
    // and threadIdx.x is equal to 50.
    // Then, col is computed as follows:
    // col = 1 * 128 - 128 + 50 = 50
    int col = blockIdx.x * halfBlock - halfBlock + threadIdx.x;

    if (fromRow == 0 && col >= 0 && col < width)
    {
        minimalEnergy[col] = energy[col];
    }
    __syncthreads();

    for (int stride = fromRow != 0 ? 0 : 1; stride < halfBlock && fromRow + stride < height; ++stride)
    {
        if (threadIdx.x < blockDim.x - (stride << 1))
        {
            int curRow = fromRow + stride;
            int curCol = col + stride;

            if (curCol >= 0 && curCol < width)
            {
                int idx = curRow * d_WIDTH + curCol;
                int aboveIdx = (curRow - 1) * d_WIDTH + curCol;

                int min = minimalEnergy[aboveIdx];
                if (curCol > 0 && minimalEnergy[aboveIdx - 1] < min)
                    min = minimalEnergy[aboveIdx - 1];

                if (curCol < width - 1 && minimalEnergy[aboveIdx + 1] < min)
                    min = minimalEnergy[aboveIdx + 1];

                minimalEnergy[idx] = min + energy[idx];
            }
        }
        __syncthreads();
    }
}

__global__ void calculate_minimal_energy_path_from_botton_device(int *energy, int *minimalEnergy, int width, int height, int fromRow)
{
    size_t halfBlock = blockDim.x / 2;

    int row = height - 1 - blockIdx.x;
    int col = threadIdx.x * 2;

    // Traverse the rows from height - 1 to 0
    if (row >= 0)
    {
        // Initialize minimalEnergy for the last row
        if (row == height - 1)
        {
            minimalEnergy[row * width + col] = energy[row * width + col];
            minimalEnergy[row * width + col + 1] = energy[row * width + col + 1];
        }
        else
        {
            int aboveRow = row + 1;
            int idx = row * width + col;
            int aboveIdx = aboveRow * width + col;

            int min = minimalEnergy[aboveIdx];
            if (col > 0 && minimalEnergy[aboveIdx - 1] < min)
            {
                min = minimalEnergy[aboveIdx - 1];
            }
            if (col < width - 1 && minimalEnergy[aboveIdx + 1] < min)
            {
                min = minimalEnergy[aboveIdx + 1];
            }

            minimalEnergy[idx] = min + energy[idx];

            idx = row * width + col + 1;
            aboveIdx = aboveRow * width + col + 1;

            min = minimalEnergy[aboveIdx];
            if (col > 0 && minimalEnergy[aboveIdx - 1] < min)
            {
                min = minimalEnergy[aboveIdx - 1];
            }
            if (col < width - 1 && minimalEnergy[aboveIdx + 1] < min)
            {
                min = minimalEnergy[aboveIdx + 1];
            }

            minimalEnergy[idx] = min + energy[idx];
        }
    }
}

void resize_image_device(uchar3 *inPixels, int width, int height, int newWidth, uchar3 *outPixels, dim3 blockSize, int calculateMethod)
{
    GpuTimer timer;
    timer.Start();

    // allocate kernel memory
    uchar3 *d_inPixels;
    CHECK(cudaMalloc(&d_inPixels, width * height * sizeof(uchar3)));
    uint8_t *d_grayPixels;
    CHECK(cudaMalloc(&d_grayPixels, width * height * sizeof(uint8_t)));
    int *d_energy;
    CHECK(cudaMalloc(&d_energy, width * height * sizeof(int)));
    int *d_leastSignificantPixel;
    CHECK(cudaMalloc(&d_leastSignificantPixel, height * sizeof(int)));
    int *d_minimalEnergy;
    CHECK(cudaMalloc(&d_minimalEnergy, width * height * sizeof(int)));

    // allocate host memory
    int *energy = (int *)malloc(width * height * sizeof(int));
    int *leastSignificantPixel = (int *)malloc(height * sizeof(int));
    int *minimalEnergy = (int *)malloc(width * height * sizeof(int));

    // dynamically sized smem used to compute energy
    size_t smemSize = ((blockSize.x + 3 - 1) * (blockSize.y + 3 - 1)) * sizeof(uint8_t);

    // blockSizeDp is a variable that determines the number of threads per block used in the calculate_minimal_energy_path_device kernel,
    // which calculates the minimal energy seam path of each strip in the image.
    int blockSizeDp = 256;
    int gridSizeDp = (((width - 1) / blockSizeDp + 1) << 1) + 1;

    // For an image size of 512x512, the stripHeight value would be calculated as follows:
    // The blockSizeDp is set to 256.
    // The stripHeight is computed as (blockSizeDp >> 1) + 1, which is equal to 129.
    int stripHeight = (blockSizeDp >> 1) + 1;

    // copy input to device
    CHECK(cudaMemcpy(d_inPixels, inPixels, width * height * sizeof(uchar3), cudaMemcpyHostToDevice));

    // turn input image to grayscale
    dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
    convert_rgb_to_gray_device<<<gridSize, blockSize>>>(d_inPixels, width, height, d_grayPixels);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    while (width > newWidth)
    {
        // update energy
        calculate_energy_map_device<<<gridSize, blockSize, smemSize>>>(d_grayPixels, width, height, d_energy);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        // compute min seam table
        // stripHeight >> 1 (which is equal to stripHeight/2)
        for (int i = 0; i < height; i += (stripHeight >> 1))
        {
            if (calculateMethod == 1)
            {
                calculate_minimal_energy_path_device_from_top<<<gridSizeDp, blockSizeDp>>>(d_energy, d_minimalEnergy, width, height, i);
            }
            else
            {
                calculate_minimal_energy_path_from_botton_device<<<gridSizeDp, blockSizeDp>>>(d_energy, d_minimalEnergy, width, height, i);
            }
            cudaDeviceSynchronize();
            CHECK(cudaGetLastError());
        }

        // find least significant pixel index of each row and store in d_leastSignificantPixel (SEQUENTIAL, in kernel or host)
        CHECK(cudaMemcpy(minimalEnergy, d_minimalEnergy, WIDTH * height * sizeof(int), cudaMemcpyDeviceToHost));
        find_seam(minimalEnergy, leastSignificantPixel, width, height);

        // carve
        CHECK(cudaMemcpy(d_leastSignificantPixel, leastSignificantPixel, height * sizeof(int), cudaMemcpyHostToDevice));
        remove_seam_device<<<height, 1>>>(d_leastSignificantPixel, d_inPixels, d_grayPixels, d_energy, width);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        --width;
    }

    CHECK(cudaMemcpy(outPixels, d_inPixels, WIDTH * height * sizeof(uchar3), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_inPixels));
    CHECK(cudaFree(d_grayPixels));
    CHECK(cudaFree(d_energy));
    CHECK(cudaFree(d_leastSignificantPixel));
    CHECK(cudaFree(d_minimalEnergy));

    free(minimalEnergy);
    free(leastSignificantPixel);
    free(energy);

    timer.Stop();
    if (calculateMethod==2){
        timer.printTime((char *)"device (optimized version)");
    } else {
        timer.printTime((char *)"device");
    }
}

int get_pixel_energy(uint8_t *grayPixels, int row, int col, int width, int height)
{
    int x_kernel = 0;
    int y_kernel = 0;

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            int r = min(max(0, row - 1 + i), height - 1); // 0 <= row - 1 + i < height
            int c = min(max(0, col - 1 + j), width - 1);  // 0 <= col - 1 + j < width

            uint8_t pixelVal = grayPixels[r * WIDTH + c]; //

            x_kernel += pixelVal * xSobel[i][j]; // Convolution with x-Sobel
            y_kernel += pixelVal * ySobel[i][j]; // Convolution with y-Sobel
        }
    }
    return abs(x_kernel) + abs(y_kernel); // Add matrix
}

void calculate_minimal_energy_path_host(int *energy, int *minimalEnergy, int width, int height)
{
    for (int c = 0; c < width; ++c)
    {
        minimalEnergy[c] = energy[c];
    }
    for (int r = 1; r < height; ++r)
    {
        for (int c = 0; c < width; ++c)
        {
            int idx = r * WIDTH + c;
            int aboveIdx = (r - 1) * WIDTH + c;

            int min = minimalEnergy[aboveIdx];
            if (c > 0 && minimalEnergy[aboveIdx - 1] < min)
            {
                min = minimalEnergy[aboveIdx - 1];
            }
            if (c < width - 1 && minimalEnergy[aboveIdx + 1] < min)
            {
                min = minimalEnergy[aboveIdx + 1];
            }

            minimalEnergy[idx] = min + energy[idx];
        }
    }
}

void resize_image_host(uchar3 *inPixels, int width, int height, int newWidth, uchar3 *outPixels)
{
    GpuTimer timer;
    timer.Start();

    memcpy(outPixels, inPixels, width * height * sizeof(uchar3));

    // Allocating memory
    int *energy = (int *)malloc(width * height * sizeof(int));
    int *minimalEnergy = (int *)malloc(width * height * sizeof(int));

    // Get grayscale
    uint8_t *grayPixels = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    convert_rgb_to_gray_host(inPixels, width, height, grayPixels);

    // Calculate all pixels energy
    for (int r = 0; r < height; ++r)
    {
        for (int c = 0; c < width; ++c)
        {
            energy[r * WIDTH + c] = get_pixel_energy(grayPixels, r, c, width, height);
        }
    }

    while (width > newWidth)
    {
        // Calculate energy to the end. (go from bottom to top)
        calculate_minimal_energy_path_host(energy, minimalEnergy, width, height);

        // find min index of last row
        int minCol = 0, r = height - 1, prevMinCol;
        for (int c = 1; c < width; ++c)
        {
            if (minimalEnergy[r * WIDTH + c] < minimalEnergy[r * WIDTH + minCol])
                minCol = c;
        }

        // Find and remove seam from last to first row
        for (; r >= 0; --r)
        {
            // remove seam pixel on row r
            for (int i = minCol; i < width - 1; ++i)
            {
                outPixels[r * WIDTH + i] = outPixels[r * WIDTH + i + 1];
                grayPixels[r * WIDTH + i] = grayPixels[r * WIDTH + i + 1];
                energy[r * WIDTH + i] = energy[r * WIDTH + i + 1];
            }

            // Update energy
            if (r < height - 1)
            {
                int affectedCol = max(0, prevMinCol - 2);

                while (affectedCol <= prevMinCol + 2 && affectedCol < width - 1)
                {
                    energy[(r + 1) * WIDTH + affectedCol] = get_pixel_energy(grayPixels, r + 1, affectedCol, width - 1, height);
                    affectedCol += 1;
                }
            }

            // find to the top
            if (r > 0)
            {
                prevMinCol = minCol;

                int aboveIdx = (r - 1) * WIDTH + minCol;
                int min = minimalEnergy[aboveIdx], minColCpy = minCol;
                if (minColCpy > 0 && minimalEnergy[aboveIdx - 1] < min)
                {
                    min = minimalEnergy[aboveIdx - 1];
                    minCol = minColCpy - 1;
                }
                if (minColCpy < width - 1 && minimalEnergy[aboveIdx + 1] < min)
                {
                    minCol = minColCpy + 1;
                }
            }
        }

        int affectedCol;
        for (affectedCol = max(0, minCol - 2); affectedCol <= minCol + 2 && affectedCol < width - 1; ++affectedCol)
        {
            energy[affectedCol] = get_pixel_energy(grayPixels, 0, affectedCol, width - 1, height);
        }

        --width;
    }

    free(grayPixels);
    free(minimalEnergy);
    free(energy);

    timer.Stop();
    timer.printTime((char *)"host");
}
