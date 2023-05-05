#include "pnm.cuh"

void read_pnm(char *file_name, int &width, int &height, uchar3 *&pixels)
{
    FILE *f = fopen(file_name, "r");
    if (f == NULL)
    {
        printf("Cannot read %s\n", file_name);
        exit(EXIT_FAILURE);
    }

    char type[3];
    fscanf(f, "%s", type);

    if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
    {
        fclose(f);
        printf("Cannot read %s\n", file_name);
        exit(EXIT_FAILURE);
    }

    fscanf(f, "%i", &width);
    fscanf(f, "%i", &height);

    int max_val;
    fscanf(f, "%i", &max_val);
    if (max_val > 255) // In this exercise, we assume 1 byte per value
    {
        fclose(f);
        printf("Cannot read %s\n", file_name);
        exit(EXIT_FAILURE);
    }

    pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    for (int i = 0; i < width * height; i++)
        fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

    fclose(f);
}

void write_pnm(uchar3 *pixels, int width, int height, int original_width, char *file_name)
{
    FILE *f = fopen(file_name, "w");
    if (f == NULL)
    {
        printf("Cannot write %s\n", file_name);
        exit(EXIT_FAILURE);
    }

    fprintf(f, "P3\n%i\n%i\n255\n", width, height);

    for (int r = 0; r < height; ++r)
    {
        for (int c = 0; c < width; ++c)
        {
            int i = r * original_width + c;
            fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
        }
    }

    fclose(f);
}