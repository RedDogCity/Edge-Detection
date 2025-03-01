// This file is an implimentation of an edge detection model using the canny algorithm
// This edge detection model is then improved using parallel programming concepts to decrease runtime

// *** NOTE *** some function types may need to change from skeleton to implimentation

#include <iostream>
#include <fstream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
using namespace std;
#include <cmath>

// Vector2 struct
struct Vector2
{
    double x, y;

    Vector2(double x = 0, double y = 0) : x(x), y(y) {}
};

// Intensity gradient struct
struct IntensityGradientImage
{
    int height, width;
    Vector2 *intensityGradientArray;

    IntensityGradientImage(int h, int w, Vector2 *iga) : height(h), width(w), intensityGradientArray(iga) {}
};

// Gets image from file path and returns gray image as 2d vector
IntensityGradientImage getImage(const char *imgPath)
{
    // load the image from the file path provided
    int width, height, channels;
    unsigned char *img = stbi_load(imgPath, &width, &height, &channels, 0);
    if (img == NULL || channels != 3) // exit if the image is not found or is incorrect number of channels
    {
        printf("Error in loading the image\n");
        exit(1);
    }

    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);

    // initialize image size variables
    size_t img_size = width * height * channels;
    size_t gray_img_size = width * height;

    // allocate space for the intensity gradient array
    Vector2 *gray_img = (Vector2 *)malloc(gray_img_size * sizeof(Vector2));

    if (gray_img == NULL)
    {
        printf("Unable to allocate memory for the gray image intensity gradient array.\n");
        exit(1);
    }

    // Convert each colored pixel in the original image to an average of the rgb values
    // Store these pixel values in the intensities and initialize the gradient values to 0.0
    int i = 0;
    for (unsigned char *p = img; p != img + img_size; p += channels, i += 1)
    {
        gray_img[i] = Vector2(((*p + *(p + 1) + *(p + 2)) / 3.0), 0.0);
    }

    // create the actual Intensity Gradient Image variable using the creted array and the image dimensions
    IntensityGradientImage ret(height, width, gray_img);

    return ret;
}

// Apply a gausian filter to the image to smooth and remove noise
IntensityGradientImage gausianFilter(IntensityGradientImage &img)
{

    int width = img.width;
    int height = img.height;

    // Define a 5x5 Gaussian Kernel
    double GKernel[5][5];
    double sigma = 1.0;
    double sum = 0.0;

    // Precompute the Gaussian Kernel
    for (int x = -2; x <= 2; x++)
    {
        for (int y = -2; y <= 2; y++)
        {
            double r = sqrt(x * x + y * y);
            GKernel[x + 2][y + 2] = exp(-(r * r) / (2.0 * sigma * sigma)) / (2.0 * M_PI * sigma * sigma);
            sum += GKernel[x + 2][y + 2];
        }
    }

    // Normalize the kernel
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            GKernel[i][j] /= sum;

    // Allocate space for the output image
    Vector2 *blurred_img = (Vector2 *)malloc(width * height * sizeof(Vector2));
    if (blurred_img == NULL)
    {
        printf("Memory allocation failed for Gaussian Filter output.\n");
        exit(1);
    }

    // Apply Gaussian Blur using Convolution
    for (int y = 2; y < height - 2; y++)
    {
        for (int x = 2; x < width - 2; x++)
        {
            double sum = 0.0;

            // Convolve with 5x5 Gaussian Kernel
            for (int ky = -2; ky <= 2; ky++)
            {
                for (int kx = -2; kx <= 2; kx++)
                {
                    int pixelIndex = (y + ky) * width + (x + kx);
                    sum += img.intensityGradientArray[pixelIndex].x * GKernel[ky + 2][kx + 2];
                }
            }

            // Store the blurred value
            int newIndex = y * width + x;
            blurred_img[newIndex] = Vector2(sum, 0.0); // Preserve the gradient as 0.0 for now
        }
    }

    // Return the new blurred image
    return IntensityGradientImage(height, width, blurred_img);
}

// Calculate the gradients for each pixel
IntensityGradientImage intensityGradient(IntensityGradientImage intensityGradients)
{

    // Calculate the intensity gradient of each pixel (round direction to the be one of the four edge directions ( | , - , / , \ ) )

    return intensityGradients;
}

// Apply gradient magnitude thresholding to find actual image edges from the gradients
IntensityGradientImage magnitudeThreshold(IntensityGradientImage intensityGradients, double weakThresh)
{

    // Remove all edges below threshold (set to 0 maybe)

    return intensityGradients;
}

// Filter out noisey edges using threshold filter
// (Strong edges are accepted, weak are accepted if connected to strong)
IntensityGradientImage hysteresis(IntensityGradientImage intensityGradients, double strongThresh)
{

    // Remove all edges below threshold that arent connected to strong edge (set to 0 maybe)

    return intensityGradients;
}

// Displays (sends to an output folder) image given by intensity gradient array
void displayImage(IntensityGradientImage intensityGradients)
{
}

// From Betelgeuse
void saveImage(const char *outputPath, IntensityGradientImage img)
{
    // getting image dimensions
    int width = img.width;
    int height = img.height;

    // allocates memory for (width*height)
    // use unsigned char * 8-bit unsigned char (0-255).
    unsigned char *output_img = (unsigned char *)malloc(width * height);

    // if allocation fails exits the program
    if (output_img == NULL)
    {
        perror("Memory allocation failed");
        exit(1);
    }

    // Converts the image Data
    for (int i = 0; i < width * height; i++)
    {
        output_img[i] = (unsigned char)(img.intensityGradientArray[i].x);
    }

    // Save image
    stbi_write_png(outputPath, width, height, 1, output_img, width);

    // if Image saving fails
    if (!stbi_write_png(outputPath, width, height, 1, output_img, width))
    {
        printf("Error saving image!\n");
    }
    // free allocated memory
    free(output_img);

}

// Main function to run everything
int main()
{

    // Get the image file an prep for transformation
    IntensityGradientImage img = getImage("./images/Rural.jpg");

    // Smooth the gray image using the gausian filter
    img = gausianFilter(img);

    // // Calculate the gradient values for ech pixel
    // img = intensityGradient(img);

    // // Apply the magnitude threshold to remove unecessary "edges"
    // img = magnitudeThreshold(img, 0); // threshold value will need adjusting

    // // Apply hysteresis to keep only the strong edges (and the important weak edges)
    // img = magnitudeThreshold(img, 100); // threshold value will need adjusting

    saveImage("output.png", img);
    // free img allocation
    stbi_image_free(img.intensityGradientArray); // Free memory

    return 0;
}
