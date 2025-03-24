// This file is an implimentation of an edge detection model using the canny algorithm
// This edge detection model is then improved using parallel programming concepts to decrease runtime

#include <iostream>
#include <fstream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBIW_WINDOWS_UTF8
#include "stb_image_write.h"
#include <omp.h>
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>

using namespace std;
#include <cmath>

bool MULTITHREAD_ACTIVE;

// Vector2 struct
struct Vector2
{
    double x, y; // x =  magnitude, y = direction
    Vector2(double x = 0, double y = 0) : x(x), y(y) {} // constructor
};

// Intensity gradient struct (IGI). Useful for passing around full image data.
struct IntensityGradientImage
{
    int height, width; // holds the height and width of the image
    Vector2 *intensityGradientArray; // pointer to a dynamic 1D array of Vector2 structs

    IntensityGradientImage(int h, int w, Vector2 *iga) // contructor with an initialzer list
        : height(h), width(w), intensityGradientArray(iga) {}
};

// *** Get Image Helper Functions *** //

// Convert a single pixel from original img to the corresponding grayscaled intensity gradient array spot
void handlePixelConversion(unsigned char* img_p, Vector2* iga_p)
{
    // Calculate average of RGB values and store it as intensity
    *iga_p = Vector2(((*img_p + *(img_p + 1) + *(img_p + 2)) / 3.0f), 0.0f);
}

// Takes the original image and converts it to grayscale
// Also stores the grayscale pixel values into the Intensity Gradient Array (IGA)
void grayscaleIGI(unsigned char* img, size_t img_size, int channels, Vector2* iga)
{
    int i = 0;

    if (!MULTITHREAD_ACTIVE) // Convert without multithreading
    {
        for (unsigned char* p = img; p != img + img_size; p += channels, i++)
        {
            handlePixelConversion(p, &iga[i]);
        }
    }
    else // Convert with multithreading
    {
        vector<thread> threads;
        int num_threads = 8; // Number of threads for batch processing
        int chunk_size = img_size / (num_threads * channels); // Size of each chunk

        // Split work into chunks
        for (int t = 0; t < num_threads; ++t)
        {
            threads.push_back(thread([=]() {
                for (int j = t * chunk_size; j < (t + 1) * chunk_size && j < img_size; ++j)
                {
                    unsigned char* p = img + j * channels;
                    handlePixelConversion(p, &iga[j]);
                }
            }));
        }

        // Wait for all threads to finish
        for (auto& t : threads) {
            t.join();
        }
    }
}



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

    // Convert image to grayscale and to intensity gradient array format
    grayscaleIGI(img, img_size, channels, gray_img);

    // create the actual Intensity Gradient Image variable using the creted array and the image dimensions
    IntensityGradientImage ret(height, width, gray_img);

    return ret;
}

// Apply a gausian filter to the image to smooth and remove noise
IntensityGradientImage gausianFilter(IntensityGradientImage &img) // step 1
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
IntensityGradientImage intensityGradient(IntensityGradientImage img) // step 2
{

    int width = img.width;
    int height = img.height;

    // Allocate space for the gradient array (replace malloc)
    Vector2 *gradArray = new Vector2[width * height];

    if (gradArray == NULL)
    {
        printf("Memory allocation failed for gradient array.\n");
        exit(1);
    }

    // Sobel kernels for x and y gradients
    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}};

    int Gy[3][3] = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}};

    // Apply Sobel filter to calculate gradients
    #pragma omp parallel for collapse(2)
    for(int y = 1; y < height - 1; y++) // Avoid border pixels
    {
        for(int x = 1; x < width - 1; x++)
        {
            double gx = 0.0, gy = 0.0;

            // Convolve with Sobel kernels
            for(int ky = -1; ky <= 1; ky++)
            {
                for(int kx = -1; kx <= 1; kx++)
                {
                    int pixelIndex = (y + ky) * width + (x + kx);
                    gx += img.intensityGradientArray[pixelIndex].x * Gx[ky + 1][kx + 1];
                    gy += img.intensityGradientArray[pixelIndex].x * Gy[ky + 1][kx + 1];
                }
            }

            // Compute magnitude and direction
            double magnitude = sqrt(gx * gx + gy * gy);
            double direction = atan2(gy, gx) * (180.0 / M_PI); // Convert to degrees

            // Store the gradient magnitude and direction
            int newIndex = y * width + x;
            gradArray[newIndex] = Vector2(magnitude, direction);
        }
    }


    // Calculate the intensity gradient of each pixel (round direction to the be one of the four edge directions ( | , - , / , \ ) )
    return IntensityGradientImage(height, width, gradArray);
}

IntensityGradientImage magnitudeThreshold(IntensityGradientImage intensityGradients, double weakThresh, double strongThresh)
{
    // get the height and width of the image
    int height = intensityGradients.height;
    int width = intensityGradients.width;
    
    // get the pointer to the gradient array
    Vector2 *gradientArray = intensityGradients.intensityGradientArray;

    // lambda function to process a single pixel
    auto processPixel = [&](int i, int j) {
        // compute the 1D index for the 2D array
        int idx = i * width + j;
        
        // retrieve the magnitude stored in the x component
        double magnitude = gradientArray[idx].x;
        
        // apply thresholding to classify edges
        if (magnitude >= strongThresh) {
            gradientArray[idx].x = 255.0; // Strong edge
        } else if (magnitude >= weakThresh) {
            gradientArray[idx].x = 128.0; // Weak edge
        } else {
            gradientArray[idx].x = 0.0; // Non-edge
        }
    };

    // use OpenMP for parallel execution if multithreading is enabled
    if (MULTITHREAD_ACTIVE) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                processPixel(i, j);
            }
        }
    } 
    // otherwise, execute sequentially
    else {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                processPixel(i, j);
            }
        }
    }

    // return the modified intensity gradient image
    return intensityGradients;
}


IntensityGradientImage nonMaximumSuppression(IntensityGradientImage img)
{
    int width = img.width;
    int height = img.height;
    Vector2* inArr = img.intensityGradientArray;

    // Output array to store suppressed results
    Vector2* outArr = new Vector2[width * height];

    #pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            double magnitude = inArr[idx].x;
            double angle = inArr[idx].y;

            // Normalize angle to [0, 180)
            if (angle < 0) angle += 180;

            double neighbor1 = 0.0, neighbor2 = 0.0;

            // Quantize gradient direction into 4 sectors
            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle < 180)) {
                // Horizontal (left–right)
                neighbor1 = inArr[y * width + (x - 1)].x;
                neighbor2 = inArr[y * width + (x + 1)].x;
            }
            else if (angle >= 22.5 && angle < 67.5) {
                // Diagonal (↙ / ↗)
                neighbor1 = inArr[(y - 1) * width + (x + 1)].x;
                neighbor2 = inArr[(y + 1) * width + (x - 1)].x;
            }
            else if (angle >= 67.5 && angle < 112.5) {
                // Vertical (top–bottom)
                neighbor1 = inArr[(y - 1) * width + x].x;
                neighbor2 = inArr[(y + 1) * width + x].x;
            }
            else if (angle >= 112.5 && angle < 157.5) {
                // Diagonal (↖ / ↘)
                neighbor1 = inArr[(y - 1) * width + (x - 1)].x;
                neighbor2 = inArr[(y + 1) * width + (x + 1)].x;
            }

            // If current pixel is local maximum, keep it; else suppress
            if (magnitude >= neighbor1 && magnitude >= neighbor2) {
                outArr[idx] = Vector2(magnitude, inArr[idx].y); // keep
            } else {
                outArr[idx] = Vector2(0.0, 0.0); // suppress
            }
        }
    }

    // Clean up and assign new array
    delete[] img.intensityGradientArray;
    img.intensityGradientArray = outArr;
    return img;
}

// Filter out noisey edges using threshold filter
// (Strong edges are accepted, weak are accepted if connected to strong)
IntensityGradientImage hysteresis(IntensityGradientImage intensityGradients, double strongThresh)
{
    int width = intensityGradients.width;
    int height = intensityGradients.height;
    Vector2* arr = intensityGradients.intensityGradientArray;

    // 8-connected neighbor offsets
    int dx[8] = {-1,  0, 1, -1, 1, -1, 0, 1};
    int dy[8] = {-1, -1, -1,  0, 0,  1, 1, 1};

    // Make a copy of the array to track updated states
    Vector2* newArr = new Vector2[width * height];
    memcpy(newArr, arr, sizeof(Vector2) * width * height);

    // Loop through all pixels
    #pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;

            // If pixel is weak (128)
            if (arr[idx].x == 128.0) {
                bool connectedToStrong = false;

                // Check 8 neighbors
                for (int k = 0; k < 8; k++) {
                    int nx = x + dx[k];
                    int ny = y + dy[k];
                    int nidx = ny * width + nx;

                    if (arr[nidx].x == 255.0) {
                        connectedToStrong = true;
                        break;
                    }
                }

                // Update in new array
                if (connectedToStrong) {
                    newArr[idx].x = 255.0; // Promote to strong
                } else {
                    newArr[idx].x = 0.0;   // Suppress
                    newArr[idx].y = 0.0;
                }
            }
        }
    }

    // Free the old array and replace it
    delete[] arr;
    intensityGradients.intensityGradientArray = newArr;

    return intensityGradients;
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

    MULTITHREAD_ACTIVE = true; // change this value if you want to activate/deactivate multithreading mode

    auto start_time = chrono::high_resolution_clock::now(); // Start runtime clock

    // Get the image file an prep for transformation
    IntensityGradientImage img = getImage("./images/NYScene.jpg");

   
    img = gausianFilter(img); // Smooth the gray image using the gausian filter
    img = intensityGradient(img); // Calculate the gradient values for ech pixel
    img = nonMaximumSuppression(img); // Apply non-maximum suppression to thin the edges
    img = magnitudeThreshold(img, 50, 100); // Apply the magnitude threshold to remove unecessary "edges"
    img = hysteresis(img, 100); // Apply the hysteresis filter to remove weak edges

    saveImage("output.png", img);

    delete[] img.intensityGradientArray; // free img allocation
    // End runtime clock
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    // Print results
    cout << "Duration = " << duration.count() << " ms" << endl;

    return 0;
}
