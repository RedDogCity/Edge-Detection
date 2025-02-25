// This file is an implimentation of an edge detection model using the canny algorithm
// This edge detection model is then improved using parallel programming concepts to decrease runtime

// *** NOTE *** some function types may need to change from skeleton to implimentation

#include <iostream>
#include <fstream>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
using namespace std;

// Vector2 struct
struct Vector2 {
    double x, y;
    
    Vector2( double x = 0, double y = 0 ) : x(x), y(y) {}
};

// Intensity gradient struct
struct IntensityGradientImage {
    int height, width;
    Vector2 * intensityGradientArray; 
    
    IntensityGradientImage( int h, int w, Vector2 * iga ) : height(h), width(w), intensityGradientArray(iga) {}
};

// Gets image from file path and returns gray image as 2d vector
IntensityGradientImage getImage(const char* imgPath) 
{
    // load the image from the file path provided
    int width, height, channels;
    unsigned char *img = stbi_load(imgPath, &width, &height, &channels, 0);
    if(img == NULL || channels != 3) // exit if the image is not found or is incorrect number of channels
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
    if(gray_img == NULL) {
        printf("Unable to allocate memory for the gray image intensity gradient array.\n");
        exit(1);
    }

    // Convert each colored pixel in the original image to an average of the rgb values
    // Store these pixel values in the intensities and initialize the gradient values to 0.0
    int i = 0;
    for(unsigned char *p = img; p != img + img_size; p += channels, i += 1) 
    {
        gray_img[i] = Vector2(((*p + *(p + 1) + *(p + 2))/3.0), 0.0);
    }

    // create the actual Intensity Gradient Image variable using the creted array and the image dimensions
    IntensityGradientImage ret(height, width, gray_img);

    return ret;

}



// Apply a gausian filter to the image to smooth and remove noise
IntensityGradientImage gausianFilter(IntensityGradientImage img)
{
    

    return img;
}

// Create a 2d array (vectors) of vector2's (structure) that represent the intesity gradients in the image (direction of edges)
IntensityGradientImage intensityGradient(IntensityGradientImage intensityGradients)
{

    // Calculate the intensity gradient of each pixel (round direction to the be one of the four edge directions ( | , - , / , \ ) )
    

    return intensityGradients;
}



// Apply gradient magnitude thresholding to find actual image edges from the gradients
IntensityGradientImage magnitudeThreshold( IntensityGradientImage intensityGradients , double weakThresh )
{
    
    // Remove all edges below threshold (set to 0 maybe)

    return intensityGradients;
}



// Filter out noisey edges using threshold filter 
// (Strong edges are accepted, weak are accepted if connected to strong)
IntensityGradientImage hysteresis( IntensityGradientImage intensityGradients , double strongThresh )
{
    
    // Remove all edges below threshold that arent connected to strong edge (set to 0 maybe)

    return intensityGradients;
}



// Displays (sends to an output folder) image given by intensity gradient array
void displayImage( Vector2 * intensityGradients )
{



}

// Main function to run everything
int main()
{

    
    // Get the image file an prep for transformation
    IntensityGradientImage img = getImage("./images/NYScene.jpg");



    // free img allocation
    stbi_image_free(img.intensityGradientArray);

    return 0;
}