// This file is an implimentation of an edge detection model using the canny algorithm
// This edge detection model is then improved using parallel programming concepts to decrease runtime

// *** NOTE *** some function types may need to change from skeleton to implimentation

#include <vector>
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

// Gets image from file path and returns gray image as 2d vector
Vector2 * getImage(const char* imgPath) 
{
    int width, height, channels;
    unsigned char *img = stbi_load(imgPath, &width, &height, &channels, 0);
    if(img == NULL || channels != 3) 
    {
        printf("Error in loading the image\n");
        exit(1);
    }
    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);

    size_t img_size = width * height * channels;
    size_t gray_img_size = width * height;
    
    Vector2 *gray_img = (Vector2 *)malloc(gray_img_size * sizeof(Vector2));
    if(gray_img == NULL) {
        printf("Unable to allocate memory for the gray image.\n");
        exit(1);
    }

    int i = 0;
    for(unsigned char *p = img; p != img + img_size; p += channels, i += 1) 
    {
        gray_img[i] = Vector2(((*p + *(p + 1) + *(p + 2))/3.0), 0.0);
    }

    return gray_img;

}



// Apply a gausian filter to the image to smooth and remove noise
void gausianFilter() // should return filtered image and have grayscaled image as parameter
{
    
}

// Create a 2d array (vectors) of vector2's (structure) that represent the intesity gradients in the image (direction of edges)
vector<vector<Vector2>> intensityGradient( int width, int height ) // parameters are temporary unless this works well (probably change to smoothed image as parameter)
{
    // Create a 2D vector of Vector2 objects to store intensity gradients.
    vector<vector<Vector2>> gradients(height, vector<Vector2>(width));

    // Calculate the intensity gradient of each pixel (round direction to the be one of the four edge directions ( | , - , / , \ ) )
    

    return gradients;
}



// Apply gradient magnitude thresholding to find actual image edges from the gradients
vector<vector<Vector2>> magnitudeThreshold( vector<vector<Vector2>> intensityGradients , double weakThresh ) // parameters are temporary unless this works well (probably change to smoothed image as parameter)
{
    
    // Remove all edges below threshold (set to 0 maybe)

    return intensityGradients;
}



// Filter out noisey edges using threshold filter 
// (Strong edges are accepted, weak are accepted if connected to strong)
vector<vector<Vector2>> hysteresis( vector<vector<Vector2>> intensityGradients , double strongThresh ) // parameters are temporary unless this works well (probably change to smoothed image as parameter)
{
    
    // Remove all edges below threshold that arent connected to strong edge (set to 0 maybe)

    return intensityGradients;
}



// Convert intesity gradients to image to show edges image (or midstep images)
void convertToImage( vector<vector<Vector2>> intensityGradients ) // return type should be an image
{

    // convert intesity gradients to an image format

}

// Main function to run everything
int main()
{

    // run functions step by step in sequential order
    
    Vector2 *img = getImage("./images/NYScene.jpg");

    // free img allocation
    stbi_image_free(img);

    return 0;
}