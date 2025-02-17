// This file is an implimentation of an edge detection model using the canny algorithm
// This edge detection model is then improved using parallel programming concepts to decrease runtime

// *** NOTE *** some function types may need to change from skeleton to implimentation

#include <vector>

// Main function to run everything
int main()
{

    // run functions step by step in sequential order

    return 0;
}


// Getting image from somewhere (Initial implimentation will likely skip this step and set a predefined 2d array later)
void getImage() // should return image and not void
{

}



// Reduce image to a grayscale
void toGrayScale() // should return grayscaled image and should have an image parameter
{

}



// Apply a gausian filter to the image to smooth and remove noise
void gausianFilter() // should return filtered image and have grayscaled image as parameter
{
    
}



// Create a 2d array (vectors) of vector2's (structure) that represent the intesity gradients in the image (direction of edges)
struct Vector2 {
    int x, y;
    
    Vector2( int x = 0, int y = 0 ) : x(x), y(y) {}
};

std::vector<std::vector<Vector2>> intensityGradient( int width, int height ) // parameters are temporary unless this works well (probably change to smoothed image as parameter)
{
    // Create a 2D vector of Vector2 objects to store intensity gradients.
    std::vector<std::vector<Vector2>> gradients(height, std::vector<Vector2>(width));

    // Calculate the intensity gradient of each pixel (round direction to the be one of the four edge directions ( | , - , / , \ ) )
    

    return gradients;
}



// Apply gradient magnitude thresholding to find actual image edges from the gradients
std::vector<std::vector<Vector2>> magnitudeThreshold( std::vector<std::vector<Vector2>> intensityGradients , double weakThresh ) // parameters are temporary unless this works well (probably change to smoothed image as parameter)
{
    
    // Remove all edges below threshold (set to 0 maybe)

    return intensityGradients;
}



// Filter out noisey edges using threshold filter 
// (Strong edges are accepted, weak are accepted if connected to strong)
std::vector<std::vector<Vector2>> hysteresis( std::vector<std::vector<Vector2>> intensityGradients , double strongThresh ) // parameters are temporary unless this works well (probably change to smoothed image as parameter)
{
    
    // Remove all edges below threshold that arent connected to strong edge (set to 0 maybe)

    return intensityGradients;
}



// Convert intesity gradients to image to show edges image (or midstep images)
void convertToImage( std::vector<std::vector<Vector2>> intensityGradients ) // return type should be an image
{

    // convert intesity gradients to an image format

}