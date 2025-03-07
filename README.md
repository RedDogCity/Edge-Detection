# Edge-Detection
COP4520-25Spring 
# Overview
This project implements an edge detection algorithm in C++ using the stb_image library for image processing. It reads an input image, processes it into a grayscale format, and applies edge detection techniques to generate an output image.
Getting Started




# 1. Ensure You're in the Correct Directory

Before compiling and running the program, navigate to the project directory:

cd ~/Edge-Detection

Your terminal should look something like this:

exampleuser@exampleuser-Think-408:~/Edge-Detection$

# 2. Compile the C++ Program

Use the following command to compile the program:

g++ -o EdgeDetection EdgeDetection.cpp -std=c++11 -lm

Ensure you have GCC installed before compiling.
# 3. Run the Executable

After compilation, run the program with:

./EdgeDetection

# Project Structure

Edge-Detection/
│── .vscode/                # VS Code configuration files
│── build/                  # (Optional) Directory for compiled files
│── images/                 # Folder containing input images
│   ├── NYScene.jpg
│   ├── Rural.jpg
│── EdgeDetection.cpp       # Main C++ source file
│── EdgeDetection.exe       # Executable file (Windows)
│── output.png              # Output image after processing
│── stb_image.h             # Image loading library
│── stb_image_write.h       # Image writing library
│── .gitignore              # Git ignore file
│── README.md               # Project documentation

# Required Files

Ensure that the following files are present in the project folder:

    stb_image.h
    stb_image_write.h
    EdgeDetection.cpp
    .gitignore
    images/ (Folder containing input images)
    build/ (Optional folder for compiled files)

# Input & Output
Input Image

    The input images are stored in the images/ folder.
    The program loads an image using:

    IntensityGradientImage img = getImage("./images/Rural.jpg"); 

    You can replace "Rural.jpg" with any other image in the images/ folder.

# Output Image

    The program processes the input image and generates a grayscale image with edge detection applied.
    The final result is saved as:

output.png


