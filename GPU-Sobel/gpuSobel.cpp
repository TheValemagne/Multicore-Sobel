#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#define NANO_TO_MILLI 1e-6

using namespace cv;
using namespace std;

/**
 * @brief Horizontal Sobel as parrallel implementation. Version 3 with OpenMP-teams.
 *
 * @param image black-white image
 * @param height height of image
 * @param width width of image
 */
void horizontalSobel(const uchar *image, uchar *result, const int height, const int width) {
	#pragma omp target
	#pragma omp teams loop shared(image)
	for (int row = 0; row < height - 2; row++) {
		for (int col = 0; col < width - 2; col++) {
			int xDerivate = (int)image[row * width + col] - (int)image[row * width + col + 2]
				+ 2 * (int)image[(row + 1) * width + col] - 2 * (int)image[(row + 1) * width + col + 2]
				+ (int)image[(row + 2) * width + col] - (int)image[(row + 2) * width + col + 2];

			result[row * width + col] = (uchar)xDerivate;
		}
	}
}

void matrixToArray(const Mat matrix, uchar *array, const int rows, const int cols){
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            array[row * cols + col] = matrix.at<uchar>(row, col);
        }
    }
}

void arrayToMatrix(const uchar *array, Mat matrix, const int rows, const int cols){
    for (auto row = 0; row < rows; row++) {
        for (auto col = 0; col < cols; col++) {
            matrix.at<uchar>(row, col) = array[row * cols + col];
        }
    }
}

int main(int argc, char** argv)
{
	// Read the image file
    const string IMAGE_PATH = "../images/";
    const string IMAGE_DIMENSION = "4500";

	string imageName = IMAGE_PATH + "horses_" + IMAGE_DIMENSION + ".jpg";
	Mat image = imread(imageName, IMREAD_GRAYSCALE);

	// Check for failure
	if (image.empty()) {
		cout << "Image Not Found!!!" << endl;
		cin.get(); //wait for any key press
		return -1;
	}

	const int imageSize = image.rows * image.cols * sizeof(char);
	const int pixels = image.rows * image.cols;
	auto imageArray = (uchar *)malloc(imageSize);
	auto imageResultArray = (uchar *)malloc(imageSize);
	matrixToArray(image, imageArray, image.rows, image.cols);

	// cpu to gpu
	#pragma omp target enter data map(to: imageArray[0:pixels], imageResultArray[0:pixels])

	#pragma omp target
	{
		if (omp_is_initial_device()) {
			printf("Running on host\n");    
		} else {
			printf("Running on device\n");
		}
	}


	auto begin = std::chrono::high_resolution_clock::now();
	horizontalSobel(imageArray, imageResultArray, image.rows, image.cols);
	auto end = std::chrono::high_resolution_clock::now();
	auto execTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

	// gpu to cpu
	#pragma omp target exit data map(from: imageResultArray[0:pixels])
	arrayToMatrix(imageResultArray, image, image.rows, image.cols);

	string imageResultName = IMAGE_PATH + "horses_" + IMAGE_DIMENSION + "_sobel.jpg";
	imwrite(imageResultName, image);

	double execTimeSobel = ((double)execTime.count() * NANO_TO_MILLI);
	printf("\n\nExect time: %f ms\n", execTimeSobel);

	image.release();

	return 0;
}