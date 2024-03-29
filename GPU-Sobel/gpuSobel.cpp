#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <iostream>
#define NANO_TO_MILLI 1e-6

using namespace cv;
using namespace std;

/**
 * @brief Horizontal Sobel as parallel implementation on GPU. Version 1 with OpenMP-teams loop.
 *
 * @param image black-white images pixel array
 * @param result result pixel array after horizental sobel filtering
 * @param height height of image
 * @param width width of image
 */
void horizontalSobel(const uchar *image, uchar *result, const int height, const int width) {
	#pragma omp target teams loop shared(image)
	for (int row = 0; row < height - 2; row++) {
		for (int col = 0; col < width - 2; col++) {
			uchar xDerivate =   image[row * width + col]           - image[row * width + col + 2]
				              + 2 * image[(row + 1) * width + col] - 2 * image[(row + 1) * width + col + 2]
				              + image[(row + 2) * width + col]     - image[(row + 2) * width + col + 2];

			result[row * width + col] = xDerivate;
		}
	}
}

/**
 * @brief Horizontal Sobel as parallel implementation on GPU. Version 2 with OpenMP-teams loop and parallel regions.
 * 
 * @param image black-white images pixel array
 * @param result result pixel array after horizental sobel filtering
 * @param height height of image
 * @param width width of image
 */
void horizontalSobel2(const uchar *image, uchar *result, const int height, const int width) {
	#pragma omp target teams loop shared(image)
	for (int row = 0; row < height - 2; row++) {
		#pragma omp parallel for shared(image)
		for (int col = 0; col < width - 2; col++) {
			uchar xDerivate =   image[row * width + col]           - image[row * width + col + 2]
				              + 2 * image[(row + 1) * width + col] - 2 * image[(row + 1) * width + col + 2]
				              + image[(row + 2) * width + col]     - image[(row + 2) * width + col + 2];

			result[row * width + col] = xDerivate;
		}
	}
}

/**
 * @brief Horizontal Sobel as parallel implementation on GPU. Version 3 with OpenMP-teams loop and collapse.
 *
 * @param image black-white images pixel array
 * @param result result pixel array after horizental sobel filtering
 * @param height height of image
 * @param width width of image
 */
void horizontalSobel3(const uchar *image, uchar *result, const int height, const int width) {
	#pragma omp target teams loop collapse(2) shared(image)
	for (int row = 0; row < height - 2; row++) {
		for (int col = 0; col < width - 2; col++) {
			uchar xDerivate =   image[row * width + col]           - image[row * width + col + 2]
				              + 2 * image[(row + 1) * width + col] - 2 * image[(row + 1) * width + col + 2]
				              + image[(row + 2) * width + col]     - image[(row + 2) * width + col + 2];

			result[row * width + col] = xDerivate;
		}
	}
}

/**
 * @brief Horizontal Sobel as parallel implementation on GPU. Version 4 with OpenMP-teams parallel for and  collapse.
 *
 * @param image black-white images pixel array
 * @param result result pixel array after horizental sobel filtering
 * @param height height of image
 * @param width width of image
 */
void horizontalSobel4(const uchar *image, uchar *result, const int height, const int width) {
	#pragma omp target teams distribute parallel for collapse(2) shared(image)
	for (int row = 0; row < height - 2; row++) {
		for (int col = 0; col < width - 2; col++) {
			uchar xDerivate =   image[row * width + col]           - image[row * width + col + 2]
				              + 2 * image[(row + 1) * width + col] - 2 * image[(row + 1) * width + col + 2]
				              + image[(row + 2) * width + col]     - image[(row + 2) * width + col + 2];

			result[row * width + col] = xDerivate;
		}
	}
}

/**
 * @brief Convert OpenCV matrix to uchar array.
 *
 * @param matrix images pixel matrix
 * @param array result pixels array
 * @param rows height of image
 * @param cols width of image
 */
void matrixToArray(const Mat matrix, uchar *array, const int rows, const int cols){
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            array[row * cols + col] = matrix.at<uchar>(row, col);
        }
    }
}

/**
 * @brief Update OpenCV matrix with data of uchar array.
 *
 * @param array images pixels array
 * @param matrix result pixel matrix
 * @param rows height of image
 * @param cols width of image
 */
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

	// Prepare and convert images data
	const int PIXELS = image.rows * image.cols;
	const int IMAGE_SIZE = PIXELS * sizeof(char);

	auto *imageArray = (uchar *)malloc(IMAGE_SIZE);
	auto *imageResultArray = (uchar *)malloc(IMAGE_SIZE);
	matrixToArray(image, imageArray, image.rows, image.cols);

	// Transfer data from cpu to gpu, only need to alloc space for imageResultArray
	#pragma omp target enter data map(to: imageArray[0:PIXELS]) map(alloc: imageResultArray[0:PIXELS])

	#pragma omp target
	{
		if (omp_is_initial_device()) {
			printf("Running on host\n");    
		} else {
			printf("Running on device\n");
		}
	}

	auto begin = std::chrono::high_resolution_clock::now();
	horizontalSobel3(imageArray, imageResultArray, image.rows, image.cols);
	auto end = std::chrono::high_resolution_clock::now();
	auto execTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

	// Tranfert result from gpu to cpu and delete/release data on GPU
	#pragma omp target exit data map(delete: imageArray[0:PIXELS]) map(from: imageResultArray[0:PIXELS])

	// Convert data to Matrix and free no needed data
	arrayToMatrix(imageResultArray, image, image.rows, image.cols);
	free(imageArray);
	free(imageResultArray);
	
	string imageResultName = IMAGE_PATH + "horses_" + IMAGE_DIMENSION + "_sobel.jpg";
	imwrite(imageResultName, image);
	image.release();

	double execTimeSobel = ((double)execTime.count() * NANO_TO_MILLI);
	printf("\nExect time: %f ms\n", execTimeSobel);

	return 0;
}