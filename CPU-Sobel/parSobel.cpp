#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#define NANO_TO_MILLI 1e-6
#define IMAGE_PATH "../images/"
#define IMAGE_DIMENSION "4500"

using namespace cv;
using namespace std;

/**
 * @brief Horizontal Sobel as naive parallel implementation on CPU.
 *
 * @param image black-white image
 * @param height height of image
 * @param width width of image
 */
void horizontalSobel(Mat image, const int height, const int width) {
	int xDerivates[3][3] = {
		{1, 0, -1},
		{2, 0, -2},
		{1, 0, -1}
	};

	#pragma omp parallel for
	for (int row = 0; row < height - 2; row++) {
		for (int col = 0; col < width - 2; col++) {
			uchar xDerivate = 0;

			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					xDerivate += xDerivates[i][j] * image.at<uchar>(row + i, col + j);
				}
			}

			image.at<uchar>(row, col) = xDerivate;
		}
	}
}

/**
 * @brief Horizontal Sobel as parallel implementation on CPU. Version 2 with OpenMP-parallel for.
 *
 * @param image black-white image
 * @param height height of image
 * @param width width of image
 */
void horizontalSobel2(Mat image, const int height, const int width) {
	#pragma omp parallel for schedule(static)
	for (int row = 0; row < height - 2; row++) {
		for (int col = 0; col < width - 2; col++) {
			uchar xDerivate = image.at<uchar>(row, col) - image.at<uchar>(row, col + 2)
				+ 2 * image.at<uchar>(row + 1, col) - 2 * image.at<uchar>(row + 1, col + 2)
				+ image.at<uchar>(row + 2, col) - image.at<uchar>(row + 2, col + 2);

			image.at<uchar>(row, col) = xDerivate;
		}
	}
}

/**
 * @brief Horizontal Sobel as parallel implementation on CPU. Version 3 with OpenMP-teams.
 *
 * @param image black-white image
 * @param height height of image
 * @param width width of image
 */
void horizontalSobel3(Mat image, const int height, const int width) {
	#pragma omp teams loop num_teams(3) thread_limit(4)
	for (int row = 0; row < height - 2; row++) {
		for (int col = 0; col < width - 2; col++) {
			uchar xDerivate = image.at<uchar>(row, col) - image.at<uchar>(row, col + 2)
				+ 2 * image.at<uchar>(row + 1, col) - 2 * image.at<uchar>(row + 1, col + 2)
				+ image.at<uchar>(row + 2, col) - image.at<uchar>(row + 2, col + 2);

			image.at<uchar>(row, col) = xDerivate;
		}
	}
}

/**
 * @brief Horizontal Sobel as parallel implementation on CPU. Version 4 with OpenMP-teams and parralel region.
 *
 * @param image black-white image
 * @param height height of image
 * @param width width of image
 */
void horizontalSobel4(Mat image, const int height, const int width) {
	#pragma omp teams loop num_teams(2) thread_limit(4)
	for (int row = 0; row < height - 2; row++) {
		#pragma omp parallel for
		for (int col = 0; col < width - 2; col++) {
			uchar xDerivate = image.at<uchar>(row, col) - image.at<uchar>(row, col + 2)
				+ 2 * image.at<uchar>(row + 1, col) - 2 * image.at<uchar>(row + 1, col + 2)
				+ image.at<uchar>(row + 2, col) - image.at<uchar>(row + 2, col + 2);

			image.at<uchar>(row, col) = xDerivate;
		}
	}
}

int main(int argc, char** argv)
{
	// Read the image file
	string imageName = std::format("{}horses_{}.jpg", IMAGE_PATH, IMAGE_DIMENSION);
	Mat image = imread(imageName, IMREAD_GRAYSCALE);

	// Check for failure
	if (image.empty()) {
		cout << "Image Not Found!!!" << endl;
		cin.get(); //wait for any key press
		return -1;
	}

	auto begin = std::chrono::high_resolution_clock::now();
	horizontalSobel2(image, image.rows, image.cols);
	auto end = std::chrono::high_resolution_clock::now();
	auto execTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

	// Wait for any keystroke in the window
	waitKey(0);
	string imageResultName = std::format("{}horses_{}_sobel.jpg", IMAGE_PATH, IMAGE_DIMENSION);
	imwrite(imageResultName, image);
	image.release();

	double execTimeSobel = ((double)execTime.count() * NANO_TO_MILLI);
	printf("\n\nExect time: %f ms\n", execTimeSobel);

	return 0;
}