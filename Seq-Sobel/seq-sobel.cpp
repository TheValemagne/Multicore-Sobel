#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <iostream>
#define NANO_TO_MILLI 1e-6
#define IMAGE_PATH "../images/"
#define IMAGE_DIMENSION "4500"

using namespace cv;
using namespace std;

/**
 * @brief Horizontal Sobel as naive implementation
 * 
 * @param image black-white image
 * @param height height of image
 * @param width width of image
 */
void horizontalSobel(Mat image, const int height, const int width) {
	for (int row = 0; row < height - 2; row++) {
		for (int col = 0; col < width - 2; col++) {
			uchar xDerivate =   image.at<uchar>(row, col)         - image.at<uchar>(row, col + 2)
				              + 2 * image.at<uchar>(row + 1, col) - 2 * image.at<uchar>(row + 1, col + 2)
				              + image.at<uchar>(row + 2, col)     - image.at<uchar>(row + 2, col + 2);

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
	horizontalSobel(image, image.rows, image.cols);
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