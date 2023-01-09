#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#define NANO_TO_MILLI 1e-6
#define IMAGE_PATH "../images/"

using namespace cv;
using namespace std;

/**
 * @brief Horizontal Sobel as naive implementation
 * 
 * @param image black-white image
 * @param height height of image
 * @param width width of image
 */
void horizontalSobel(Mat image, int height, int width) {
	int xDerivates[3][3] = {
		{1, 0, -1},
		{2, 0, -2},
		{1, 0, -1}
	};

	for (int row = 0; row < height - 2; row++) {
		for (int col = 0; col < width - 2; col++) {
			int xDerivate = 0;

			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					xDerivate += xDerivates[i][j] * (int)image.at<uchar>(row + i, col + j);
				}
			}

			image.at<uchar>(row, col) = (uchar)xDerivate;
		}
	}
}

int main(int argc, char** argv)
{
	// Read the image file
	Mat image = imread(IMAGE_PATH"horses_1920.jpg", IMREAD_GRAYSCALE);

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


	// Show our image inside a window.
	imshow("Image result with horizental sobel operator", image);

	// Wait for any keystroke in the window
	waitKey(0);
	imwrite(IMAGE_PATH"horses_1920_sobel.jpg", image);

	double execTimeSobel = ((double)execTime.count() * NANO_TO_MILLI);
	printf("\n\nExect time: %f ms\n", execTimeSobel);

	image.release();

	return 0;
}