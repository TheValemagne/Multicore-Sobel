
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "iostream"
#define THREAD_DIM 20 // Cuda threadlimits in one block: 32 * 32 = 1024
#define NANO_TO_MILLI 1e-6

using namespace cv;
using namespace std;

/**
 * @brief Horizontal Sobel as parallel implementation on GPU. Version for Cuda with 2-Dim grid and 2-Dim blocks.
 *
 * @param image black-white image
 * @param result result image after horizental filtering
 * @param height height of image
 * @param width width of image
 */
__global__ void horizontalSobel(const uchar *image, uchar *result, const int height, const int width) {
    const int row = threadIdx.x + blockIdx.x * blockDim.x;
    const int col = threadIdx.y + blockIdx.y * blockDim.y;

    if(row <= height - 2 && col <= width - 2){
        uchar xDerivate = image[row * width + col] - image[row * width + col + 2]
	        + 2 * image[(row + 1) * width + col] - 2 * image[(row + 1) * width + col + 2]
	        + image[(row + 2) * width + col] - image[(row + 2) * width + col + 2];

        result[row * width + col] = xDerivate;
    }
}

/**
 * @brief Convert OpenCV matrix to uchar array.
 *
 * @param matrix data of image
 * @param array result images pixels in an array
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
 * @param array images pixels in an array
 * @param matrix data of image to update
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

int main(void) {
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

    // Prepare and convete images data
	const int PIXELS = image.rows * image.cols;

	auto *imageArray = (uchar *)malloc(PIXELS * sizeof(char));
	auto *imageResultArray = (uchar *)malloc(PIXELS * sizeof(char));
    matrixToArray(image, imageArray, image.rows, image.cols);

    // Cuda stuff
    dim3 threads(THREAD_DIM, THREAD_DIM, 1);
    dim3 blocks(ceil(image.rows/(double)THREAD_DIM), ceil(image.cols/(double)THREAD_DIM), 1);
    printf("Blocks: %d, threads per block: %d", blocks.x * blocks.y, threads.x * threads.y);

    uchar *devImageArray;
    uchar *devImageResultArray;
    cudaMalloc((void**)&devImageArray, PIXELS * sizeof(char));
    cudaMalloc((void**)&devImageResultArray, PIXELS * sizeof(char));

    cudaMemcpy(devImageArray, imageArray, PIXELS * sizeof(char), cudaMemcpyHostToDevice);

    auto begin = std::chrono::high_resolution_clock::now();
    horizontalSobel<<<blocks, threads>>>(devImageArray, devImageResultArray, image.rows, image.cols);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
	auto execTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    cudaMemcpy(imageResultArray, devImageResultArray, PIXELS * sizeof(char), cudaMemcpyDeviceToHost);
    arrayToMatrix(imageResultArray, image, image.rows, image.cols);

    cudaFree(devImageArray);
    cudaFree(devImageResultArray);
	free(imageArray);
	free(imageResultArray);

    string imageResultName = IMAGE_PATH + "horses_" + IMAGE_DIMENSION + "_sobel.jpg";
    imwrite(imageResultName, image);

    image.release();

    double execTimeSobel = ((double)execTime.count() * NANO_TO_MILLI);
	printf("\nExect time: %f ms\n", execTimeSobel);
}