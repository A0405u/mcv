/*  For description look into the help() function. */

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <arm_neon.h>
#include <chrono>

using namespace std;
using namespace cv;

void binarize(const uint8_t* rgb, uint8_t* binarized, const uint8_t threshold, int num_pixels)
{
	auto t1_cv = chrono::high_resolution_clock::now();
	for(int i=0; i<num_pixels; ++i, rgb+=3) {

		int v = (77*rgb[0] + 150*rgb[1] + 29*rgb[2]);
		binarized[i] = ((v>>8) > threshold) * 255;
	}
	auto t2_cv = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(t2_cv-t1_cv).count();
	cout << "inside function binarize" << endl;
	cout << duration << " us" << endl;
}

void binarize_cv(const Mat rgb, Mat binarized, const uint8_t threshold, int num_pixels)
{
	auto t1 = chrono::high_resolution_clock::now();
	
	Mat gray(rgb.rows, rgb.cols, CV_8UC1, Scalar(0));

	cvtColor(rgb, gray, COLOR_RGB2GRAY);
	cv::threshold(gray, binarized, threshold, 255, THRESH_BINARY);

	auto t2 = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(t2-t1).count();
	cout << "inside function binarize" << endl;
	cout << duration << " us" << endl;
}

void binarize_neon(const uint8_t* rgb, uint8_t* binarized, const uint8_t threshold, int num_pixels)
{
	// We'll use 64-bit NEON registers to process 8 pixels in parallel.
	num_pixels /= 8;

	// Duplicate the weight 8 times.
	uint8x8_t w_r = vdup_n_u8(77);
	uint8x8_t w_g = vdup_n_u8(150);
	uint8x8_t w_b = vdup_n_u8(29);

	// For intermediate results. 16-bit/pixel to avoid overflow.
	uint16x8_t temp;

	// For the converted grayscale values.
	uint8x8_t gray;

	// For threshold value.
	uint8x8_t threshold_vector;

	// Loading threshold value into register.
	threshold_vector = vld1_dup_u8(&threshold);

	// For binarized values.
	uint8x8_t result;

	auto t1_neon = chrono::high_resolution_clock::now();
	for(int i=0; i<num_pixels; ++i, rgb+=8*3, binarized+=8) 
	{
		// Load 8 pixels into 3 64-bit registers, split by channel.
		uint8x8x3_t src = vld3_u8(rgb);

		// Multiply all eight red pixels by the corresponding weights.
		temp = vmull_u8(src.val[0], w_r);

		// Combined multiply and addition.
		temp = vmlal_u8(temp, src.val[1], w_g);
		temp = vmlal_u8(temp, src.val[2], w_b);
		
		// Shift right by 8, "narrow" to 8-bits (recall temp is 16-bit).
		gray = vshrn_n_u16(temp, 8);
		
		// Compare pixels with threshold value.
		result = vclt_u8(threshold_vector, gray);
		
		// Store converted pixels in the output grayscale image.
		vst1_u8(binarized, result);
	}

	auto t2_neon = chrono::high_resolution_clock::now();
	auto duration_neon = chrono::duration_cast<chrono::microseconds>(t2_neon-t1_neon).count();
	cout << "inside function binarized_neon" << endl;
	cout << duration_neon << " us" << endl;
}

int main(int argc,char** argv)
{
	uint8_t * rgb_arr;
	uint8_t * binarized;
	uint8_t * binarized_cv;
	uint8_t * binarized_neon;

	if (argc != 3) {
		cout << "Usage: opencv_neon image_name" << endl;
		return -1;
	}

	Mat rgb_image;
	rgb_image = imread(argv[1], IMREAD_COLOR);
	if (!rgb_image.data) {
		cout << "Could not open the image" << endl;
		return -1;
	}
	if (rgb_image.isContinuous()) {
		rgb_arr = rgb_image.data;
	}
	else {
		cout << "data is not continuous" << endl;
		return -2;
	}

	uint8_t threshold = stoi(argv[2]);

	int width = rgb_image.cols;
	int height = rgb_image.rows;
	int num_pixels = width*height;

	// Binarization
	Mat binarized_image(height, width, CV_8UC1, Scalar(0));
	binarized = binarized_image.data;

	auto t1 = chrono::high_resolution_clock::now();
	binarize(rgb_arr, binarized, threshold, num_pixels);
	auto t2 = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(t2-t1).count();
	cout << "binarize" << endl;
	cout << duration << " us" << endl;

	imwrite("binarized.png", binarized_image);

	// OpenCV binarization
	Mat binarized_image_cv(height, width, CV_8UC1, Scalar(0));

	auto t1_cv = chrono::high_resolution_clock::now();
	binarize_cv(rgb_image, binarized_image_cv, threshold, num_pixels);
	auto t2_cv = chrono::high_resolution_clock::now();
	auto duration_cv = chrono::duration_cast<chrono::microseconds>(t2_cv-t1_cv).count();
	cout << "binarize_cv" << endl;
	cout << duration_cv << " us" << endl;

	imwrite("binarized_cv.png", binarized_image_cv);

	// NEON binarization
	Mat binarized_image_neon(height, width, CV_8UC1, Scalar(0));
	binarized_neon = binarized_image_neon.data;

	auto t1_neon = chrono::high_resolution_clock::now();
	binarize_neon(rgb_arr, binarized_neon, threshold, num_pixels);
	auto t2_neon = chrono::high_resolution_clock::now();
	auto duration_neon = chrono::duration_cast<chrono::microseconds>(t2_neon-t1_neon).count();
	cout << "binarize_neon" << endl;
	cout << duration_neon << " us" << endl;

	imwrite("binarized_neon.png", binarized_image_neon);

    return 0;
}
