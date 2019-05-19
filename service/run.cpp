#include <fstream>
#include <io.h>

#include "wordRecognize.h"
#include "toolUtils.h"

int tmain() {
	cv::Mat src_img = cv::imread("0021.bmp", cv::IMREAD_GRAYSCALE);
	cv::Mat bin_img;
	float degree;
	vector<vector<rectStruct>> a = getSingleWordRect(src_img, bin_img, degree, true);
	showColLoc(src_img, a, "a");
	cv::waitKey();

}
