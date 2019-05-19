#ifndef _LOCATE_H_
#define _LOCATE_H_

#include "checkSet.h"


vector<vector<rectStruct>> getSingleWordRect(cv::Mat& src_img, cv::Mat& bin_img, float& degree, bool debug = false);


#endif // !_LOCATE_H_

