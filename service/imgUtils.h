#ifndef _IMG_UTILS_H_
#define _IMG_UTILS_H_

#include "mathTool.h"


void equalizeBrightness(cv::Mat& src_img, cv::Mat& res_img, int kernelSize = BRIGHTNESS_KERNEL);
void adaptiveBinaryzation(cv::Mat& src_img, cv::Mat& res_img);
void deNoise(cv::Mat& src_img, int word_height);
float getWBPercent(cv::Mat src_img);
double calcDegree(cv::Mat src_img);
cv::Mat rotateImg(cv::Mat src_img, double degree);
cv::Mat binImg(cv::Mat src_img, float wb_percent, int const_c = CONST_C);
vector<rectStruct> getConnectDomain(cv::Mat src_img);
vector<rectStruct> splitCol(cv::Mat src_img, rectStruct loc_rect);
vector<rectStruct> overSplitCol(cv::Mat src_img, rectStruct loc_rect);
vector<vector<rectStruct>> getColLocList(cv::Mat src_img, vector<rectStruct> row_loc_list, int word_width);
rectStruct connectToRect(cv::Mat src_img, rectStruct loc, float word_height);
vector<cv::Mat> getImgList(cv::Mat src_img, vector<rectStruct> word_loc_list);
vector<cv::Mat> getImgListAll(cv::Mat src_img, vector<vector<rectStruct>> word_loc_list);


void showSetList(cv::Mat src_img, vector<vector<rectStruct>> set_list);
void showRowLoc(cv::Mat src_img, vector<rectStruct> row_loc_list, string file_name = "row_loc");
void showColLoc(cv::Mat src_img, vector<vector<rectStruct>> col_loc_list, string file_name = "col_loc");


class RLSA
{
public:
	static void horizontalRunLengthSmooth(cv::Mat& src, cv::Mat& dst, int hthreshold, uchar whitePxValue = 255);

	static void verticalRunLengthSmooth(cv::Mat& src, cv::Mat& dst, int threshold, uchar whitePxValue = 255);

	static void runLengthSmooth(cv::Mat& src, cv::Mat& dst, int hsv, int vsv, int asv, uchar whitePxValue = 255);
};


#endif // !_IMG_UTILS_H_