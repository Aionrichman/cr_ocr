#ifndef _MATH_TOOL_H_
#define _MATH_TOOL_H_

#include "constant.h"


bool leftComp(const rectStruct &a, const rectStruct &b);
bool widthComp(const rectStruct &a, const rectStruct &b);
bool heightComp(const rectStruct &a, const rectStruct &b);
bool rowComp(const rectStruct &a, const rectStruct &b);
void rowSort(vector<rectStruct> &rect_list, float word_height);
void rowSortRotate(vector<cv::RotatedRect> &rect_list, float word_height);
float quantileHeight(vector<rectStruct> rect_list, float quantile);
float absTan(double x1, double y1, double x2, double y2);


#endif // !_MATH_TOOL_H_