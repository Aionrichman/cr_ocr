#ifndef _CHECK_SET_H_
#define _CHECK_SET_H_

#include "imgUtils.h"

float getWordHeight(vector<rectStruct> rect_list);
vector<vector<rectStruct>> rowColCheckSet(vector<rectStruct> rect_list, float split_width, float thre_tan = THRE_TAN);
vector<rectStruct> getRowLocList(vector<vector<rectStruct>> set_list);
vector<rectStruct> mergeRect(vector<rectStruct> row_loc_list, int merge_width);
vector<rectStruct> mergeRowByRect(vector<rectStruct> row_loc_list);
vector<rectStruct> mergeWordByRect(vector<rectStruct> row_loc_list);
vector<vector<rectStruct>> mergeWordLocList(vector<vector<rectStruct>> col_loc_list, int avg_width);
rectStruct getWordLocList(cv::Mat src_img, rectStruct col_loc, float split_height);
vector<vector<rectStruct>> regroupWordLocList(vector<vector<rectStruct>> split_word_loc_list, cv::Mat bin_img, float word_height, bool debug = false);

#endif


