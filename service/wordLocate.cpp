#include "wordLocate.h"


vector<vector<rectStruct>> getSingleWordRect(cv::Mat& src_img, cv::Mat& bin_img, float& degree, bool debug) {
	vector<vector<rectStruct>> merge_word_loc_list;
	vector<rectStruct> rect_list;
	float wb_percent, word_height;
	cv::Mat proc_img = src_img.clone();

	// 光照均衡
	equalizeBrightness(proc_img, proc_img);
	wb_percent = getWBPercent(proc_img);

	// 去噪
	cv::GaussianBlur(proc_img, proc_img, cv::Size(3, 3), 0);

	// 二值化
	bin_img = binImg(proc_img, wb_percent);
	rect_list = getConnectDomain(bin_img);

	// 检测到前景框低于2直接返回
	if (rect_list.size() > 2) {
		word_height = getWordHeight(rect_list);
	}
	else {
		bin_img = binImg(src_img, wb_percent);
		rect_list = getConnectDomain(bin_img);
		if (rect_list.size() > 2) {
			word_height = getWordHeight(rect_list);
		}
		else {
			return merge_word_loc_list;
		}
	}
	
	// 排除非文字干扰项
	deNoise(bin_img, word_height);

	// 倾斜校正
	degree = calcDegree(bin_img);
	
	if (abs(degree) > 1.5) {
		bin_img = rotateImg(bin_img, degree);
		src_img = rotateImg(src_img, degree);
	}
	else {
		degree = 0;
	}

	
	// 连通域分析
	rect_list = getConnectDomain(bin_img);
	if (rect_list.size() > 2) {
		word_height = getWordHeight(rect_list);
	}
	else {
		return merge_word_loc_list;
	}

	rect_list = mergeRect(rect_list, word_height);

	
	// 获取行的并查集
	vector<vector<rectStruct>> set_list;
	set_list = rowColCheckSet(rect_list, word_height);
	if (debug) {
		cv::Mat set_img = src_img.clone();
		showSetList(set_img, set_list);
	}
	
	// 聚合并查集
	vector<rectStruct> row_loc_list;
	row_loc_list = getRowLocList(set_list);
	if (debug) {
		cv::Mat row_img = src_img.clone();
		showRowLoc(row_img, row_loc_list, "(1)row_loc");
	}

	// 行合并
	vector<rectStruct> merge_row_loc_list;
	merge_row_loc_list = mergeRowByRect(row_loc_list);
	if (debug) {
		cv::Mat merge_row_img = src_img.clone();
		showRowLoc(merge_row_img, merge_row_loc_list, "(2)merge_row_loc");
	}

	// 列切分
	vector<vector<rectStruct>> col_loc_list;
	col_loc_list = getColLocList(bin_img, merge_row_loc_list, word_height);
	if (debug) {
		cv::Mat col_img = src_img.clone();
		showColLoc(col_img, col_loc_list, "(3)col_loc");
	}

	// 文字定位
	vector<vector<rectStruct>> split_word_loc_list;
	vector<rectStruct> word_loc_list;
	rectStruct word_loc;
	for (int i = 0; i < col_loc_list.size(); i++) {
		for (int j = 0; j < col_loc_list[i].size(); j++) {
			word_loc = getWordLocList(bin_img, col_loc_list[i][j], word_height);
			if (word_loc.order > 0) {
				word_loc_list.push_back(word_loc);
			}
		}

		if (word_loc_list.size() > 0) {
			split_word_loc_list.push_back(word_loc_list);
			word_loc_list.clear();
		}

	}

	// 重组文字框
	vector<vector<rectStruct>> regroup_split_word_loc_list;
	regroup_split_word_loc_list = regroupWordLocList(split_word_loc_list, bin_img, word_height, debug);
	if (debug) {
		cv::Mat word_img = src_img.clone();
		showColLoc(word_img, regroup_split_word_loc_list, "(4)word_loc");
	}
	
	// 文字合并
	merge_word_loc_list = mergeWordLocList(regroup_split_word_loc_list, word_height);
	if (debug) {
		cv::Mat merge_word_img = src_img.clone();
		showColLoc(merge_word_img, merge_word_loc_list, "(5)merge_word_loc");
	}

	return merge_word_loc_list;
}
