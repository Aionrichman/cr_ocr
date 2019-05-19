#include "mathTool.h"


bool leftComp(const rectStruct &a, const rectStruct &b) {
	return a.left < b.left;
}


bool widthComp(const rectStruct &a, const rectStruct &b) {
	return (a.right - a.left) < (b.right - b.left);
}


bool heightComp(const rectStruct &a, const rectStruct &b) {
	return (a.down - a.top) < (b.down - b.top);
}


void rowSortRotate(vector<cv::RotatedRect> &rect_list, float word_height) {
	cv::RotatedRect tmp_rect;
	for (int i = 0; i < rect_list.size(); i++) {
		for (int j = 0; j < rect_list.size(); j++) {
			if (abs(rect_list[i].center.y - rect_list[j].center.y) < 0.8*word_height) {
				if (rect_list[i].center.x < rect_list[j].center.x) {
					tmp_rect = rect_list[i];
					rect_list[i] = rect_list[j];
					rect_list[j] = tmp_rect;
				}
			}
			else if (rect_list[i].center.y < rect_list[j].center.y) {
				tmp_rect = rect_list[i];
				rect_list[i] = rect_list[j];
				rect_list[j] = tmp_rect;
			}
		}
	}
}


void rowSort(vector<rectStruct> &rect_list, float word_height) {
	rectStruct tmp_rect;
	for (int i = 0; i < rect_list.size(); i++) {
		for (int j = 0; j < rect_list.size(); j++) {
			if (abs(rect_list[i].top - rect_list[j].top) < 0.8*word_height) {
				if (rect_list[i].left < rect_list[j].left) {
					tmp_rect = rect_list[i];
					rect_list[i] = rect_list[j];
					rect_list[j] = tmp_rect;
				}
			}
			else if (rect_list[i].top < rect_list[j].top) {
				tmp_rect = rect_list[i];
				rect_list[i] = rect_list[j];
				rect_list[j] = tmp_rect;
			}
		}
	}
}


bool rowComp(const rectStruct &a, const rectStruct &b) {
	bool flag;
	float mid_height = min((a.down - a.top), (b.down - b.top));

	if (abs(a.top - b.top) < mid_height) {
		if (a.left < b.left) {
			flag = true;
		}
		else
		{
			flag = false;
		}
	}
	else if (a.top < b.top) {
		flag = true;
	}
	else
	{
		flag = false;
	}

	return flag;
}


float quantileHeight(vector<rectStruct> rect_list, float quantile) {
	sort(rect_list.begin(), rect_list.end(), heightComp);
	int median_num = rect_list.size() * quantile;

	return (rect_list[median_num].down - rect_list[median_num].top);
}


float absTan(double x1, double y1, double x2, double y2) {
	float abs_tan;
	if (x1 == x2) {
		if (y1 == y2) {
			abs_tan = 0.0;
		}
		else {
			abs_tan = 10.0;
		}
	}
	else
	{
		abs_tan = abs((y1 - y2) / (x1 - x2));
	}
	return abs_tan;
}