#include "checkSet.h"


float getWordHeight(vector<rectStruct> rect_list) {
	int n_comps = rect_list.size();
	float word_height;

	if (n_comps < 100) {
		word_height = quantileHeight(rect_list, QUANT_PERCEENT + 0.1);
	}
	else if (n_comps < 200) {
		word_height = quantileHeight(rect_list, QUANT_PERCEENT + 0.2);
	}
	else {
		word_height = quantileHeight(rect_list, QUANT_PERCEENT + 0.25);
	}

	return word_height;
}


vector<vector<rectStruct>> rowColCheckSet(vector<rectStruct> rect_list, float split_width, float thre_tan) {
	/*
	根据连通域间的正弦值与距离建立简单的行分类并查集
	*/
	vector<vector<rectStruct>> row_list;
	vector<rectStruct> col_list;
	rectStruct front_rect;

	int front_row_order, distance;
	float tmp_tan;

	sort(rect_list.begin(), rect_list.end(), leftComp);

	col_list.push_back(rect_list[0]);
	row_list.push_back(col_list);
	for (int i = 1; i < rect_list.size(); i++) {
		int find_set_order = 0;
		float min_tan = 100;
		for (int j = 0; j < row_list.size(); j++) {
			front_row_order = row_list[j].size() - 1;
			front_rect = row_list[j][front_row_order];
			tmp_tan = absTan(
				(float)(rect_list[i].left + rect_list[i].right) / 2,
				(float)(rect_list[i].top + rect_list[i].down) / 2,
				(float)(front_rect.left + front_rect.right) / 2,
				(float)(front_rect.top + front_rect.down) / 2);

			if (tmp_tan < min_tan) {
				distance = (rect_list[i].left - front_rect.right) + abs(rect_list[i].top - front_rect.top);
				if (abs(distance) < split_width) {
					find_set_order = j;
					min_tan = tmp_tan;
				}
			}
		}

		if (min_tan < thre_tan) {
			row_list[find_set_order].push_back(rect_list[i]);
		}
		else {
			col_list.clear();
			col_list.push_back(rect_list[i]);
			row_list.push_back(col_list);
		}

	}

	return row_list;
}

vector<rectStruct> getRowLocList(vector<vector<rectStruct>> set_list) {
	/*
	根据并查集生成行定位集合
	*/
	vector<rectStruct> res_list;
	rectStruct res;

	for (int k = 0; k < set_list.size(); k++) {
		res.left = set_list[k][0].left;
		res.top = set_list[k][0].top;
		res.right = set_list[k][set_list[k].size() - 1].right;
		res.down = set_list[k][0].down;
		for (int l = 0; l < set_list[k].size(); l++) {
			if (set_list[k][l].top < res.top) {
				res.top = set_list[k][l].top;
			}

			if (set_list[k][l].down > res.down) {
				res.down = set_list[k][l].down;
			}
		}
		res_list.push_back(res);
	}

	return res_list;
}


vector<rectStruct> mergeRect(vector<rectStruct> row_loc_list, int merge_width) {
	vector<rectStruct>::iterator it, it_next;
	cv::Rect intersect_rect, union_rect;
	float word_height;

	word_height = MAX_SIMILAR * merge_width;
	sort(row_loc_list.begin(), row_loc_list.end(), rowComp);

	for (it = row_loc_list.begin(); it != row_loc_list.end();it++) {
		for (it_next = it + 1; it_next != row_loc_list.end();) {
			cv::Rect A(
				it->left - merge_width,
				it->top - merge_width,
				it->right - it->left + merge_width,
				it->down - it->top + merge_width);
			cv::Rect B(
				it_next->left - merge_width,
				it_next->top - merge_width,
				it_next->right - it_next->left + merge_width,
				it_next->down - it_next->top + merge_width);
			cv::Rect A_(
				it->left,
				it->top,
				it->right - it->left,
				it->down - it->top);
			cv::Rect B_(
				it_next->left,
				it_next->top,
				it_next->right - it_next->left,
				it_next->down - it_next->top);
			intersect_rect = A & B;
			union_rect = A_ | B_;
			if ((intersect_rect.area() > 0)
				& (union_rect.height < word_height)
				& (union_rect.width < word_height)) {
				it->left = min(it->left, it_next->left);
				it->top = min(it->top, it_next->top);
				it->right = max(it->right, it_next->right);
				it->down = max(it->down, it_next->down);
				it_next = row_loc_list.erase(it_next);
			}
			else {
				it_next++;
			}
		}
	}

	return row_loc_list;
}


vector<rectStruct> mergeRowByRect(vector<rectStruct> row_loc_list) {
	vector<rectStruct>::iterator it, it_next;

	std::sort(row_loc_list.begin(), row_loc_list.end(), rowComp);
	for (it = row_loc_list.begin(); it != row_loc_list.end(); it++) {
		for (it_next = it + 1; it_next != row_loc_list.end();) {
			cv::Rect A(
				it->left,
				it->top,
				it->right - it->left,
				it->down - it->top);
			cv::Rect B(
				it_next->left,
				it_next->top,
				it_next->right - it_next->left,
				it_next->down - it_next->top);
			if ((A & B).area() > 0.6*min(A.area(), B.area())) {
				it->left = min(it->left, it_next->left);
				it->top = min(it->top, it_next->top);
				it->right = max(it->right, it_next->right);
				it->down = max(it->down, it_next->down);
				it_next = row_loc_list.erase(it_next);
			}
			else {
				it_next++;
			}
		}
	}
	
	return row_loc_list;
}


vector<rectStruct> mergeWordByRect(vector<rectStruct> word_loc_list) {
	vector<rectStruct>::iterator it, it_next;

	std::sort(word_loc_list.begin(), word_loc_list.end(), rowComp);

	for (it = word_loc_list.begin(); it != word_loc_list.end(); it++) {
		for (it_next = it + 1; it_next != word_loc_list.end();) {
			cv::Rect A(
				it->left,
				it->top,
				it->right - it->left,
				it->down - it->top);
			cv::Rect B(
				it_next->left,
				it_next->top,
				it_next->right - it_next->left,
				it_next->down - it_next->top);
			if ((A & B).area() > 0) {
				if (A.area() < B.area()) {
					it->order = it_next->order;
				}

				it->left = min(it->left, it_next->left);
				it->top = min(it->top, it_next->top);
				it->right = max(it->right, it_next->right);
				it->down = max(it->down, it_next->down);
				it_next = word_loc_list.erase(it_next);
			}
			else {
				it_next++;
			}
		}
	}

	return word_loc_list;
}


vector<vector<rectStruct>> mergeWordLocList(vector<vector<rectStruct>> col_loc_list, int avg_width) {
	vector<vector<rectStruct>> merge_word_list;
	vector<rectStruct> split_word_list, tmp_list;
	rectStruct col_loc;
	float  min_merge_width, max_merge_width, max_merge_height, tmp_width, tmp_height;
	int list_count, merge_width, tmp_width_interval1, tmp_width_interval2;
	bool merge_flag;

	for (int i = 0; i < col_loc_list.size(); i++) {
		list_count = col_loc_list[i].size() - 1;

		if (list_count < 0) {
			continue;
		}

		if (list_count == 0) {
			tmp_list.push_back(col_loc_list[i][0]);
		}
		else {
			if (list_count > 5) {
				merge_width = quantileHeight(col_loc_list[i], QUANT_PERCEENT + 0.2);
			}
			else
			{
				merge_width = avg_width;
			}
			
			min_merge_width = merge_width * MIN_SIMILAR;
			max_merge_width = merge_width * MAX_SIMILAR;

			for (int j = 0; j < list_count;) {
				col_loc = col_loc_list[i][j];
				merge_flag = false;
				tmp_width_interval1 = col_loc_list[i][j + 1].left - col_loc_list[i][j].right;
				for (int k = j + 1; k < list_count + 1; k++) {
					tmp_width = col_loc_list[i][k].right - col_loc.left;
					tmp_height = max(col_loc.down, col_loc_list[i][k].down) - min(col_loc.top, col_loc_list[i][k].top);

					max_merge_height = max(col_loc.down - col_loc.top, col_loc_list[i][k].down - col_loc_list[i][k].top);
					if (max_merge_height < max_merge_width) {
						max_merge_height = max_merge_width;
					}
					else {
						max_merge_height = max_merge_height * MAX_SIMILAR;
					}

					if ((tmp_width > max_merge_width) || (tmp_height > max_merge_height)) {
						break;
					}
					else {
						tmp_width_interval1 = max(tmp_width_interval1, (col_loc_list[i][k].left - col_loc_list[i][k - 1].right));
						if (k < list_count) {
							tmp_width_interval2 = col_loc_list[i][k + 1].left - col_loc_list[i][k].right;
						}
						else {
							tmp_width_interval2 = max_merge_width;
						}

						col_loc.top = min(col_loc.top, col_loc_list[i][k].top);
						col_loc.down = max(col_loc.down, col_loc_list[i][k].down);
						if ((tmp_width > min_merge_width) & (tmp_width_interval2 >= tmp_width_interval1)) {
							col_loc.right = max(col_loc.right, col_loc_list[i][k].right);
							merge_flag = true;
							j = k + 1;
						}
					}
				}

				if (!merge_flag) {
					col_loc = col_loc_list[i][j];
					j = j + 1;
				}

				tmp_list.push_back(col_loc);
				if (j == list_count) {
					tmp_list.push_back(col_loc_list[i][j]);
				}

			}
		}
		merge_word_list.push_back(tmp_list);
		tmp_list.clear();
	}

	return merge_word_list;
}


rectStruct getWordLocList(cv::Mat src_img, rectStruct col_loc, float split_height) {
	rectStruct word_loc;

	cv::Mat loc_img(src_img, cv::Rect(
		col_loc.left,
		col_loc.top,
		col_loc.right - col_loc.left,
		col_loc.down - col_loc.top));

	word_loc = connectToRect(loc_img, col_loc, split_height);

	return word_loc;
}


//vector<vector<rectStruct>> regroupWordLocList(vector<vector<rectStruct>> split_word_loc_list, float word_height) {
//	vector<rectStruct> word_loc_list, merge_word_loc_list;
//	vector<vector<rectStruct>> regroup_word_loc_list;
//	vector<vector<rectStruct>>::iterator it;
//	int row_order, tmp_interval;
//
//	if (split_word_loc_list.size() > 0) {
//		regroup_word_loc_list.resize(split_word_loc_list.size());
//
//		row_order = 0;
//		for (int l = 0; l < split_word_loc_list[0].size(); l++) {
//			split_word_loc_list[0][l].order = row_order;
//			word_loc_list.push_back(split_word_loc_list[0][l]);
//		}
//
//		for (int i = 1; i < split_word_loc_list.size(); i++) {
//			tmp_interval = abs(split_word_loc_list[i][0].left - split_word_loc_list[i - 1][split_word_loc_list[i - 1].size() - 1].right) +
//				abs(split_word_loc_list[i][0].top - split_word_loc_list[i - 1][split_word_loc_list[i - 1].size() - 1].top);
//
//			if (tmp_interval > word_height) {
//				row_order = row_order + 1;
//			}
//
//			for (int j = 0; j < split_word_loc_list[i].size(); j++) {
//				split_word_loc_list[i][j].order = row_order;
//				word_loc_list.push_back(split_word_loc_list[i][j]);
//			}
//		}
//
//		merge_word_loc_list = mergeWordByRect(word_loc_list);
//		for (int k = 0; k < merge_word_loc_list.size(); k++) {
//			regroup_word_loc_list[merge_word_loc_list[k].order].push_back(merge_word_loc_list[k]);
//		}
//	}
//	else
//	{
//		regroup_word_loc_list = split_word_loc_list;
//	}
//	
//	return regroup_word_loc_list;
//}


vector<vector<rectStruct>> regroupWordLocList(vector<vector<rectStruct>> split_word_loc_list, cv::Mat bin_img, float word_height, bool debug) {
	vector<vector<cv::Point>> contours;
	vector<cv::Vec4i> hierarcy;
	vector<vector<rectStruct>> regroup_word_loc_list;
	cv::Mat rlsa_img, rlsa_after;
	cv::Point rect_vertex[4];
	float max_area, row_index;

	rlsa_img = bin_img.clone();
	for (int x = 0; x < split_word_loc_list.size(); x++) {
		for (int y = 0; y < split_word_loc_list[x].size(); y++) {
			if (split_word_loc_list[x][y].down - split_word_loc_list[x][y].top > 2 * word_height) {
				rect_vertex[0] = cv::Point(split_word_loc_list[x][y].left, split_word_loc_list[x][y].top);
				rect_vertex[1] = cv::Point(split_word_loc_list[x][y].left, split_word_loc_list[x][y].down);
				rect_vertex[2] = cv::Point(split_word_loc_list[x][y].right, split_word_loc_list[x][y].down);
				rect_vertex[3] = cv::Point(split_word_loc_list[x][y].right, split_word_loc_list[x][y].top);
				cv::fillConvexPoly(rlsa_img, rect_vertex, 4, 0);
			}
		}
	}
	if (debug) {
		cv::imshow("bin_img", rlsa_img);
	}

	rlsa_after = rlsa_img.clone();
	RLSA::runLengthSmooth(rlsa_after, rlsa_after, 2 * word_height, 0.1 * word_height, 1.2 * word_height);
	cv::findContours(rlsa_after, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	vector<cv::RotatedRect> row_rect_list;
	cv::RotatedRect row_rect;
	float row_height;
	float min_height = 0.6 * word_height;
	float max_height = 2.4 * word_height;
	bool large_flag = false;
	for (int i = 0; i < contours.size(); i++) {
		row_rect = cv::minAreaRect(contours[i]);
		row_height = min(row_rect.size.width, row_rect.size.height);
		if (row_height > min_height) {
			row_rect_list.push_back(row_rect);
		}

		if (row_height > max_height) {
			large_flag = true;
			row_rect_list.clear();
			break;
		}
	}

	if (large_flag) {
		rlsa_after = rlsa_img.clone();
		RLSA::runLengthSmooth(rlsa_after, rlsa_after, 1.2 * word_height, 0.1 * word_height, 0.8 * word_height);
		cv::findContours(rlsa_after, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		for (int i = 0; i < contours.size(); i++) {
			row_rect = cv::minAreaRect(contours[i]);
			row_height = min(row_rect.size.width, row_rect.size.height);
			if (row_height > min_height) {
				row_rect_list.push_back(row_rect);
			}
		}

	}

	rowSortRotate(row_rect_list, word_height);

	if (debug) {
		for (int j = 0; j < row_rect_list.size(); j++) {
			cv::Point2f P[4];
			row_rect_list[j].points(P);
			for (int k = 0; k < 4; k++) {
				cv::line(rlsa_after, P[k], P[(k + 1) % 4], cv::Scalar(255), 2);
			}
			cv::putText(
				rlsa_after,
				to_string(j),
				cv::Point(P[2].x, P[2].y),
				cv::FONT_HERSHEY_SIMPLEX,
				1,
				cv::Scalar(255)
			);
		}
		cv::imshow("rlsa_img", rlsa_after);
	}

	if (row_rect_list.size() > 0) {
		regroup_word_loc_list.resize(row_rect_list.size());
		for (int l = 0; l < split_word_loc_list.size(); l++) {
			for (int m = 0; m < split_word_loc_list[l].size(); m++) {
				cv::RotatedRect word_rect(
					cv::Point2f(split_word_loc_list[l][m].left, split_word_loc_list[l][m].top),
					cv::Point2f(split_word_loc_list[l][m].left, split_word_loc_list[l][m].down),
					cv::Point2f(split_word_loc_list[l][m].right, split_word_loc_list[l][m].down));

				max_area = 0;
				row_index = -1;
				for (int n = 0; n < row_rect_list.size(); n++) {
					vector<cv::Point2f> point_list;
					cv::rotatedRectangleIntersection(word_rect, row_rect_list[n], point_list);
					if (point_list.size() > 0) {
						float intersect_area = cv::contourArea(point_list);
						if (intersect_area > max_area) {
							max_area = intersect_area;
							row_index = n;
						}
					}
				}

				if (row_index > -1) {
					split_word_loc_list[l][m].order = row_index;
					regroup_word_loc_list[row_index].push_back(split_word_loc_list[l][m]);
				}
			}
		}

		
		for (int j = 0; j < regroup_word_loc_list.size(); j++) {
			sort(regroup_word_loc_list[j].begin(), regroup_word_loc_list[j].end(), leftComp);
			/*if (regroup_word_loc_list[j].size() > 0) {
				row_rect = row_rect_list[regroup_word_loc_list[j][0].order];
				if (min(row_rect.size.width, row_rect.size.height) < max_height) {
					sort(regroup_word_loc_list[j].begin(), regroup_word_loc_list[j].end(), leftComp);
				}
				else {
					rowSort(regroup_word_loc_list[j], word_height);
				}
			}*/
		}
	}
	else {
		regroup_word_loc_list = split_word_loc_list;
	}

	return regroup_word_loc_list;
}