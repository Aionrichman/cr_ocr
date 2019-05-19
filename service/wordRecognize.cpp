#include "wordRecognize.h"


bool splitWord(int order) {
	if (order < 0 || order == 76 || order == 84 || order == 4 || order == 35) {
		return true;
	}
	else {
		return false;
	}
}


bool ignoreWord(int order) {
	if (((order > 11) && (order < 15)) || ((order > 57) && (order < 64)) || ((order > 89) && (order < 95))) {
		return true;
	}
	else {
		return false;
	}
}


vector<rectStruct> addSpaceForENG(vector<rectStruct> no_space_list) {
	vector<rectStruct> space_list;
	rectStruct space;
	int gap_width, ave_width;
	float eng_count;
	
	eng_count = 0.0;
	ave_width = 0;
	
	for (int i = 0; i < no_space_list.size(); i++) {
		if (no_space_list[i].order > 31 && no_space_list[i].order < 90) {
			eng_count++;
		}
		ave_width += (no_space_list[i].right - no_space_list[i].left);
	}

	ave_width = 0.4 * ave_width / no_space_list.size();
	if (eng_count/ no_space_list.size() > 0.5) {
		for (int j = 0; j < no_space_list.size() - 1; j++) {
			if (no_space_list[j].order == 15) {
				no_space_list[j].order = 78;
			}
			if (no_space_list[j].order == 16) {
				no_space_list[j].order = 72;
			}

			space_list.push_back(no_space_list[j]);

			gap_width = no_space_list[j + 1].left - no_space_list[j].right;
			if (gap_width > ave_width) {
				space.order = WORD_NUM;
				space.left = no_space_list[j].right;
				space.top = no_space_list[j].top;
				space.down = no_space_list[j].down;
				space.right = no_space_list[j + 1].left;
				space_list.push_back(space);
			}
		}

		space_list.push_back(no_space_list[no_space_list.size() - 1]);
		space.order = WORD_NUM;
		space.left = no_space_list[no_space_list.size() - 1].left;
		space.top = no_space_list[no_space_list.size() - 1].top;
		space.down = no_space_list[no_space_list.size() - 1].down;
		space.right = no_space_list[no_space_list.size() - 1].left + 1;
		space_list.push_back(space);
	}
	else {
		space_list = no_space_list;
	}

	return space_list;
}


vector<vector<rectStruct>> addSpaceForRes(vector<vector<rectStruct>> res_list) {
	vector<vector<rectStruct>> res_list_plus_space;
	vector<rectStruct> space_list;
	for (int i = 0; i < res_list.size(); i++) {
		space_list = addSpaceForENG(res_list[i]);
		res_list_plus_space.push_back(space_list);
	}

	return res_list_plus_space;
}


void rotateRects(vector<vector<rectStruct>>& rectsLines, const cv::Point2f& center, double degree) {
	cv::Mat rotateMatrix = cv::getRotationMatrix2D(center, degree, 1);
	vector<cv::Point3d> pts;
	for (auto& rectLine : rectsLines) {
		for (auto& rect : rectLine) {
			pts.push_back(cv::Point3d(rect.left, rect.top, 1));
			pts.push_back(cv::Point3d(rect.right, rect.down, 1));
		}
	}

	cv::Mat ptMat(pts.size(), 3, CV_64F, pts.data());
	cv::Mat processedMat;
	processedMat = rotateMatrix * ptMat.t();
	int counter = 0;
	for (int i = 0; i < rectsLines.size(); ++i) {
		for (int j = 0; j < rectsLines[i].size(); ++j) {
			rectStruct& oneRect = rectsLines[i][j];
			oneRect.left = (int) processedMat.at<double>(0, 2 * counter);
			oneRect.top = (int)processedMat.at<double>(1, 2 * counter);
			oneRect.right = (int)processedMat.at<double>(0, 2 * counter + 1);
			oneRect.down = (int)processedMat.at<double>(1, 2 * counter + 1);
   
			counter += 1;
		}
	}
}


vector<vector<int>> doubleRecognize(cv::Mat& src_img, ChineseRecognize* recognizer) {
	vector<cv::Mat> merge_img_list, split_img_list;
	vector<vector<rectStruct>> word_loc_list, split_col_list;
	vector<rectStruct> unmatch_word_list, split_loc_list, split_col;
	vector<int> merge_results, split_results, split_result, final_results;
	vector<vector<int>> split_results_list, final_results_list;
	bool split_flag;
	int merge_order, split_order;
	float degree;
	cv::Mat bin_img;

	double start, end;
	start = clock();

	// 获取定位框
	word_loc_list = getSingleWordRect(src_img, bin_img, degree);
	
	if (word_loc_list.size() > 0) {
		// 获取识别结果
		merge_img_list = getImgListAll(src_img, word_loc_list);
	
		Tensor input_tensor(tensorflow::DT_FLOAT, TensorShape({ (long long)merge_img_list.size(), IMG_HEIGHT, IMG_WIDTH, 1 }));
		convertMatsToTensors(merge_img_list, input_tensor);
		merge_results = recognizer->recognizeChinese(merge_img_list);

		// 过切分无法识别的定位框
		merge_order = 0;
		for (int i = 0; i < word_loc_list.size(); i++) {
			for (int j = 0; j < word_loc_list[i].size(); j++) {
				if (splitWord(merge_results[merge_order])) {
					split_loc_list.push_back(word_loc_list[i][j]);
				}

				merge_order++;
			}
		}

		for (int k = 0; k < split_loc_list.size(); k++) {
			cv::Mat word_img(bin_img, cv::Rect(
				split_loc_list[k].left,
				split_loc_list[k].top,
				split_loc_list[k].right - split_loc_list[k].left,
				split_loc_list[k].down - split_loc_list[k].top));

			split_col = overSplitCol(word_img, split_loc_list[k]); 

			split_col_list.push_back(split_col);
		}

		if (split_col_list.size() > 0) {
			// 获取过切分识别结果
			split_img_list = getImgListAll(src_img, split_col_list);

			Tensor split_tensor(tensorflow::DT_FLOAT, TensorShape({ (long long)split_img_list.size(), IMG_HEIGHT, IMG_WIDTH, 1 }));
			convertMatsToTensors(split_img_list, split_tensor);
			split_results = recognizer->recognizeChinese(split_img_list, PREDICT_CONFIDENT_LOW);

			// 按照定位结构整合过切分识别结果
			split_order = 0;
			for (int l = 0; l < split_col_list.size(); l++) {
				for (int n = 0; n < split_col_list[l].size(); n++) {
					if (ignoreWord(split_results[split_order])) {
						split_results[split_order] = -1;
					}

					split_result.push_back(split_results[split_order]);
					split_order++;
				}

				split_results_list.push_back(split_result);
				split_result.clear();
			}

			// 按照定位结构整合所有识别结果
			merge_order = 0;
			split_order = 0;
			for (int x = 0; x < word_loc_list.size(); x++) {
				for (int y = 0; y < word_loc_list[x].size(); y++) {
					if (splitWord(merge_results[merge_order])) {
						split_flag = false;

						for (int z = 0; z < split_results_list[split_order].size(); z++) {
							if (split_results_list[split_order][z] > -1) {
								split_flag = true;
								break;
							}
						}

						if (split_flag) {
							for (int z = 0; z < split_results_list[split_order].size(); z++) {
								final_results.push_back(split_results_list[split_order][z]);
							}
						}
						else {
							if (ignoreWord(merge_results[merge_order])) {
								merge_results[merge_order] = -1;
							}
							final_results.push_back(merge_results[merge_order]);
						}

						split_order++;
					}
					else {
						if (ignoreWord(merge_results[merge_order])) {
							merge_results[merge_order] = -1;
						}
						final_results.push_back(merge_results[merge_order]);
					}

					merge_order++;
				}

				final_results_list.push_back(final_results);
				final_results.clear();
			}
		}
		else {
			merge_order = 0;
			for (int x = 0; x < word_loc_list.size(); x++) {
				for (int y = 0; y < word_loc_list[x].size(); y++) {
					if (ignoreWord(merge_results[merge_order])) {
						merge_results[merge_order] = -1;
					}

					final_results.push_back(merge_results[merge_order]);
					merge_order++;
				}

				final_results_list.push_back(final_results);
				final_results.clear();
			}
		}
	}

	end = clock();
	cout << "r_cost:" << (end - start) / CLOCKS_PER_SEC << endl;


	return final_results_list;
}


vector<vector<rectStruct>> doubleRecognizeWithPos(cv::Mat& src_img, ChineseRecognize* recognizer) {
	vector<cv::Mat> merge_img_list, split_img_list;
	vector<rectStruct> split_loc_list, split_col, split_results, final_results;
	vector<vector<rectStruct>> word_loc_list, split_col_list;
	vector<int> merge_re_results, split_re_results;
	vector<vector<rectStruct>> split_results_list, final_results_list, final_results_list_plus_space;
	bool split_flag;
	int merge_order, split_order;
	float degree;
	cv::Mat bin_img;
	rectStruct split_result, final_result;

	double start, end;
	start = clock();


	// 获取定位框
	word_loc_list = getSingleWordRect(src_img, bin_img, degree);

	if (word_loc_list.size() > 0) {
		// 获取识别结果
		merge_img_list = getImgListAll(src_img, word_loc_list);

		Tensor input_tensor(tensorflow::DT_FLOAT, TensorShape({ (long long)merge_img_list.size(), IMG_HEIGHT, IMG_WIDTH, 1 }));
		convertMatsToTensors(merge_img_list, input_tensor);
		merge_re_results = recognizer->recognizeChinese(merge_img_list);

		// 过切分无法识别的定位框
		merge_order = 0;
		for (int i = 0; i < word_loc_list.size(); i++) {
			for (int j = 0; j < word_loc_list[i].size(); j++) {
				if (splitWord(merge_re_results[merge_order])) {
					split_loc_list.push_back(word_loc_list[i][j]);
				}

				merge_order++;
			}
		}

		for (int k = 0; k < split_loc_list.size(); k++) {
			cv::Mat word_img(bin_img, cv::Rect(
				split_loc_list[k].left,
				split_loc_list[k].top,
				split_loc_list[k].right - split_loc_list[k].left,
				split_loc_list[k].down - split_loc_list[k].top));

			split_col = splitCol(word_img, split_loc_list[k]);

			split_col_list.push_back(split_col);
		}

		if (split_col_list.size() > 0) {
			// 获取过切分识别结果
			split_img_list = getImgListAll(src_img, split_col_list);

			Tensor split_tensor(tensorflow::DT_FLOAT, TensorShape({ (long long)split_img_list.size(), IMG_HEIGHT, IMG_WIDTH, 1 }));
			convertMatsToTensors(split_img_list, split_tensor);
			split_re_results = recognizer->recognizeChinese(split_img_list, PREDICT_CONFIDENT_LOW);

			// 按照定位结构整合过切分识别结果
			split_order = 0;
			for (int l = 0; l < split_col_list.size(); l++) {
				for (int n = 0; n < split_col_list[l].size(); n++) {
					if (ignoreWord(split_re_results[split_order])) {
						split_re_results[split_order] = -1;
					}

					split_result = split_col_list[l][n];
					split_result.order = split_re_results[split_order];
					split_results.push_back(split_result);
					split_order++;
				}

				split_results_list.push_back(split_results);
				split_results.clear();
			}

			// 按照定位结构整合所有识别结果
			merge_order = 0;
			split_order = 0;
			for (int x = 0; x < word_loc_list.size(); x++) {
				for (int y = 0; y < word_loc_list[x].size(); y++) {
					if (splitWord(merge_re_results[merge_order])) {
						split_flag = false;

						for (int z = 0; z < split_results_list[split_order].size(); z++) {
							if (split_results_list[split_order][z].order > -1) {
								split_flag = true;
								break;
							}
						}

						if (split_flag) {
							for (int z = 0; z < split_results_list[split_order].size(); z++) {
								final_results.push_back(split_results_list[split_order][z]);
							}
						}
						else {
							if (ignoreWord(merge_re_results[merge_order])) {
								merge_re_results[merge_order] = -1;
							}

							final_result = word_loc_list[x][y];
							final_result.order = merge_re_results[merge_order];
							final_results.push_back(final_result);
						}

						split_order++;
					}
					else {
						if (ignoreWord(merge_re_results[merge_order])) {
							merge_re_results[merge_order] = -1;
						}
						
						final_result = word_loc_list[x][y];
						final_result.order = merge_re_results[merge_order];
						final_results.push_back(final_result);
					}

					merge_order++;
				}

				final_results_list.push_back(final_results);
				final_results.clear();
			}
		}
		else {
			merge_order = 0;
			for (int x = 0; x < word_loc_list.size(); x++) {
				for (int y = 0; y < word_loc_list[x].size(); y++) {
					if (ignoreWord(merge_re_results[merge_order])) {
						merge_re_results[merge_order] = -1;
					}

					final_result = word_loc_list[x][y];
					final_result.order = merge_re_results[merge_order];
					final_results.push_back(final_result);
					merge_order++;
				}

				final_results_list.push_back(final_results);
				final_results.clear();
			}
		}
	}

	end = clock();
	cout << "r_cost:" << (end - start) / CLOCKS_PER_SEC << endl;

	final_results_list_plus_space = addSpaceForRes(final_results_list);

	if (abs(degree) > 0) {
		cv::Point2f center;
		center.x = float(src_img.cols / 2.0);
		center.y = float(src_img.rows / 2.0);
		rotateRects(final_results_list_plus_space, center, -degree);
	}

	return final_results_list_plus_space;
}


