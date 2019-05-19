#include "imgUtils.h"


void equalizeBrightness(cv::Mat& src_img, cv::Mat& res_img, int kernelSize) {
	int channels = src_img.channels();

	cv::Mat med;
	cv::medianBlur(src_img, med, kernelSize);

	src_img.convertTo(res_img, CV_32FC(channels));
	med.convertTo(med, CV_32FC(channels));
	cv::divide(res_img, med / 128, res_img);

	res_img.convertTo(res_img, CV_8UC(channels));

}


float getAvgBrightness(cv::Mat src_img) {
	cv::Scalar scalar = cv::mean(src_img);

	return scalar.val[0];
}


float getMeanStd(cv::Mat src_img) {
	cv::Mat grad_x, grad_y, abs_grad_x, abs_grad_y, abs_grad;
	cv::Mat mat_mean, mat_variance;
	
	cv::Sobel(src_img, grad_x, CV_16S, 1, 0, 3, 1, 1, cv::BORDER_DEFAULT);
	cv::convertScaleAbs(grad_x, abs_grad_x);
	cv::Sobel(src_img, grad_y, CV_16S, 0, 1, 3, 1, 1, cv::BORDER_DEFAULT);
	cv::convertScaleAbs(grad_y, abs_grad_y);
	cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, abs_grad);

	cv::meanStdDev(abs_grad, mat_mean, mat_variance);
	
	return mat_variance.at<double>(0, 0);
}


float getWBPercent(cv::Mat src_img) {
	float wb_percent;
	int white_count = 0;
	
	for (int x = 0; x < src_img.cols; x++) {
		for (int y = 0; y < src_img.rows; y++) {
			if (src_img.at<uchar>(y, x) > 125) {
				white_count += 1;
			}
		}
	}
	wb_percent = float(white_count) / float(src_img.rows * src_img.cols);
	
	return wb_percent;
}

void blurWithMask(cv::Mat& src_img, cv::Mat& res_img, cv::Size ksize, const cv::Mat& mask) {
	cv::Mat maskedImg, sameTypeMask;
	src_img.copyTo(maskedImg, mask);

	if (src_img.type() == mask.type())
	{
		sameTypeMask = mask;
	}
	else
	{
		mask.convertTo(sameTypeMask, src_img.type());
	}

	cv::Mat blurImg, blurMask;
	cv::blur(maskedImg, blurImg, ksize);
	cv::blur(sameTypeMask, blurMask, ksize);

	res_img = blurImg / blurMask;
}


void adaptiveBinaryzation(cv::Mat& src_img, cv::Mat& res_img) {
	double min, max;
	cv::minMaxLoc(src_img, &min, &max);
	double duration = max - min;

	int kernel = 21;
	cv::Mat tmp_img;
	cv::GaussianBlur(src_img, tmp_img, cv::Size(kernel, kernel), 0);

	cv::Mat gradX, gradY;
	cv::Mat absGradX, absGradY;
	cv::Sobel(tmp_img, gradX, CV_32F, 1, 0, 3, 1, 1, cv::BORDER_DEFAULT);
	cv::convertScaleAbs(gradX, absGradX);
	cv::Sobel(tmp_img, gradY, CV_32F, 0, 1, 3, 1, 1, cv::BORDER_DEFAULT);
	cv::convertScaleAbs(gradY, absGradY);
	cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, tmp_img);

	cv::Scalar thresh = cv::mean(tmp_img);
	cv::Mat mask;
	cv::threshold(tmp_img, mask, thresh[0], 255, cv::THRESH_BINARY);

	cv::Size ksize(50, 50);

	cv::Mat meanText, meanBg;
	cv::Mat floatImg;
	src_img.convertTo(floatImg, CV_64F);
	blurWithMask(floatImg, meanText, ksize, mask / 255);
	blurWithMask(floatImg, meanBg, ksize, (~mask) / 255);

	cv::Mat mDist = cv::abs(meanText - meanBg);
	cv::Mat mThresh(mDist.rows, mDist.cols, mDist.type(), cv::Scalar::all(0));
	cv::add(mThresh, (meanText + meanBg) / 2, mThresh, mDist > 10);
	cv::add(mThresh, 10 - mDist * 5 / duration, mThresh, mDist <= 10);

	src_img.copyTo(res_img);
	res_img.setTo(0, floatImg > mThresh);
	res_img.setTo(255, floatImg <= mThresh);

}


cv::Mat binImg(cv::Mat src_img, float wb_percent, int const_c) {
	/*
	图像二值化
	*/
	cv::Mat res_img;
	if (wb_percent>0.75) {
		cv::adaptiveThreshold(src_img, res_img, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 43, const_c);
	}
	else {
		cv::threshold(src_img, res_img, 0, 255, CV_THRESH_BINARY_INV + CV_THRESH_OTSU);
	}

	return res_img;
}


cv::Mat dilateImg(cv::Mat src_img, int kernel_x = DIL_KERNEL_X, int kernel_y = DIL_KERNEL_Y) {
	/*
	图像膨胀
	*/
	cv::Mat res_img;
	cv::Mat k_element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_x, kernel_y));

	cv::dilate(src_img, res_img, k_element);

	return res_img;
}


cv::Mat erodeImg(cv::Mat src_img, int kernel_x = ERO_KERNEL_X, int kernel_y = ERO_KERNEL_Y) {
	/*
	图像腐蚀
	*/
	cv::Mat res_img;
	cv::Mat k_element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_x, kernel_y));

	cv::erode(src_img, res_img, k_element);

	return res_img;
}


double degreeTrans(double theta) {
	double res = theta / CV_PI * 180;
	return res;
}


cv::Mat rotateImg(cv::Mat src_img, double degree) {
	cv::Point2f center;
	cv::Mat M, res_img;

	center.x = float(src_img.cols / 2.0);
	center.y = float(src_img.rows / 2.0);

	M = cv::getRotationMatrix2D(center, degree, 1);
	//cv::warpAffine(src_img, res_img, M, cv::Size(src_img.cols, src_img.rows), 1, cv::BORDER_REPLICATE);
	cv::warpAffine(src_img, res_img, M, src_img.size(), 1, 0, cv::Scalar(0));

	return res_img;
}


double calcDegree(cv::Mat src_img) {
	cv::Mat dil_img, mid_img;
	vector<cv::Vec2f> lines;

	dil_img = dilateImg(src_img, 6, 1);
	cv::Canny(dil_img, mid_img, 50, 200, 3);
	cv::HoughLines(mid_img, lines, 1, CV_PI / 180, 200, 0, 0);

	if (!lines.size()) {
		cv::HoughLines(mid_img, lines, 1, CV_PI / 180, 150, 0, 0);
	}
	if (!lines.size()) {
		return 0;
	}

	float rho, theta, avg_theta, sum_theta = 0;
	double angle;
	int count_theta = 0;

	for (int i = 0; i < lines.size(); i++) {
		rho = lines[i][0];
		theta = lines[i][1];

		if ((theta > 0.8) && (theta < 2.3)) {
			sum_theta += theta;
			count_theta += 1;
		}
	}

	if (count_theta > 0) {
		avg_theta = sum_theta / count_theta;
	}
	else {
		return 0;
	}
	angle = degreeTrans(avg_theta) - 90;

	return angle;
}


void deNoise(cv::Mat& src_img, int word_height) {
	vector<vector<cv::Point>> all_points_list;
	vector<cv::Vec4i> hierarchy;
	cv::RotatedRect rotate_box;
	cv::Rect box;
	cv::Point2f rect_vertex_f[4];
	cv::Point rect_vertex[4];
	vector<int> father_frame_list;
	int repeat_time;

	float max_height = 3.5 * word_height;
	float noise_height = 0.3 * word_height;
	
	cv::Mat dil_img = dilateImg(src_img);
	//cv::Mat ero_img = erodeImg(dil_img);

	cv::findContours(dil_img, all_points_list, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	for (int i = 0; i < all_points_list.size(); i++) {
		rotate_box = cv::minAreaRect(all_points_list[i]);
		box = rotate_box.boundingRect();
		float w_h_ratio = (float)box.width / (float)box.height;
		if (((box.width < noise_height) && (box.height < noise_height)) 
			|| ((box.height < word_height) &&  w_h_ratio > MAX_WH_RATIO) 
			|| (w_h_ratio < MIN_WH_RATIO)) {
			rotate_box.points(rect_vertex_f);
			for (int j = 0; j < 4; j++)
			{
				rect_vertex[j] = rect_vertex_f[j];
			}
			cv::fillConvexPoly(src_img, rect_vertex, 4, 0);
		}
		else if ((box.height > max_height) && (hierarchy[i][3] < 2)) {
			cv::drawContours(src_img, all_points_list, i, 0, -1, 8, hierarchy, 1);
		}
		else {
			if ((hierarchy[i][3] > 0) && (hierarchy[hierarchy[i][3]][3] > 0)) {
				if (hierarchy[hierarchy[hierarchy[i][3]][3]][3] > 0) {
					if (hierarchy[hierarchy[hierarchy[hierarchy[i][3]][3]][3]][3] > 0) {
						father_frame_list.push_back(hierarchy[hierarchy[hierarchy[hierarchy[i][3]][3]][3]][3]);
					}
					else {
						father_frame_list.push_back(hierarchy[hierarchy[hierarchy[i][3]][3]][3]);
					}
				}
				else {
					father_frame_list.push_back(hierarchy[hierarchy[i][3]][3]);
				}
			}
		}
	}

	vector<int> father_frame_list_unique(father_frame_list);
	sort(father_frame_list_unique.begin(), father_frame_list_unique.end());
	vector<int>::iterator it = unique(father_frame_list_unique.begin(), father_frame_list_unique.end());
	father_frame_list_unique.erase(it, father_frame_list_unique.end());

	for (int k = 0; k < father_frame_list_unique.size(); k++) {
		repeat_time = count(father_frame_list.begin(), father_frame_list.end(), father_frame_list_unique[k]);
		if (repeat_time > 5) {
			cv::drawContours(src_img, all_points_list, father_frame_list_unique[k], 0, -1, 8, hierarchy, 1);
		}
	}
}

vector<rectStruct> getConnectDomain(cv::Mat src_img) {
	/*
	获取连通域相关信息
	*/
	cv::Mat img_label, img_state, img_centroid;
	vector<rectStruct> rect_list;
	rectStruct rect;
	int n_comps;

	n_comps = connectedComponentsWithStats(src_img, img_label, img_state, img_centroid);

	for (int i = 1; i < n_comps; i++) {
		if ((img_state.at<int>(i, cv::CC_STAT_WIDTH) > MIN_STATE_BORDER) || (img_state.at<int>(i, cv::CC_STAT_HEIGHT) > MIN_STATE_BORDER)) {
			rect.order = i;
			rect.left = img_state.at<int>(i, cv::CC_STAT_LEFT);
			rect.right = img_state.at<int>(i, cv::CC_STAT_LEFT) + img_state.at<int>(i, cv::CC_STAT_WIDTH);
			rect.top = img_state.at<int>(i, cv::CC_STAT_TOP);
			rect.down = img_state.at<int>(i, cv::CC_STAT_TOP) + img_state.at<int>(i, cv::CC_STAT_HEIGHT);
			rect_list.push_back(rect);
		}
	}

	return rect_list;
}


vector<rectStruct> splitCol(cv::Mat src_img, rectStruct loc_rect) {
	vector<rectStruct> tmp_list;
	rectStruct col_loc;
	bool last_flag, current_flag;

	col_loc.top = loc_rect.top;
	col_loc.down = loc_rect.down;

	last_flag = true;
	for (int x = 0; x < src_img.cols; x++) {
		current_flag = true;

		for (int y = 0; y < src_img.rows; y++) {
			if (src_img.at<uchar>(y, x) > 128) {
				current_flag = false;
				if (last_flag) {
					col_loc.left = loc_rect.left + x;
				}
				break;
			}
		}

		if (!last_flag) {
			if (current_flag) {
				col_loc.right = loc_rect.left + x;
				tmp_list.push_back(col_loc);

			}
			else if (x == src_img.cols - 1) {
				col_loc.right = loc_rect.right;
				tmp_list.push_back(col_loc);
			}
		}

		last_flag = current_flag;
	}

	return tmp_list;
}


vector<rectStruct> overSplitCol(cv::Mat src_img, rectStruct loc_rect) {
	vector<rectStruct> tmp_list;
	rectStruct col_loc;
	float white_count;
	bool last_flag, current_flag;

	col_loc.top = loc_rect.top;
	col_loc.down = loc_rect.down;

	last_flag = true;
	for (int x = 0; x < src_img.cols; x++) {
		white_count = 0.0;
		current_flag = true;

		for (int y = 0; y < src_img.rows; y++) {
			if (src_img.at<uchar>(y, x) > 128) {
				white_count++;
			}
		}

		if (white_count / src_img.rows > SPLIT_PERCENT) {
			current_flag = false;
			if (last_flag) {
				col_loc.left = loc_rect.left + x;
			}
		}

		if (!last_flag) {
			if (current_flag) {
				col_loc.right = loc_rect.left + x;
				tmp_list.push_back(col_loc);

			}
			else if (x == src_img.cols - 1) {
				col_loc.right = loc_rect.right;
				tmp_list.push_back(col_loc);
			}
		}

		last_flag = current_flag;
	}

	return tmp_list;
}


vector<vector<rectStruct>> getColLocList(cv::Mat src_img, vector<rectStruct> row_loc_list, int word_width) {
	/*
	根据空白列对行定位结果进行切分
	*/
	vector<vector<rectStruct>> row_col_list;
	vector<rectStruct> col_loc_list, tmp_list, tmp_list_over;
	int max_width, min_width, row_width, row_height;

	max_width = 2 * word_width;
	min_width = 0.2 * word_width;
	for (int i = 0; i < row_loc_list.size(); i++) {
		row_width = row_loc_list[i].right - row_loc_list[i].left;
		row_height = row_loc_list[i].down - row_loc_list[i].top;
		if ((row_width < min_width) || (row_height < min_width)) {
			continue;
		}

		cv::Mat con_img(src_img, cv::Rect(
			row_loc_list[i].left,
			row_loc_list[i].top,
			row_width,
			row_height));

		tmp_list = splitCol(con_img, row_loc_list[i]);
		for (int j = 0; j < tmp_list.size(); j++) {
			if (tmp_list[j].right - tmp_list[j].left > max_width) {
				cv::Mat con_img_over(src_img, cv::Rect(
					tmp_list[j].left,
					tmp_list[j].top,
					tmp_list[j].right - tmp_list[j].left,
					tmp_list[j].down - tmp_list[j].top));

				tmp_list_over = overSplitCol(con_img_over, tmp_list[j]);
				col_loc_list.insert(col_loc_list.end(), tmp_list_over.begin(), tmp_list_over.end());
			}
			else {
				col_loc_list.push_back(tmp_list[j]);
			}
		}

		row_col_list.push_back(col_loc_list);
		col_loc_list.clear();
		tmp_list.clear();
	}

	return row_col_list;
}


//rectStruct connectToRect(cv::Mat src_img, rectStruct loc, float word_height) {
//	rectStruct res;
//	cv::Mat dil_img, img_label, img_state, img_centroid;
//	int n_comps, max_area, max_area_index, word_left, word_top, word_right, word_down, tmp_left, tmp_top, tmp_right, tmp_down;
//
//	dil_img = dilateImg(src_img, 6, 6);
//	n_comps = connectedComponentsWithStats(dil_img, img_label, img_state, img_centroid);
//	if (n_comps > 1) {
//		max_area = 0;
//		max_area_index = 0;
//		for (int l = 1; l < n_comps; l++) {
//			if (img_state.at<int>(l, cv::CC_STAT_AREA) > max_area) {
//				max_area = img_state.at<int>(l, cv::CC_STAT_AREA);
//				max_area_index = l;
//			}
//		}
//
//		if ((img_state.at<int>(max_area_index, cv::CC_STAT_WIDTH)/img_state.at<int>(max_area_index, cv::CC_STAT_HEIGHT)) < 2) {
//			word_left = img_state.at<int>(max_area_index, cv::CC_STAT_LEFT);
//			word_top = img_state.at<int>(max_area_index, cv::CC_STAT_TOP);
//			word_right = img_state.at<int>(max_area_index, cv::CC_STAT_LEFT) + img_state.at<int>(max_area_index, cv::CC_STAT_WIDTH);
//			word_down = img_state.at<int>(max_area_index, cv::CC_STAT_TOP) + img_state.at<int>(max_area_index, cv::CC_STAT_HEIGHT);
//
//			cv::Mat loc_img(src_img, cv::Rect(word_left, word_top, word_right - word_left, word_down - word_top));
//			n_comps = connectedComponentsWithStats(loc_img, img_label, img_state, img_centroid);
//			if (n_comps > 1) {
//				tmp_left = img_state.at<int>(1, cv::CC_STAT_LEFT);
//				tmp_top = img_state.at<int>(1, cv::CC_STAT_TOP);
//				tmp_right = img_state.at<int>(1, cv::CC_STAT_LEFT) + img_state.at<int>(1, cv::CC_STAT_WIDTH);
//				tmp_down = img_state.at<int>(1, cv::CC_STAT_TOP) + img_state.at<int>(1, cv::CC_STAT_HEIGHT);
//				for (int i = 2; i < n_comps; i++) {
//					tmp_left = min(tmp_left, img_state.at<int>(i, cv::CC_STAT_LEFT));
//					tmp_top = min(tmp_top, img_state.at<int>(i, cv::CC_STAT_TOP));
//					tmp_right = max(tmp_right, img_state.at<int>(i, cv::CC_STAT_LEFT) + img_state.at<int>(i, cv::CC_STAT_WIDTH));
//					tmp_down = max(tmp_down, img_state.at<int>(i, cv::CC_STAT_TOP) + img_state.at<int>(i, cv::CC_STAT_HEIGHT));
//				}
//
//				res.order = 1;
//				res.left = loc.left + word_left + tmp_left;
//				res.top = loc.top + word_top + tmp_top;
//				res.right = loc.left + word_left + tmp_right;
//				res.down = loc.top + word_top + tmp_down;
//			}
//			else {
//				res.order = 0;
//			}
//
//		}
//		else
//		{
//			n_comps = connectedComponentsWithStats(src_img, img_label, img_state, img_centroid);
//			if (n_comps > 1) {
//				tmp_left = img_state.at<int>(1, cv::CC_STAT_LEFT);
//				tmp_top = img_state.at<int>(1, cv::CC_STAT_TOP);
//				tmp_right = img_state.at<int>(1, cv::CC_STAT_LEFT) + img_state.at<int>(1, cv::CC_STAT_WIDTH);
//				tmp_down = img_state.at<int>(1, cv::CC_STAT_TOP) + img_state.at<int>(1, cv::CC_STAT_HEIGHT);
//				for (int i = 2; i < n_comps; i++) {
//					tmp_left = min(tmp_left, img_state.at<int>(i, cv::CC_STAT_LEFT));
//					tmp_top = min(tmp_top, img_state.at<int>(i, cv::CC_STAT_TOP));
//					tmp_right = max(tmp_right, img_state.at<int>(i, cv::CC_STAT_LEFT) + img_state.at<int>(i, cv::CC_STAT_WIDTH));
//					tmp_down = max(tmp_down, img_state.at<int>(i, cv::CC_STAT_TOP) + img_state.at<int>(i, cv::CC_STAT_HEIGHT));
//				}
//
//				res.order = 1;
//				res.left = loc.left + tmp_left;
//				res.top = loc.top + tmp_top;
//				res.right = loc.left + tmp_right;
//				res.down = loc.top + tmp_down;
//			}
//			else {
//				res.order = 0;
//			}
//		}
//		
//	}
//	else {
//		res.order = 0;
//	}
//
//	return res;
//}


rectStruct connectToRect(cv::Mat src_img, rectStruct loc, float word_height) {
	cv::Mat img_label, img_state, img_centroid;
	int n_comps, tmp_left, tmp_top, tmp_right, tmp_down, tmp_interval, word_left, word_top, word_right, word_down;
	double min_height, max_height, comp_height, comp_width;
	rectStruct res, comp_distance;
	vector<rectStruct> comp_distance_list;

	n_comps = connectedComponentsWithStats(src_img, img_label, img_state, img_centroid);

	if (n_comps > 1) {	
		min_height = word_height;
		max_height = 1.2 * word_height;
		int mid_site = 0.5 * src_img.rows;

		for (int i = 1; i < n_comps; i++) {
			comp_distance.order = i;
			comp_height = img_state.at<int>(i, cv::CC_STAT_HEIGHT);
			comp_width = img_state.at<int>(i, cv::CC_STAT_WIDTH);
			comp_distance.left = abs(img_centroid.at<double>(i, 1) - mid_site);
			comp_distance_list.push_back(comp_distance);	
		}

		if (comp_distance_list.size() > 0) {
			sort(comp_distance_list.begin(), comp_distance_list.end(), leftComp);

			word_left = img_state.at<int>(comp_distance_list[0].order, cv::CC_STAT_LEFT);
			word_top = img_state.at<int>(comp_distance_list[0].order, cv::CC_STAT_TOP);
			word_right = img_state.at<int>(comp_distance_list[0].order, cv::CC_STAT_LEFT) + img_state.at<int>(comp_distance_list[0].order, cv::CC_STAT_WIDTH);
			word_down = img_state.at<int>(comp_distance_list[0].order, cv::CC_STAT_TOP) + img_state.at<int>(comp_distance_list[0].order, cv::CC_STAT_HEIGHT);
			for (int j = 0; j < comp_distance_list.size(); j++) {
				if (word_top < img_state.at<int>(comp_distance_list[j].order, cv::CC_STAT_TOP)) {
					tmp_interval = img_state.at<int>(comp_distance_list[j].order, cv::CC_STAT_TOP) - word_down;
				}
				else {
					tmp_interval = word_top - img_state.at<int>(comp_distance_list[j].order, cv::CC_STAT_TOP) - img_state.at<int>(comp_distance_list[j].order, cv::CC_STAT_HEIGHT);
				}

				if (tmp_interval < min_height) {
					tmp_left = min(word_left, img_state.at<int>(comp_distance_list[j].order, cv::CC_STAT_LEFT));
					tmp_top = min(word_top, img_state.at<int>(comp_distance_list[j].order, cv::CC_STAT_TOP));
					tmp_right = max(word_right, img_state.at<int>(comp_distance_list[j].order, cv::CC_STAT_LEFT) + img_state.at<int>(comp_distance_list[j].order, cv::CC_STAT_WIDTH));
					tmp_down = max(word_down, img_state.at<int>(comp_distance_list[j].order, cv::CC_STAT_TOP) + img_state.at<int>(comp_distance_list[j].order, cv::CC_STAT_HEIGHT));

					if ((tmp_down - tmp_top) > max_height) {
						break;
					}
					else {
						word_left = tmp_left;
						word_top = tmp_top;
						word_right = tmp_right;
						word_down = tmp_down;
					}
				}
			}
			res.order = 1;
			res.left = loc.left + word_left;
			res.top = loc.top + word_top;
			res.right = loc.left + word_right;
			res.down = loc.top + word_down;
		}
		else {
			res.order = 0;
		}
	}
	else {
		res.order = 0;
	}

	return res;
}


vector<cv::Mat> getImgList(cv::Mat src_img, vector<rectStruct> word_loc_list) {
	vector<cv::Mat> img_list;
	rectStruct word_loc;

	for (int i = 0; i < word_loc_list.size(); i++) {
		word_loc = word_loc_list[i];
		cv::Mat word_img(src_img, cv::Rect(
			word_loc.left,
			word_loc.top,
			word_loc.right - word_loc.left,
			word_loc.down - word_loc.top));

		word_img.convertTo(word_img, CV_32FC1);
		resize(word_img, word_img, cv::Size(IMG_HEIGHT, IMG_WIDTH), 0, 0, cv::INTER_LINEAR);
		word_img = word_img / 255.0;

		img_list.push_back(word_img);
	}

	return img_list;
}


vector<cv::Mat> getImgListAll(cv::Mat src_img, vector<vector<rectStruct>> word_loc_list) {
	vector<cv::Mat> img_list;
	rectStruct word_loc;

	for (int i = 0; i < word_loc_list.size(); i++) {
		for (int j = 0; j < word_loc_list[i].size(); j++) {
			word_loc = word_loc_list[i][j];
			cv::Mat word_img(src_img, cv::Rect(
				word_loc.left,
				word_loc.top,
				word_loc.right - word_loc.left,
				word_loc.down - word_loc.top));

			word_img.convertTo(word_img, CV_32FC1);
			resize(word_img, word_img, cv::Size(IMG_HEIGHT, IMG_WIDTH), 0, 0, cv::INTER_LINEAR);
			word_img = word_img / 255.0;

			img_list.push_back(word_img);
		}
	}

	return img_list;
}


void showSetList(cv::Mat src_img, vector<vector<rectStruct>> set_list) {
	for (int i = 0; i < set_list.size(); i++) {
		for (int j = 0; j < set_list[i].size(); j++) {
			rectStruct set_rect = set_list[i][j];

			cv::rectangle(
				src_img,
				cv::Point(set_rect.left, set_rect.top),
				cv::Point(set_rect.right, set_rect.down),
				cv::Scalar(0, 0, 255)
			);
			cv::putText(
				src_img,
				to_string(i),
				cv::Point(set_rect.left, set_rect.top),
				cv::FONT_HERSHEY_SIMPLEX,
				1,
				cv::Scalar(255, 255, 0)
			);
		}
	}

	cv::imshow("set_loc", src_img);
	cv::imwrite("(0)set_loc.jpg", src_img);
}


void showRowLoc(cv::Mat src_img, vector<rectStruct> row_loc_list, string file_name) {
	rectStruct row_loc;

	for (int i = 0; i < row_loc_list.size(); i++) {
		row_loc = row_loc_list[i];
		cv::rectangle(
			src_img,
			cv::Point(row_loc.left, row_loc.top),
			cv::Point(row_loc.right, row_loc.down),
			cv::Scalar(0, 0, 255)
		);
		cv::putText(
			src_img,
			to_string(i),
			cv::Point(row_loc.left, row_loc.top),
			cv::FONT_HERSHEY_SIMPLEX,
			1,
			cv::Scalar(255, 255, 0)
		);
	}

	cv::imshow(file_name, src_img);
	cv::imwrite(file_name + ".jpg", src_img);
}


void showColLoc(cv::Mat src_img, vector<vector<rectStruct>> col_loc_list, string file_name) {
	rectStruct col_loc;

	for (int i = 0; i < col_loc_list.size(); i++) {
		for (int j = 0; j < col_loc_list[i].size(); j++) {
			col_loc = col_loc_list[i][j];
			cv::rectangle(
				src_img,
				cv::Point(col_loc.left, col_loc.top),
				cv::Point(col_loc.right, col_loc.down),
				cv::Scalar(0, 0, 255)
			);
			cv::putText(
				src_img,
				to_string(i),
				cv::Point(col_loc.left, col_loc.top),
				cv::FONT_HERSHEY_SIMPLEX,
				1,
				cv::Scalar(255, 255, 0)
			);
		}
	}

	cv::imshow(file_name, src_img);
	cv::imwrite(file_name + ".jpg", src_img);
}


inline void rangeSet(cv::Mat& src, int rowStart, int rowEnd, int colStart, int colEnd, uchar pxValue)
{
	int channels = src.channels();

	uchar* p;
	for (int i = rowStart; i < rowEnd; ++i)
	{
		p = src.ptr<uchar>(i);
		for (int j = colStart; j < colEnd; ++j)
		{
			p[j] = pxValue;
		}
	}
}

void RLSA::horizontalRunLengthSmooth(cv::Mat& src, cv::Mat& dst, int hthreshold, uchar whitePxValue)
{

	for (int i = 0; i < src.rows; ++i)
	{
		int zeroCount = 0;
		for (int j = 0; j < src.cols; ++j)
		{
			if (src.at<uchar>(i, j) == whitePxValue)
			{
				if (zeroCount < hthreshold)
				{
					rangeSet(dst, i, i + 1, j - zeroCount, j + 1, whitePxValue);
				}
				else
				{
					dst.at<uchar>(i, j) = whitePxValue;
				}

				zeroCount = 0;
			}
			else
			{
				++zeroCount;
			}
		}
	}

}

void RLSA::verticalRunLengthSmooth(cv::Mat& src, cv::Mat& dst, int threshold, uchar whitePxValue)
{

	for (int i = 0; i < src.cols; ++i)
	{
		int zeroCount = 0;
		for (int j = 0; j < src.rows; ++j)
		{
			if (src.at<uchar>(j, i) == whitePxValue)
			{
				if (zeroCount < threshold)
				{
					rangeSet(dst, j - zeroCount, j + 1, i, i + 1, whitePxValue);
				}
				else
				{
					dst.at<uchar>(j, i) = whitePxValue;
				}

				zeroCount = 0;
			}
			else
			{
				++zeroCount;
			}
		}
	}
}


void RLSA::runLengthSmooth(cv::Mat& src, cv::Mat& dst, int hsv, int vsv, int asv, uchar whitePxValue)
{
	cv::Mat hrls(src.rows, src.cols, CV_8U, cv::Scalar::all(0));
	cv::Mat vrls = hrls.clone();

	horizontalRunLengthSmooth(src, hrls, hsv, whitePxValue);

	verticalRunLengthSmooth(src, vrls, vsv, whitePxValue);

	cv::Mat andRls = hrls & vrls;

	horizontalRunLengthSmooth(andRls, dst, asv, whitePxValue);
}
