#include "toolUtils.h"

float whiteBGPercent(cv::Mat src_img) {
	int counter = 0;
	for (int x = 0; x < src_img.cols; x++) {
		for (int y = 0; y < src_img.rows; y++) {
			if (src_img.at<uchar>(y, x) > 10) {
				counter += 1;
			}
		}
	}

	return float(counter) / float(src_img.rows * src_img.cols);
}


void getTrainDataForP2P() {
	int count = 0;
	string in_path_str = "imgClear\\";
	string out_path_str = "bin2gray\\";

	intptr_t handle;
	_finddata_t findData;

	handle = _findfirst((in_path_str + "*.jpg").c_str(), &findData);
	if (handle == -1) {
		cout << "file not found" << endl;
	}

	do {
		cv::Mat gray_img = cv::imread((in_path_str + findData.name).c_str(), cv::IMREAD_GRAYSCALE);
		cv::Mat bin_img;
		float degree;
		vector<vector<rectStruct>> loc_list = getSingleWordRect(gray_img, bin_img, degree);
		vector<rectStruct> word_loc_list;
		rectStruct word_loc;

		int mid_width, min_width, max_width, tmp_width;

		for (int i = 0; i < loc_list.size(); i++) {
			for (int j = 0; j < loc_list[i].size(); j++) {
				word_loc_list.push_back(loc_list[i][j]);
			}

		}

		if (word_loc_list.size() > 2) {
			mid_width = quantileHeight(word_loc_list, QUANT_PERCEENT);
			max_width = 1.2*mid_width;
			min_width = 0.8*mid_width;

			for (int k = 0; k < word_loc_list.size(); k++) {
				word_loc = word_loc_list[k];
				tmp_width = word_loc.right - word_loc.left;
				if ((tmp_width > min_width) && (tmp_width < max_width)) {
					cv::Mat word_gray_img(gray_img, cv::Rect(
						word_loc.left,
						word_loc.top,
						word_loc.right - word_loc.left,
						word_loc.down - word_loc.top));

					cv::Mat word_bin_img(bin_img, cv::Rect(
						word_loc.left,
						word_loc.top,
						word_loc.right - word_loc.left,
						word_loc.down - word_loc.top));

					if (whiteBGPercent(word_bin_img) < 0.5) {
						resize(word_gray_img, word_gray_img, cv::Size(IMG_HEIGHT, IMG_WIDTH), 0, 0, cv::INTER_LINEAR);
						resize(word_bin_img, word_bin_img, cv::Size(IMG_HEIGHT, IMG_WIDTH), 0, 0, cv::INTER_LINEAR);
						cv::Mat tmp[] = { word_gray_img, word_bin_img };
						cv::Mat res;
						cv::hconcat(tmp, 2, res);

						cv::imwrite((out_path_str + to_string(count) + "_" + to_string(k) + ".jpg").c_str(), res);
					}
				}
			}
		}

		cout << count << endl;
		count++;
	} while (_findnext(handle, &findData) == 0);
}


void getOcrImg() {
	rectStruct col_loc;
	intptr_t handle;
	_finddata_t findData;

	string in_path_str = "test7\\";
	string out_path_str = "ocr7\\";

	handle = _findfirst((in_path_str + "*.bmp").c_str(), &findData);
	if (handle == -1) {
		cout << "ÎÄ¼þ¼ÐÎª¿Õ\n" << endl;
	}

	do {
		cout << findData.name << endl;
		cv::Mat gray_img = cv::imread((in_path_str + findData.name).c_str(), cv::IMREAD_GRAYSCALE);
		cv::Mat org_img = cv::imread((in_path_str + findData.name).c_str());
		cv::Mat bin_img;
		float degree;
		vector<vector<rectStruct>> col_loc_list = getSingleWordRect(gray_img, bin_img, degree);

		if (col_loc_list.size() > 0) {
			for (int i = 0; i < col_loc_list.size(); i++) {
				for (int j = 0; j < col_loc_list[i].size(); j++) {
					col_loc = col_loc_list[i][j];
					rectangle(
						gray_img,
						cv::Point(col_loc.left, col_loc.top),
						cv::Point(col_loc.right, col_loc.down),
						cv::Scalar(0, 0, 255)
					);
					putText(
						gray_img,
						to_string(i),
						cv::Point(col_loc.left, col_loc.top),
						cv::FONT_HERSHEY_SIMPLEX,
						1,
						cv::Scalar(255, 255, 0)
					);
				}
			}

			imwrite((out_path_str + findData.name).c_str(), gray_img);
		}

	} while (_findnext(handle, &findData) == 0);

	cin.get();
}


void getCharBinImg() {
	string in_path_str = "train\\";
	string out_path_str = "train\\";

	intptr_t handle;
	_finddata_t findData;

	intptr_t charHandle;
	_finddata_t charFindData;

	handle = _findfirst((in_path_str + "*").c_str(), &findData);
	if (handle == -1) {
		cout << "file not found" << endl;
	}

	do {

		if (strcmp(findData.name, ".") != 0 && strcmp(findData.name, "..") != 0) {
			_mkdir((out_path_str + findData.name).c_str());
			charHandle = _findfirst((in_path_str + findData.name + "\\*").c_str(), &charFindData);

			do {

				if (strcmp(charFindData.name, ".") != 0 && strcmp(charFindData.name, "..") != 0) {
					cout << in_path_str + findData.name + "\\" + charFindData.name << endl;
					cv::Mat gray_img = cv::imread((in_path_str + findData.name + "\\" + charFindData.name).c_str(), cv::IMREAD_GRAYSCALE);
					cv::Mat bin_img;
					cv::threshold(gray_img, bin_img, 0, 255, CV_THRESH_BINARY_INV + CV_THRESH_OTSU);

					cv::resize(gray_img, gray_img, cv::Size(IMG_HEIGHT, IMG_WIDTH), 0, 0, cv::INTER_LINEAR);
					cv::resize(bin_img, bin_img, cv::Size(IMG_HEIGHT, IMG_WIDTH), 0, 0, cv::INTER_LINEAR);
					cv::Mat tmp[] = { gray_img, bin_img };
					cv::Mat res;
					cv::hconcat(tmp, 2, res);

					cv::imwrite((out_path_str + findData.name + "\\" + charFindData.name), res);
				}
			} while (_findnext(charHandle, &charFindData) == 0);


			cout << findData.name << endl;
		}

	} while (_findnext(handle, &findData) == 0);
}


vector<string> getCharImg(ChineseRecognize* recognizer, string img_url, string question_id, string char_dir) {
	vector<vector<rectStruct>> res_list;
	vector<string> img_name_list;
	cv::Mat gray_img;
	string char_img_path, char_img_name;

	gray_img = cv::imread(img_url, cv::IMREAD_GRAYSCALE);
	res_list = doubleRecognizeWithPos(gray_img, recognizer);
	for (int i = 0; i < res_list.size(); i++) {
		for (int j = 0; j < res_list[i].size(); j++) {
			cv::Mat word_img(gray_img, cv::Rect(
				res_list[i][j].left,
				res_list[i][j].top,
				res_list[i][j].right - res_list[i][j].left,
				res_list[i][j].down - res_list[i][j].top));

			char_img_path = char_dir + "\\" + to_string(res_list[i][j].order);
			CreateDirectory(char_img_path.c_str(), NULL);

			char_img_name = question_id + "_" + to_string(i) + to_string(j) + ".png";
			cv::imwrite(char_img_path + "\\" + char_img_name, word_img);

			img_name_list.push_back(char_img_name);
		}
	}

	return img_name_list;
}


void autoTest() {
	cv::Mat gray_img, bin_img;
	vector<vector<int>> results;

	string in_path_str = "test\\";
	string out_path_str = "res\\";

	intptr_t imgHandle;
	_finddata_t imgFindData;


	ChineseRecognize* recognizer = new ChineseRecognize("cnftl6_cnn.pd", IMG_HEIGHT, IMG_WIDTH, 1, "Placeholder:0", "Softmax:0");

	imgHandle = _findfirst((in_path_str + "*").c_str(), &imgFindData);
	do {
		if (strcmp(imgFindData.name, ".") != 0 && strcmp(imgFindData.name, "..") != 0) {
			gray_img = cv::imread((in_path_str + imgFindData.name).c_str(), cv::IMREAD_GRAYSCALE);
			results = doubleRecognize(gray_img, recognizer);

			ofstream ofile;
			ofile.open(out_path_str + imgFindData.name + ".txt");

			for (int m = 0; m < results.size(); m++) {
				for (int n = 0; n < results[m].size(); n++) {
					ofile << results[m][n] << " ";
				}

				ofile << endl;
			}

			ofile.close();

			cout << imgFindData.name << endl;
		}
	} while (_findnext(imgHandle, &imgFindData) == 0);
}
