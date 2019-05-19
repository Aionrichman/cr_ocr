#ifndef _RE_MODEL_H_
#define _RE_MODEL_H_

#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

#include "constant.h"

using namespace tensorflow;


void convertMatsToTensors(vector<cv::Mat>& src, Tensor& dst);


class ChineseRecognize
{
	Session* session;
	const string input_tensor_name;
	const string output_tensor_name;
	int img_height;
	int img_width;
	int img_channel;

	void loadModel(const string& model_path);

public:
	ChineseRecognize(const string& model_path, int img_height, int img_width, int channel, const string& input_tensor_name, const string& output_tensor_name);
	~ChineseRecognize();

	vector<int> recognizeChinese(vector<cv::Mat> img_list, float predict_confident = PREDICT_CONFIDENT);

};


#endif // !_RE_MODEL_H_