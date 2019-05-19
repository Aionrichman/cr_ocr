#include "reModel.h"


void copyMatToTensor(cv::Mat& src, Eigen::TensorMap<Eigen::Tensor<float, 4, 1, Eigen::DenseIndex>, 16, Eigen::MakePointer>& dst, int batch_index)
{
	int channels = src.channels();
	int rows = src.rows;
	int cols = src.cols * channels;

	int target_channel = 0;

	float* p;
	for (int y = 0; y < rows; ++y)
	{
		p = src.ptr<float>(y);
		for (int x = 0; x < cols; ++x)
		{
			dst(batch_index, y, x, target_channel) = p[x];
			target_channel = (target_channel + 1) % channels;
		}
	}
}


void convertMatsToTensors(vector<cv::Mat>& src, Tensor& dst) {
	auto input_tensor_list = dst.tensor<float, 4>();

	for (int i = 0; i < src.size(); i++) {
		copyMatToTensor(src[i], input_tensor_list, i);
	}
}


ChineseRecognize::ChineseRecognize(const string& model_path, int img_height, int img_width, int img_channel, const string& input_tensor_name, const string& output_tensor_name)
	:input_tensor_name(input_tensor_name), output_tensor_name(output_tensor_name)
{
	this->img_height = img_height;
	this->img_width = img_width;
	this->img_channel = img_channel;

	//tensorflow::SessionOptions session_options;
	//session_options.config.mutable_gpu_options()->set_allow_growth(false);
	//session_options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);

	Status status = NewSession(tensorflow::SessionOptions(), &session);
	if (!status.ok()) {
		string error_msg = status.ToString();
		throw runtime_error(error_msg.c_str());
	}

	this->loadModel(model_path);
}


ChineseRecognize::~ChineseRecognize() {
	this->session->Close();
}


void ChineseRecognize::loadModel(const string& model_path) {
	GraphDef graph_def;

	Status status = ReadBinaryProto(Env::Default(), model_path, &graph_def);
	if (!status.ok()) {
		string error_msg = status.ToString() + "Can't read model from " + model_path;
		throw runtime_error(error_msg.c_str());
	}

	status = session->Create(graph_def);
	if (!status.ok()) {
		string error_msg = status.ToString() + "Can't load graphDef from session";
		throw runtime_error(error_msg.c_str());
	}
}


vector<int>  ChineseRecognize::recognizeChinese(vector<cv::Mat> img_list, float predict_confident) {
	Tensor input_tensor(tensorflow::DT_FLOAT, TensorShape({ (ptrdiff_t)img_list.size(), this->img_height, this->img_width, this->img_channel }));
	convertMatsToTensors(img_list, input_tensor);

	vector<pair<string, Tensor>> inputs = { {this->input_tensor_name, input_tensor} };
	vector<Tensor> outputs;
	Status status = this->session->Run(inputs, { this->output_tensor_name }, {}, &outputs);
	if (!status.ok()) {
		string error_msg = status.ToString() + "Can't run session from model";
		throw runtime_error(error_msg.c_str());
	}

	vector<int> results;
	int match_index;

	auto predicts = outputs[0].tensor<float, 2>();
	for (int i = 0; i < predicts.dimension(0); i++) {

		match_index = -1;
		for (int j = 0; j < predicts.dimension(1); j++) {
			if (predicts(i, j) > predict_confident) {
				match_index = j;
				break;
			}
		}
		
		results.push_back(match_index);
	}

	return results;
}