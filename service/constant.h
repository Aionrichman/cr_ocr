#ifndef _CONSTANT_H_
#define _CONSTANT_H_

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

// 光照均衡参数
const int BRIGHTNESS_KERNEL = 43;

// 倾斜检测参数
const int GRAY_THRESH = 150;
const int HOUGH_VOTE = 100;

// 二值化参数
const int CONST_C = 9;

// 膨胀核尺寸
const int DIL_KERNEL_X = 3;
const int DIL_KERNEL_Y = 3;

// 腐蚀核尺寸
const int ERO_KERNEL_X = 3;
const int ERO_KERNEL_Y = 3;

// 方向角阈值
const float THRE_TAN = 0.09;

// 连通域噪点阈值
const int MIN_STATE_BORDER = 7;

// 连通域分位
const float QUANT_PERCEENT = 0.6;

// 汉字范围
const float MAX_SIMILAR = 1.2;
const float MIN_SIMILAR = 0.2;

// 过切分阈值
const float SPLIT_PERCENT = 0.1;

// 高宽比
const float MAX_WH_RATIO = 9;
const float MIN_WH_RATIO = 0.1;

// 预测置信度
const float PREDICT_CONFIDENT = 0.9;
const float PREDICT_CONFIDENT_LOW = 0.6;

// 识别图片大小
//const int IMG_HEIGHT = 28;
//const int IMG_WIDTH = 28;
const int IMG_HEIGHT = 32;
const int IMG_WIDTH = 32;


// 字符个数
const int WORD_NUM = 3982;

struct rectStruct
{
	int order;
	int left;
	int top;
	int right;
	int down;
};

#endif // !_CONSTANT_H_