#ifndef _CONSTANT_H_
#define _CONSTANT_H_

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

// ���վ������
const int BRIGHTNESS_KERNEL = 43;

// ��б������
const int GRAY_THRESH = 150;
const int HOUGH_VOTE = 100;

// ��ֵ������
const int CONST_C = 9;

// ���ͺ˳ߴ�
const int DIL_KERNEL_X = 3;
const int DIL_KERNEL_Y = 3;

// ��ʴ�˳ߴ�
const int ERO_KERNEL_X = 3;
const int ERO_KERNEL_Y = 3;

// �������ֵ
const float THRE_TAN = 0.09;

// ��ͨ�������ֵ
const int MIN_STATE_BORDER = 7;

// ��ͨ���λ
const float QUANT_PERCEENT = 0.6;

// ���ַ�Χ
const float MAX_SIMILAR = 1.2;
const float MIN_SIMILAR = 0.2;

// ���з���ֵ
const float SPLIT_PERCENT = 0.1;

// �߿��
const float MAX_WH_RATIO = 9;
const float MIN_WH_RATIO = 0.1;

// Ԥ�����Ŷ�
const float PREDICT_CONFIDENT = 0.9;
const float PREDICT_CONFIDENT_LOW = 0.6;

// ʶ��ͼƬ��С
//const int IMG_HEIGHT = 28;
//const int IMG_WIDTH = 28;
const int IMG_HEIGHT = 32;
const int IMG_WIDTH = 32;


// �ַ�����
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