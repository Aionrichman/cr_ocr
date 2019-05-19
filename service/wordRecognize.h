#ifndef _WORD_RECOGNIZE_H_
#define _WORD_RECOGNIZE_H_

#include "wordLocate.h"
#include "reModel.h"


vector<vector<int>> doubleRecognize(cv::Mat& src_img, ChineseRecognize* recognizer);
vector<vector<rectStruct>> doubleRecognizeWithPos(cv::Mat& src_img, ChineseRecognize* recognizer);


#endif // !_WORD_RECOGNIZE_H

