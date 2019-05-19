#ifndef _TOOL_UTILS_H_
#define _TOOL_UTILS_H_

#include <fstream>
#include <io.h>
#include <direct.h>

#include "wordRecognize.h"


float whiteBGPercent(cv::Mat src_img);
void getTrainDataForP2P();
void getOcrImg();
void getCharBinImg();
vector<string> getCharImg(ChineseRecognize* recognizer, string img_url, string question_id, string char_dir);
void autoTest();


#endif // !_TOOL_UTILS_H_