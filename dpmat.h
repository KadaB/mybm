#ifndef DPMAT_H
#define DPMAT_H

#include "essentials.h"

class DPmat
{
public:
    DPmat();
    static void preCalc(cv::Mat &matrix, cv::Mat &sum, cv::Mat &dirs);
    static void disparityFromDirs(cv::Mat &sum, cv::Mat &dirs, cv::Mat &disp, int line, int offset);
    static void drawPath(cv::Mat &sum, cv::Mat &dirs, cv::Mat &image);
};

#endif // DPMAT_H
