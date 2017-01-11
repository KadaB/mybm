#ifndef BLOCKMATCHING_H
#define BLOCKMATCHING_H

#include <opencv2/opencv.hpp>
#include <limits>
#include "filters.h"

/* interface cost_function:
 *   aggregate(roiLeft, roiRight)
 *   costFunction(posLeft, posRight)
 *   lambda     // weight
 */
class CostFunction {
public:
    float lambda;

    cv::Mat left;
    cv::Mat right;

    int blocksize;
    int margin;

    float normCost;
    float normWin;

    CostFunction( cv::Mat left, cv::Mat right, float lambda) {
		lambda = 1.0;
        this->left = left;
        this->right = right;
        imageType(left, right);
        this->lambda = lambda;
    }

    virtual bool imageType(cv::Mat left, cv::Mat right) {
        assert(left.size() == right.size());

        return true;
    }

    virtual float aggregate(int x1, int x2, int y) = 0;

    float p(float cost) {
        return 1 - exp(-cost / lambda);
    }

    ~CostFunction() {}
};

class RGBCost : public CostFunction {
public:
    RGBCost(cv::Mat left, cv::Mat right, float lambda) : CostFunction(left, right, lambda) {}

    bool imageType(cv::Mat left, cv::Mat right) {
        assert(left.type() == right.type() && "imgL imgR types not equal");
        assert(left.type() == CV_8UC3 && "img type not supported");

        return true;
    }

    // aggregate over a ROI of input images
    float aggregate(int x1, int x2, int y) {
        float sum = 0;
        for (int i = y - margin; i <= y + margin; ++i) {
            cv::Vec3b* lptr = left.ptr<cv::Vec3b>(i);
            cv::Vec3b* rptr = right.ptr<cv::Vec3b>(i);

            for ( int j = -margin; j <= margin; ++j) {
                sum += eukl(lptr[x1 + j], rptr[x2 + j]);      // cost function
            }
        }

        return sum / sqrt(255*255 + 255*255+255*255);      // normalize to winsize*1.0
    }

    float eukl(cv::Vec3b l, cv::Vec3b r) {
        float a = l[0] - r[0];
        float b = l[1] - r[1];
        float c = l[2] - r[2];
        return std::sqrt(a*a + b*b + c*c);
    }

    ~RGBCost() {}
};

class FloatCost : public CostFunction {
public:
    FloatCost(cv::Mat left, cv::Mat right, float lambda) : CostFunction(left, right, lambda) {}

    bool imageType(cv::Mat left, cv::Mat right) {
        assert(left.type() == right.type() && "imgL imgR types not equal");
        assert(left.type() == CV_32F && "img type not supported");

        return true;
    }

    // aggregate over a ROI of input images
    float aggregate(int x1, int x2, int y) {
        float sum = 0;
        for (int i = y - margin; i <= y + margin; ++i) {
            float* lptr = left.ptr<float>(i);
            float* rptr = right.ptr<float>(i);

            for ( int j = -margin ; j <= margin; ++j) {
                sum += abs(lptr[x1 + j] - rptr[x2 + j]);      // cost function
            }
        }

        return sum / (blocksize*blocksize*lambda);
    }
};

class CondHistCost : public CostFunction {
public:
    cv::Mat nuLeft, nuRight;
    CondHistCost(cv::Mat left, cv::Mat right, float lambda) : CostFunction(left, right, lambda) {
        cv::Mat histl = condHist(left, 3);
        nuLeft = switchColors(left, histl);
        cv::Mat histr = condHist(right, 3);
        nuRight = switchColors(right, histr);
    }

    bool imageType(cv::Mat left, cv::Mat right) {
        assert(left.type() == right.type() && "imgL imgR types not equal");
        assert(left.type() == CV_32F && "img type not supported");

        return true;
    }

    // aggregate over a ROI of input images
    float aggregate(int x1, int x2, int y) {
        float sum = 0;
        for (int i = y - margin; i <= y + margin; ++i) {
            float* lptr = nuLeft.ptr<float>(i);
            float* rptr = nuRight.ptr<float>(i);

            for ( int j = -margin ; j <= margin; ++j) {
                sum += abs(lptr[x1 + j] - rptr[x2 + j]);      // cost function
            }
        }

        return sum / (blocksize*blocksize*lambda);
    }
};


class GrayCost : public CostFunction {
public:
    GrayCost(cv::Mat left, cv::Mat right, float lambda) : CostFunction(left, right, lambda) {}

    bool imageType(cv::Mat left, cv::Mat right) {
        assert(left.type() == right.type() && "imgL imgR types not equal");
        assert(left.type() == CV_8UC1 && "img type not supported");

        return true;
    }

    // aggregate over a ROI of input images
    float aggregate(int x1, int x2, int y) {
        float sum = 0;
        for (int i = y - margin; i <= y + margin; ++i) {
            uchar* lptr = left.ptr<uchar>(i);
            uchar* rptr = right.ptr<uchar>(i);

            for ( int j = -margin; j <= margin; ++j) {
                sum += abs(lptr[x1 + j] - rptr[x2 + j]);      // cost function
            }
        }

        return sum / (blocksize*blocksize*255.0);
    }
};

class GradientCost : public CostFunction {
public:
    cv::Mat l_grad;   // 3 channel float
    cv::Mat r_grad;   // 3 channel float

    GradientCost(const cv::Mat left, const cv::Mat right, float lambda) : CostFunction(left, right, lambda) {
        l_grad = getRGBGradientAngle(left);
        r_grad = getRGBGradientAngle(right);

        //displayGradientPic(l_grad);
        //displayGradientPic(r_grad);
    }

    bool imageType(cv::Mat left, cv::Mat right) {
        assert(left.type() == right.type() && "imgL imgR types not equal");
        assert(left.type() == CV_8UC3 && "img type not supported");

        return true;
    }

    // aggregate over a ROI of input images
    float aggregate(int x1, int x2, int y) {
        float sum = 0;
        for (int i = y - margin; i <= y + margin; ++i) {
            cv::Vec3f* lptr = l_grad.ptr<cv::Vec3f>(i);
            cv::Vec3f* rptr = r_grad.ptr<cv::Vec3f>(i);

            for ( int j = -margin; j <= margin; ++j) {
                sum += eukl(lptr[x1 + j], rptr[x2 + j]);      // cost function
            }
        }

        return sum / sqrt(255*255 + 255*255 + 255*255);      // normalize to winSize * 1.0
    }

    float eukl(cv::Vec3f l, cv::Vec3f r) {
        float a = l[0] - r[0];
        float b = l[1] - r[1];
        float c = l[2] - r[2];
        return std::sqrt(a*a + b*b + c*c);
    }

    ~GradientCost() {
        l_grad.release();
        r_grad.release();
    }
};

class CensusCost : public CostFunction {
public:
    int censusWindow;
    int censusMargin;
    CensusCost(cv::Mat left, cv::Mat right, int censusWindow, float lambda) : CostFunction(left, right, lambda) {
        // census.... nimmt einen Block
        this->censusWindow = censusWindow;
        this->censusMargin = censusWindow / 2;

        this->normWin = censusWindow * censusWindow;
        // nimmt einen Block
    }

    bool imageType(cv::Mat left, cv::Mat right) {
        assert(left.type() == right.type() && "imgL imgR types not equal");
        assert(left.type() == CV_8UC1 && "img type not supported");

        return true;
    }

    unsigned int census(int x1, int x2, int y, uchar c1, uchar c2) {
        unsigned int diff = 0;

        for(int i = y - censusMargin; i <= y + censusMargin; ++i) {
            uchar* lptr = left.ptr<uchar>(i);
            uchar* rptr = right.ptr<uchar>(i);

            for(int j = -censusMargin; j <= censusMargin; ++j) {
                bool t1 = (c1 < lptr[x1 + j]);
                bool t2 = (c2 < rptr[x2 + j]);

                if(t1 != t2) diff++;
            }
        }

        return diff; /// (censusWindow*censusWindow);
    }

    float aggregate(int x1, int x2, int y) {
        float sum = 0;
        /*for(int i = y - margin; i <= y + margin; ++i) {
            uchar *lptr = left.ptr<uchar>(i);
            uchar *rptr = right.ptr<uchar>(i);

            for(int j = -margin; j <= margin; ++j)
                sum += census(x1 + j, x2 + j, i, lptr[x1 + j], rptr[x2 + j]);
        }*/
        uchar *lptr = left.ptr<uchar>(y);
        uchar *rptr = right.ptr<uchar>(y);
        sum = census(x1, x2, y, lptr[x1], rptr[x2]);
        return sum / normWin;
    }
};

class CensusFloatCost : public CostFunction {
public:
    int censusWindow;
    int censusMargin;

    CensusFloatCost(cv::Mat left, cv::Mat right, int censusWindow, float lambda) : CostFunction(left, right, lambda) {
        // census.... nimmt einen Block
        this->censusWindow = censusWindow;
        this->censusMargin = censusWindow / 2;
    }

    bool imageType(cv::Mat left, cv::Mat right) {
        assert(left.type() == right.type() && "imgL imgR types not equal");
        assert(left.type() == CV_32F && "img type not supported");

        return true;
    }

    unsigned int census(int x1, int x2, int y, float c1, float c2) {
        unsigned int diff = 0;

        for(int i = y - censusMargin; i <= y + censusMargin; ++i) {
            float* lptr = left.ptr<float>(i);
            float* rptr = right.ptr<float>(i);

            for(int j = -censusMargin; j <= censusMargin; ++j) {
                bool t1 = (c1 < lptr[x1 + j]);
                bool t2 = (c2 < rptr[x2 + j]);

                if(t1 != t2) diff++;
            }
        }

        return diff;
    }

    float aggregate(int x1, int x2, int y) {
        float sum = 0;
        for(int i = y - margin; i <= y + margin; ++i) {
            float *lptr = left.ptr<float>(i);
            float *rptr = right.ptr<float>(i);

            for(int j = -margin; j <= margin; ++j)
                sum += census(x1 + j, x2 + j, i, lptr[x1 + j], rptr[x2 + j]);
        }
        float *lptr = left.ptr<float>(y);
        float *rptr = right.ptr<float>(y);
        //sum = census(x1, x2, y, lptr[x1], rptr[x2]);
        return sum / (censusWindow*censusWindow*lambda);
    }
};

class RGBCensusCost : public CostFunction {
public:
    int censusWindow;
    int censusMargin;

    RGBCensusCost(cv::Mat left, cv::Mat right, int censusWindow, float lambda) : CostFunction(left, right, lambda) {
        // census.... nimmt einen Block
        this->censusWindow = censusWindow;
        this->censusMargin = censusWindow / 2;
        normCost = censusWindow*censusWindow*3;
        // nimmt einen Block
    }

    bool imageType(cv::Mat left, cv::Mat right) {
        assert(left.type() == right.type() && "imgL imgR types not equal");
        assert(left.type() == CV_8UC3 && "img type not supported");

        return true;
    }

    unsigned int census(int x1, int x2, int y, cv::Vec3b c1, cv::Vec3b c2) {
        unsigned int diff = 0;

        for(int i = y - censusMargin; i <= y + censusMargin; ++i) {
            cv::Vec3b* lptr = left.ptr<cv::Vec3b>(i);
            cv::Vec3b* rptr = right.ptr<cv::Vec3b>(i);

            for(int j = -censusMargin; j <= censusMargin; ++j) {
                cv::Vec3b cl = lptr[x1 + j];
                cv::Vec3b cr = rptr[x2 + j];

                for(int ch = 0; ch < 3; ++ch) {
                    bool t1 = (c1[ch] < cl[ch]);
                    bool t2 = (c2[ch] < cr[ch]);

                    if(t1 != t2) diff++;
                }
            }
        }

        return diff;
    }

    float aggregate(int x1, int x2, int y) {
        float sum = 0;
        for(int i = y - margin; i <= y + margin; ++i) {
            cv::Vec3b *lptr = left.ptr<cv::Vec3b>(i);
            cv::Vec3b *rptr = right.ptr<cv::Vec3b>(i);

            for(int j = -margin; j <= margin; ++j)
                sum += census(x1 + j, x2 + j, i, lptr[x1 + j], rptr[x2 + j]);
        }
        //cv::Vec3b *lptr = left.ptr<cv::Vec3b>(y);
        //cv::Vec3b *rptr = right.ptr<cv::Vec3b>(y);

        return sum / normCost;
    }
};

class RGBGradCensusCost : public CostFunction {
public:
    int censusWindow;
    int censusMargin;

    float normCost;
    float normWin;

    cv::Mat l_grad;
    cv::Mat r_grad;

    RGBGradCensusCost(cv::Mat left, cv::Mat right, int censusWindow, float lambda) : CostFunction(left, right, lambda) {
        // census.... nimmt einen Block
        this->censusWindow = censusWindow;
        this->censusMargin = censusWindow / 2;
        normWin = censusWindow*censusWindow*3;
        // nimmt einen Block
        l_grad = getRGBGradientAngle(left);
        r_grad = getRGBGradientAngle(right);
    }

    bool imageType(cv::Mat left, cv::Mat right) {
        assert(left.type() == right.type() && "imgL imgR types not equal");
        assert(left.type() == CV_8UC3 && "img type not supported");

        return true;
    }

    unsigned int census(int x1, int x2, int y, cv::Vec3f c1, cv::Vec3f c2) {
        unsigned int diff = 0;

        for(int i = y - censusMargin; i <= y + censusMargin; ++i) {
            cv::Vec3f* lptr = l_grad.ptr<cv::Vec3f>(i);
            cv::Vec3f* rptr = r_grad.ptr<cv::Vec3f>(i);

            for(int j = -censusMargin; j <= censusMargin; ++j) {
                cv::Vec3f cl = lptr[x1 + j];
                cv::Vec3f cr = rptr[x2 + j];

                for(int ch = 0; ch < 3; ++ch) {
                    bool t1 = (c1[ch] < cl[ch]);
                    bool t2 = (c2[ch] < cr[ch]);

                    if(t1 != t2) diff++;
                }
            }
        }

        return diff;
    }

    float aggregate(int x1, int x2, int y) {
        float sum = 0;
        /*for(int i = y - margin; i <= y + margin; ++i) {
            cv::Vec3f *lptr = l_grad.ptr<cv::Vec3f>(i);
            cv::Vec3f *rptr = r_grad.ptr<cv::Vec3f>(i);

            for(int j = -margin; j <= margin; ++j)
                sum += census(x1 + j, x2 + j, i, lptr[x1 + j], rptr[x2 + j]);
        }*/
        cv::Vec3f *lptr = l_grad.ptr<cv::Vec3f>(y);
        cv::Vec3f *rptr = r_grad.ptr<cv::Vec3f>(y);

        sum = census(x1, x2, y, lptr[x1], rptr[x2]);
        return sum / normWin;
    }
};

class BlockMatching
{
public:
    std::vector<CostFunction*> functions;

    // stats
    std::vector<float> mins;
    std::vector<float> maxs;

    BlockMatching();
    ~BlockMatching();

    cv::Mat SADmap(cv::Mat left, cv::Mat right, int blocksize);
    void lineSAD(cv::Mat left, cv::Mat right, int blocksize, cv::Mat &map, int y);
    //static void getSimularityMap(cv::Mat left, cv::Mat right, int blocksize, std::vector<int> entries);
    cv::Mat combineDisparitySpace(std::vector<cv::Mat> &maps, std::vector<float> &factors);
    cv::Mat disparitySpace(cv::Size imageSize, int blocksize, int y);
    cv::Mat compute(cv::Size imageSize, int blocksize);
};

#endif // BLOCKMATCHING_H
