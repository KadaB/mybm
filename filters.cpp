#include "filters.h"

using namespace std;
using namespace cv;

cv::Mat getGradientAngle(cv::Mat src) {
    assert(src.type() == CV_8UC1);

    cv::Mat gradX, gradY;
    Scharr(src, gradX, CV_32F, 1, 0);
    Scharr(src, gradY, CV_32F, 0, 1);

    cv::Mat mat, angle;
    cartToPolar(gradX, gradY, mat, angle);

    return angle;
}

cv::Mat getRGBGradientAngle(cv::Mat src) {
    std::vector<cv::Mat> ch;
    cv::split(src, ch);
    cv::Mat angB = getGradientAngle(ch[0]);
    cv::Mat angG = getGradientAngle(ch[1]);
    cv::Mat angR = getGradientAngle(ch[2]);

    std::vector<cv::Mat> rgb;
    rgb.push_back(angB);
    rgb.push_back(angG);
    rgb.push_back(angR);

    cv::Mat grad;
    cv::merge(rgb, grad);

    for(int i = 0; i < 3; ++i) rgb[i].release();

    assert(grad.type() == CV_32FC3);
    return grad;
}

void displayGradientPic(cv::Mat src) {
    double minValang, maxValang;
    minMaxLoc(src, &minValang, &maxValang);

    cv::Mat show;
    double range = maxValang - minValang;
    src.convertTo(show, CV_8U, 255.0/range, -minValang*255.0/range);

    cv::imshow("left", show);
    cv::waitKey(1000);
    cv::destroyWindow("left");
}


float rgbBlockEntropySm(cv::Mat image, SimpleMap& smap) {
    int width = image.cols;
    int height = image.rows;
    int total = width * height;

    // reset histogram map
    smap.resetMap();

    for(int y = 0; y < height; ++y) {
        cv::Vec3b* ptr = image.ptr<cv::Vec3b>(y);

        for(int x = 0; x < width; ++x) {
            cv::Vec3b color = ptr[x];

            int val = ((color[0] & 0xFF) << 16) |
                               ((color[1] & 0xFF) << 8)  |
                               (color[2] & 0xFF);

            // write value into map
            smap.add(val, 1);
        }
    }

    // go through map and calculate entropy
    float entr = 0.0;
    for(int i = 0; i < smap.counter; i++) {
        float count = smap.value[i];

        entr += (count / total) * log2(total/count);
    }

    // return entropy for block
    return entr;
}

Mat RGBEntropy(Mat image, int blocksize) {

    // set local help variables
    int width = image.cols;
    int height = image.rows;

    Mat filtered = Mat::zeros(height, width, CV_32F);

    // set up own simple dictionary datatype
    int margin = blocksize / 2;
    SimpleMap smap(blocksize*blocksize);

    // go through every pixel
    // calculate entropy value for block
    for(int y = 0; y < height; ++y) {
        float* ptr = filtered.ptr<float>(y);

        for(int x = 0; x < width; ++x) {
            Rect roi(max(x - margin, 0), max(y - margin, 0), min(blocksize, width - x), min(blocksize, height - y));
            Mat block = image(roi);

            float val = rgbBlockEntropySm(block, smap);
            ptr[x] = val;
        }
    }

    return filtered;
}

bool isInList(vector<int> v, int x) {
    if(std::find(v.begin(), v.end(), x) != v.end())
        return true;
    else
        return false;
}
void blockCondHist(cv::Mat block, cv::Mat& hist, int c) {
    int width = block.cols;
    int height = block.rows;

    // Vector ok, because of limited envirenment
    std::vector<int> alreadyProcessed;

    for(int y = 0; y < height; y++) {
        cv::Vec3b* ptr = block.ptr<Vec3b>(y);

        for(int x = 0; x < width; x++) {
            Vec3b color = ptr[x];
            int n = ((color[2] & 0xFF) << 16) |
                    ((color[1] & 0xFF) << 8)  |
                     (color[0] & 0xFF);

            if(n != c) {
                if(!isInList(alreadyProcessed, n)) {    // if already processed, do nothing
                    alreadyProcessed.push_back(n);      // (1) save in already processed

                    hist.at<int>(n,0) += 1;  // (2) save in conditional Matrix
                }
            }
        }
    }

    alreadyProcessed.clear();
}

Mat condHist(cv::Mat image, int blocksize) {
    int width = image.cols;
    int height= image.rows;
    int margin = blocksize/2;

    assert(sizeof(int) == 4);

    int range = UINT24_RANGE;

    Mat hist = Mat::zeros(range, 1, CV_32S);

    for(int y = 0; y < height; ++y) {
        cv::Vec3b* ptr = image.ptr<Vec3b>(y);

        for(int x = 0; x < width; ++x) {
            Rect roi(max(x - margin, 0), max(y - margin, 0),
                     min(blocksize, width - x), min(blocksize, height - y));
            Mat block = image(roi);

            Vec3b color = ptr[x];
            int c = ((color[2] & 0xFF) << 16) |
                    ((color[1] & 0xFF) << 8)  |
                     (color[0] & 0xFF);
            blockCondHist(block, hist, c);
        }
    }

    return hist;
}

cv::Mat switchColors(cv::Mat image, cv::Mat hist) {
    int width = image.cols;
    int height = image.rows;
    cv::Mat nuImg = cv::Mat::zeros(height, width, CV_32F);

    for(int y = 0; y < height; ++y) {
        float* ptr = nuImg.ptr<float>(y);
        cv::Vec3b* img_ptr = image.ptr<cv::Vec3b>(y);

        for(int x = 0; x < width; ++x) {
            Vec3b color = img_ptr[x];
            int c = ((color[2] & 0xFF) << 16) |
                    ((color[1] & 0xFF) << 8)  |
                     (color[0] & 0xFF);

            ptr[x] = hist.at<int>(c, 0);
        }
    }

    return nuImg;
}
