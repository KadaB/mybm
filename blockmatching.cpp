#include "blockmatching.h"
#include "dpmat.h"

using namespace std;
using namespace cv;

BlockMatching::BlockMatching()
{
}

BlockMatching::~BlockMatching() {
    for(size_t i = 0; i < functions.size(); ++i)
        delete functions[i];
}

cv::Mat BlockMatching::disparitySpace(Size imageSize, int blocksize, int y) {
    int margin = blocksize / 2;
    int start = margin;
    int stopW = imageSize.width - margin;
    int workSpace = stopW - start;

    // leave out the borders
    //Mat map = Mat(workSpace, workSpace, CV_32F);        // not preinitialized.. // numeric_limits<float>::max());
    Mat map = Mat(workSpace, workSpace, CV_32F, numeric_limits<float>::max());

    //int dmax = 101;
    for(int x1 = start; x1 < stopW; x1++) {
        float* ptr = map.ptr<float>(x1 - margin);       // [x1 - margin, x2 - margin]

        //ptr[max(x1 - 1, start) - margin] = numeric_limits<float>::max();              // fast borders
        //ptr[min(x1 + dmax, stopW - 1) - margin] = numeric_limits<float>::max();
        //for(int x2 = x1; x2 < min(x1 + dmax, stopW); x2++) {
        for(int x2 = start; x2 < stopW; x2++) {

            // combine costs
            float cost = 0;
            for(size_t i = 0; i < functions.size(); ++i) {
                float val = functions[i]->aggregate(x1, x2, y);
                mins[i] = min(mins[i], val);                        // debug
                maxs[i] = max(maxs[i], val);                        // debug
                cost += val;
            }

            // x1, x2. Das heißt x1 sind die Zeilen. Wir gehen jedes Mal die Zeilen runter.
            // geht nur von 0 - workspace, deshalb margin abziehen
            //map.at<float>(x1 - margin, x2 - margin) = greySad(leftRoi, rightRoi);
            ptr[x2 - margin] = cost;
        }
    }
    return map;
}


Mat BlockMatching::compute(Size imageSize, int blocksize) {
    Mat disparity = Mat::zeros(imageSize.height, imageSize.width, CV_16U);

    int margin = blocksize / 2;
    int start = margin;
    int stopH = imageSize.height - margin;
    int stopW = imageSize.width - margin;

    assert(stopH - start > 0);          // image to small
    assert(stopW - start > 0);          // image to small

    cout << "process: " << flush;

    int tenpercent = (stopH - start) / 10;

    // set blocksize in costfunctions
    for(size_t i = 0; i < functions.size(); ++i) {
        functions[i]->blocksize = blocksize;
        functions[i]->margin = blocksize / 2;
    }

    mins.reserve(functions.size()); for(size_t i = 0; i < functions.size(); ++i) mins[i] = numeric_limits<float>::max();    // debug
    maxs.reserve(functions.size()); for(size_t i = 0; i < functions.size(); ++i) maxs[i] = numeric_limits<float>::min();    // debug

    // Each scanline do dynamic programming disparity space traversion, write back disparity values
    for(int y = start; y < stopH; ++y) {
        Mat simmap = disparitySpace(imageSize, blocksize, y);

        Mat sum, dirs;

        DPmat::preCalc(simmap, sum, dirs);
        DPmat::disparityFromDirs(sum, dirs, disparity, y, margin);

        if(((stopH - y) % tenpercent) == 0) cout << (((stopH - y)*10) / tenpercent) << "%, " << flush;
    }
    cout << endl;

    for(size_t i = 0; i < functions.size(); ++i)                            // debug
        cout << i << ": min: " << mins[i] << " max: " << maxs[i] << endl;   // debug

    return disparity;
}

Mat BlockMatching::combineDisparitySpace(vector<Mat> &maps, vector<float> &factors) {
    assert(maps.size() > 0);

    int width = maps[0].cols;
    int height = maps[0].rows;
    Mat combined = Mat(height, width, maps[0].type());

    for(int y = 0; y < height; ++y) {
        for(int x = 0; x < width; ++x) {

            float sum = 0;
            for(size_t i = 0; i < maps.size(); ++i) {
                // erste map...
                sum += maps[i].at<float>(y, x) * factors[i];
            }

            combined.at<float>(y, x) = sum;
        }
    }

    return combined;
}

/*
 * Debugging Function. Displays Disparity Space in Grayscale Image.
void BlockMatching::getSimularityMap(Mat left, Mat right, int blocksize, vector<int> entries) {
    cout << "assertions" << endl;
    assert(left.size == right.size);
    assert(left.type() == right.type());
    assert(left.type() == CV_8UC1);             // should be 8bit picture

    int margin = blocksize / 2;
    int start = margin;
    int stopH = left.rows - margin;
    int stopW = left.cols - margin;

    assert(stopH - start > 0);          // image to small
    assert(stopW - start > 0);          // image to small

    cv::namedWindow("test");

    // Gradienten... schauen, ob man die Gradienten Matchen kann
    // Schauen, ob man die anderen Werte matchen kann
    //

    for(size_t i = 0; i < entries.size(); i++) {
        cout << "entry: " << entries[i] << endl;

        // Disparity space matrix
        vector<float> factors;
        vector<Mat> maps;
        vector<CostFunction*> funcs;
        CostFunction a = RGBCost(&left, &right, 1.0);
        funcs.push_back(&a);

        maps.push_back(disparitySpace(left, right, blocksize, entries[i], funcs));
        factors.push_back(1);

        Mat combined = BlockMatching::combineDisparitySpace(maps, factors);

        Mat grey, image;

        cv::normalize(combined, grey, 0, 255, NORM_MINMAX, CV_8UC1);
        cv::cvtColor(grey, image, CV_GRAY2BGR);

        // Disparity space optimization
        Mat sum, dirs;
        DPmat::preCalc(combined, sum, dirs);
        DPmat::drawPath(sum, dirs, image);

        cv::imshow("test", image);
        cv::waitKey(0);
    }

    cv::destroyAllWindows();
}*/
