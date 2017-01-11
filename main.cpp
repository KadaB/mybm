#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <vector>
#include <limits>
#include <ctime>

#include "filters.h"
#include "blockmatching.h"
#include "dpmat.h"

using namespace std;
using namespace cv;

void testNuDP();

void print_help() {
    cout << "Dynamic programming Stereo blockmatching algorithm:" << endl;
    cout << "Usage:\t-s <leftfile> <rightfile>" << endl;
    cout << "\t-o <disparity_output>" << endl;
    cout << "\t-t test" << endl;
    cout << "\t-ti DSI test" << endl;
    cout << "\t-b <Blocksize>" << endl;
    cout << "\t-c color map(jet)" << endl;
}

int main(int argc, char *argv[])
{
    string leftFile = "";
    string rightFile= "";
    bool files = false;
    bool display = false;
    bool dsi = false;
    bool gradient = false;
    bool cmap = false;
    string outfile = "";
    int blocksize = 3;

    for(int i = 1; i < argc; ++i) {
        string arg = argv[i];

        if(arg == "-s" && i + 2 < argc) {
            // left and right image
            leftFile = argv[++i];
            rightFile= argv[++i];
            files = true;
        }
        if(arg == "-h") {
            print_help();
        }
        if(arg == "-o" && i + 1 < argc) {
            outfile = argv[++i];
        }
        if(arg == "-ti") {
            dsi = true;
        }
        if(arg == "-d") {
            display = true;
        }
        if(arg == "-b" && i + 1 < argc) {
            blocksize = atoi(argv[++i]);
        }

        if(arg == "-g") {
            // use gradient pics
            gradient = true;
        }

        if(arg == "-c") {
            cmap = true;
        }
    }

    if(files) {
        Mat left = imread(leftFile);
        Mat right = imread(rightFile);
        Mat leftg = imread(leftFile, IMREAD_GRAYSCALE);
        Mat rightg = imread(rightFile, IMREAD_GRAYSCALE);
        /*Mat l, r;
        Mat h1 = condHist(left, 3);
        Mat h2 = condHist(right, 3);
        Mat l1 = switchColors(left, h1);
        Mat r1 = switchColors(right, h2);
        normalize(l1, l, 0, 255, CV_MINMAX, CV_8UC1);
        normalize(r1, r, 0, 255, CV_MINMAX, CV_8UC1);*/

        assert(left.size() == right.size() && "image size not equal");
        /*
        if(dsi) {
            vector<int> list;
            list.push_back(10);
            list.push_back(50);
            list.push_back(300);
            BlockMatching::getSimularityMap(left, right, blocksize, list);

            return 0;
        }*/

        BlockMatching bm;

        // Aggregate Blockmatchingfunctions
        bm.functions.push_back(new RGBCost(left, right, 1));
        //bm.functions.push_back(new GradientCost(left, right, 1));
        //bm.functions.push_back(new CensusCost(leftg, rightg, 3, 1));
        //bm.functions.push_back(new CondHistCost(left, right, 1.0));

        clock_t start = clock();
        Mat disparity = bm.compute(left.size(), blocksize);

        // benchmarking
        double time = ((clock() - start) / (double) CLOCKS_PER_SEC);
        if(time > 60) {
            cout << "Time taken: " <<  time / 60 << " minutes" << endl;
        }
        else {
            cout << "Time taken: " <<  time << " seconds" << endl;
        }

        // Normalize results to display as image
        Mat out, out2;
        cv::normalize(disparity, out, 0, 255, NORM_MINMAX, CV_8UC1);

        if(cmap) {
            applyColorMap(out, out2, COLORMAP_JET);
        }
        else {
            out2 = out;
        }

        if(!outfile.empty()) {
            imwrite(outfile, out2);
        }
        else {
            display = true;
        }

        if(display){
            namedWindow("disparity");
            imshow("disparity", out2);
            waitKey(0);
        }
    }
    else {
        print_help();
    }

    return 0;
}

// Debugging function for minimal path
void testNuDP() {
    float data[] = { 83, 35, 62, 26, 11,
                     86, 86, 27, 40, 68,
                     77, 92, 90, 26, 67,
                     15, 49, 59, 72, 29,
                     93, 21, 63, 36, 82 };

    for(int i = 0; i < 5; i++) {
        for(int j= 0; j < 5; j++) {
            data[j*5+i] = rand() % 5;
        }
    }

    Mat matrix = Mat(5,5, CV_32F, data);

    cout << matrix << endl;
    cout << "---------------------------------------" << endl;

    Mat dirs, sum;
    DPmat::preCalc(matrix, sum, dirs);
    cout << "sum: " << endl << sum << endl;
    cout << "dirs: " << endl << dirs << endl;
}
