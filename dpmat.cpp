#include "dpmat.h"

using namespace cv;
using namespace std;

DPmat::DPmat() {
}

// (1) set last entry sink in matrix (last value)
// (2-3) Initializ Edges
//    (2) initialize travelpath for last col (only south direction)
//    (3) initialize travelpath for last row (only east direction)
// (4) calculate paths till last sink (last entry) till xLast - 1, yLast - 1
// (-) save all (chosen) directions along the way
void DPmat::preCalc(Mat &matrix, Mat &sum, Mat &dirs) {
    float occlusion_south = 1.0f;
    float occlusion_east = 1.0f;
    sum = Mat::zeros(matrix.rows, matrix.cols, matrix.type());         // not initialized with zero, should not be a problem,
    dirs = Mat::zeros(matrix.rows, matrix.cols, CV_16U);               // because traversion is pre initialized with borders

    // dirs = (1, south), (2, south-east), (3, east)

    int rowLast = matrix.rows - 1;           // last index inclusive
    int colLast = matrix.cols - 1;           // last index inclusive

    // (1) initialize sink (last Entry/ terminal point/ matrix exit value)
    sum.at<float>(rowLast, colLast) = matrix.at<float>(rowLast, colLast);

    // (2-3) Initialize Edges

    // (2) calculate all last row entries down to exit value | only downward directions (so upward pre calculation)
    for(int y = rowLast - 1; y >= 0; --y) {
        // sum[y,x] = M[y,x] * occlusion_south + sum[y+1,x]
        sum.at<float>(y, colLast) = matrix.at<float>(y, colLast) * occlusion_south + sum.at<float>(y + 1, colLast);   // add current value + successor min(value)
        dirs.at<ushort>(y, colLast) = 1;    // south
    }

    // (3) initialize last col
    for(int x = colLast - 1; x >= 0; --x) {
        // sum[y,x] = M[y,x] * occlusion_east + sum[y+1,x]
        sum.at<float>(rowLast, x) = matrix.at<float>(rowLast, x) * occlusion_east + sum.at<float>(rowLast, x + 1);   // add current value + successor min(value)
        dirs.at<ushort>(rowLast, x) = 3;    // east
    }

    // (4) Main calculation (3 way [south(s), east(e), south-east(se)])
    for(int y = rowLast - 1; y >= 0; --y) {
        float* sum_ptr = sum.ptr<float>(y);
        float* sum_south_ptr = sum.ptr<float>(y+1);
        float* mat_ptr = matrix.ptr<float>(y);
        ushort* dirs_ptr = dirs.ptr<ushort>(y);

        for(int x = colLast - 1; x >= 0; --x) {
            // dirs
            //float s = sum.at<float>(y + 1, x);
            //float se = sum.at<float>(y + 1, x + 1);
            //float e = sum.at<float>(y, x + 1);
            float s = sum_south_ptr[x] * occlusion_south;        // (y+1,x)     occlusion dir
            float se = sum_south_ptr[x + 1];                     // (y+1,x+1)
            float e = sum_ptr[x + 1] * occlusion_east;           // (y, x+1)    occlusion dir

            // lowest cost till current point
            float p = min(s, min(se, e));

            //sum.at<float>(y, x) = p + matrix.at<float>(y, x);   // sum till current (cost + lowest path)
            sum_ptr[x] = p + mat_ptr[x];        // sum[y,x] = p + mat[y, x]

            // selection for traversion direction
            //if(p == s) dirs.at<ushort>(y, x) = 1;   // occlusion
            //if(p == se) dirs.at<ushort>(y, x) = 2;   // math
            //if(p == e) dirs.at<ushort>(y, x) = 3;   // occlusion

            if(p == s) dirs_ptr[x] = 1;   // occlusion
            if(p == se) dirs_ptr[x] = 2;   // math
            if(p == e) dirs_ptr[x] = 3;   // occlusion
        }
    }
}

/*
 * Traversion backtracking. Walks back the direction matrix (dirs).
 * 1 - south        add x2      occluded
 * 2 - south-east   add x1, x2  match   (cyclopian)
 * 3 - east         add x1      occluden
 * abwärts ist index für links
 * x1 linkes Bild, x2 Rechtes Bild
 */
void DPmat::disparityFromDirs(Mat &sum, Mat &dirs, Mat &disp, int line, int offset) {
    assert(dirs.type() == CV_16U);

    // wir bekommen jetzt einen index x, y
    int rowLast = dirs.rows - 1;
    int colLast = dirs.cols - 1;

    int lastval = -1;
    int x1 = 0;
    int x2 = 0;

    float minVal = numeric_limits<float>::max();
    int minIndex = 0;

    // seek top entry
    for(x2 = 0; x2 < sum.cols; ++x2) {
        float val = sum.at<float>(x1, x2);
        if(val > minVal) {
            minIndex = x2;
            minVal = val;
        }
    }

    x2 = minIndex;

    // safe x1, x2 as disparity match
    ushort disparity = abs(x2 - x1);
    ushort* disp_ptr = disp.ptr<ushort>(line);

    disp_ptr[x1 + offset] = disparity;

    while(x1 < rowLast && x2 < colLast) {
        ushort d = dirs.at<ushort>(x1, x2);

        if(d == 1) {    // 1 = down, skipping left index, left got occloded (occlusion from right)
            x1++;
            if(lastval >= 0) disp_ptr[x1 + offset] = lastval;   // dips[line, x1 + offset] = lastval
            //disp_ptr[x1 + offset] = 0;
        }
        if(d == 2) { // match
            // next entry will be match
            x1++;
            x2++;
            disparity = abs(x2 - x1);

            disp_ptr[x1 + offset] = disparity;
            lastval = disparity;
        }
        if(d == 3) { // 2 = right, skipping right index, occlusion don't care..
            x2++;
            if(lastval >= 0) disp_ptr[x1 + offset] = lastval;   // dips[line, x1 + offset] = lastval
            //disp_ptr[x1 + offset]= 0;
        }
    }
}

// Draw path in disparity space image
void DPmat::drawPath(Mat &sum, Mat &dirs, Mat &image) {
    assert(dirs.type() == CV_16U);

    // wir bekommen jetzt einen index x, y
    int rowLast = dirs.rows - 1;
    int colLast = dirs.cols - 1;

    int x1 = 0;
    int x2 = 0;

    float minVal = numeric_limits<float>::max();
    int minIndex = 0;

    // seek top entry
    for(x2 = 0; x2 < sum.cols; ++x2) {

        float val = sum.at<float>(x1, x2);

        if(val < minVal) {
            minIndex = x2;
            minVal = val;
        }
    }

    x2 = minIndex;

    Vec3b r, g, b;

    r[0] = 255;
    r[1] = 0;
    r[2] = 0;

    g[0] = 0;
    g[1] = 255;
    g[2] = 0;

    b[0] = 0;
    b[1] = 0;
    b[2] = 255;

    image.at<Vec3b>(x1, x2) = r;

    while(x1 < rowLast && x2 < colLast) {
        ushort d = dirs.at<ushort>(x1, x2);

        if(d == 1) {    // 1 = down, skipping left index, left got occloded (occlusion from right)
            x1++;
            image.at<Vec3b>(x1, x2) = r;
        }
        else if(d == 2) { // match
            // next entry will be match
            x1++;
            x2++;

            image.at<Vec3b>(x1, x2) = g;
        }
        else if(d == 3) { // 2 = right, skipping right index, occlusion don't care..
            x2++;

            image.at<Vec3b>(x1, x2) = b;
        }
    }
}
