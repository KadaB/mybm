#ifndef FILTERS_H
#define FILTERS_H

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

#define UINT24_RANGE 16777216

// simple map implementation.
class SimpleMap
{
public:
    int range;
    int counter;

    int* map;
    int* value;

    SimpleMap(int range)
    {
        map = new int[range];
        value = new int[range];

        counter = 0;
        this->range = range;
    }

    ~SimpleMap() {
        delete map;
        delete value;
    }

    int getIndex(int key) {
        for(int i = 0; i < counter; i++) {
           if(key == map[i]) return i;
        }

        return -1;
    }

    bool getVal(int key, int& val) {
        int i = getIndex(key);

        if(i < 0) return false;

        val = value[i];

        return true;
    }

    bool setKey(int key, int val) {
        int i = getIndex(key);

        if(i < 0) {
            if(!push(key, val)) return false;
        }
        else {
            value[i] = val;
        }

        return true;
    }

    bool add(int key, int add) {
        // get index
        int i = getIndex(key);

        if(i < 0) {
            // not found create new entry (with add)
            if(!push(key, add)) return false;
        }
        else {
            value[i] += add;
        }

        return true;
    }

    void resetMap() {
        counter = 0;
    }
private:
    bool push(int key, int val) {
        if(counter < range) {
            map[counter] = key;
            value[counter] = val;
            counter++;

            return true;
        }

        return false;
    }

};

cv::Mat getGradientAngle(cv::Mat src);
cv::Mat getRGBGradientAngle(cv::Mat src);
void displayGradientPic(cv::Mat src);

float rgbBlockEntropySm(cv::Mat image, SimpleMap& smap);
cv::Mat RGBEntropy(cv::Mat image, int blocksize);

bool isInList(std::vector<int> v, int x);
void blockCondHist(cv::Mat block, cv::Mat& hist, int c);
cv::Mat condHist(cv::Mat image, int blocksize);
cv::Mat switchColors(cv::Mat image, cv::Mat hist);

#endif // FILTERS_H
