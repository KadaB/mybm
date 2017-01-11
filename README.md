# mybm

A stereo blockmatching algorithm I programmed for my thesis.

<img src="https://cloud.githubusercontent.com/assets/22398803/21848895/52d191fe-d803-11e6-86bf-82419f65b896.png">

It uses dynamic programming cost minimization and has the ability to aggregate cost functions.

* Input:

  rectified stereo image (for example left.png right.png)
* Output:

  Disparity image/ inverse depth map.

There is no minimal or maximal disparity so calculating the image might take a while.

**Usage:**

`mybm -s left.png right.png`

will calculate a disparity map for the stereoset left.png right.png.

`mybm -s left.png right.png -c -o disparity.png`

will save the disparity map in an image file. "-c" will make sure that the disparity map is colored in the opencv "jet" colormap for better reckognition of height.
disparity.png will be the output image

Based on the following libraries:
* OpenCV

### Building and execution on a linux based system

**Install dependencies on Ubuntu (or Debian based distributions):**

`sudo apt-get install build-essential cmake libopencv-dev`

**Build the programm:**

1. `cmake .`
2. `make`

Release version:

1. `mkdir release`
2. `cd release`
3. `cmake -DCMAKE_BUILD_TYPE=Release ..`
4. `make`

If you build the programm succesfully you should be able to see the "mybm" executable
