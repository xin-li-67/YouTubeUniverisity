#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace std;
using namespace cv;

////// IMAGE //////
void main() {
    string path = "";
    Mat img = imread(path);
    imshow("Image", img);
    waitKey(0);
}

////// VIDEO //////
void main() {
    string path = "";
    VideoCapture cap(path);
    Mat img;

    while (true) {
        cap.read(img);
        imshow("Image", img);
        waitKey(0);
    }
}

////// WEBCAM /////
void main() {
    VideoCapture cap(0);
    Mat img;

    while (true) {
        cap.read(img);
        imshow("Image", img);
        waitKey(1);
    }
}