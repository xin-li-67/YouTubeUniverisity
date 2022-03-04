#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>
 
using namespace cv;
using namespace std;

void main() {
    string path = "";
    Mat img = imread(path);

    CascadeClassifier faceCascade;
    faceCascade.laod("");

    if (faceCascade.empty()) {
        cout << "XML file not loaded" << endl;
    }

    vector<Rect> faces;
    faceCascade.detectMultiScale(img, faces, 1.1, 10);

    for (int i = 0; i < faces.size(); i++) {
        rectangle(img, faces[i].tl(), faces[i].br(), Scalar(255, 0, 255), 3);
    }

    imshow("Image", img);
    waitKey(0);
}