#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

//////////////////////////////
Mat img, imgGray, imgBlur, imgCanny, imgThre, imgDil, imgErode, imgWarp, imgCrop;
vector<Point> initialPoints, docPoints;
float w = 420, h = 596;
//////////////////////////////

Mat preProcessing(Mat img) {
    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);
    Canny(imgBlur, imgCanny, 25, 75);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(imgCanny, imgDil, kernel);
    // erode(imgDil, imgErode, kernel);

    return imgDil;
}

vector<Point> getContours(Mat img) {}

void drawPoints(vecot<Point> points) {
    for (int i = 0; i < points.size(); i++) {
		circle(imgOriginal, points[i], 10, color, FILLED);
		putText(imgOriginal, to_string(i), points[i], FONT_HERSHEY_PLAIN, 4, color, 4);
	}
}

vector<Point> reorder(vector<Point> points) {
    vector<Point> newPoints;
    vector<int> sumPoints, subPoints;

    for (int i = 0; i < 4; i++) {
        sumPoints.push_back(points[i].x + points[i].y);
        subPoints.push_back(points[i].x + points[i].y);
    }

    newPoints.push_back(points[min_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); // 0
    newPoints.push_back(points[max_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); // 1
    newPoints.push_back(points[min_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); // 2
    newPoints.push_back(points[max_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); // 3

    return newPoints;
}

Mat getWarp(Mat img, vector<Point> points, float w, float h) {
    Point2f src[4] = { points[0], points[1], points[2], points[3] };
    Point2f dst[4] = { {0.0f, 0.0f}, {w, 0.0f}, {0.0f, h}, {w, h} };

    Mat matrix = getPerspectiveTransform(src, dst);
    warpPerspective(img, imgWarp, matrix, Point(w, h));

    return imgWarp;
}

void main() {
    string path = "resources.paper.jpg";
    imgOriginal = imread(path);

    // step 1: preprocess
    imgThre = preProcessing(imgOriginal);

    // step 2: get the biggest contours
    initialPoints = getContours(imgThre);
    docPoints = recorder(initialPoints);

    // step 3: warp
    imgWarp = getWarp(imgOriginal, docPoints, w, h);

    // step 4: crop
    int cropVal = 5;
    Rect roi(cropVal, cropVal, w - (2 * cropVal), h - (2 * cropVal));
    imgCrop = imgWarp(roi);

    imshow("Image", imgOriginal);
	//imshow("Image Dilation", imgThre);
	//imshow("Image Warp", imgWarp);
	imshow("Image Crop", imgCrop);
	waitKey(0);
}