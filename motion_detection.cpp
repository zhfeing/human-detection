#include <motion_detection.h>
#include <project.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void max_difference_3channels(const Mat &a, const Mat &b, Mat &res)     // a - b
{
    check_size_compatibility(a, b);

    res = Mat::zeros(a.rows, a.cols, CV_32F);
    Mat tmp = abs(a - b);
    std::vector<cv::Mat> rgbChannels(3);
    split(tmp, rgbChannels);
    res = max(rgbChannels[0], rgbChannels[1]);
    res = max(res, rgbChannels[2]);
}
void motion_detection(const Mat &frame_now, const Mat &frame_last, Mat &dst, float thresh)
{
    max_difference_3channels(frame_now, frame_last, dst);
    threshold(dst, dst, thresh, 255, CV_THRESH_TOZERO);
    threshold(dst, dst, thresh, 255, CV_THRESH_BINARY);
    Mat k = getStructuringElement(MORPH_RECT, Size(4, 4));
    morphologyEx(dst, dst, CV_MOP_OPEN, k);
    dst.convertTo(dst, CV_8U);
}
