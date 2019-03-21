#ifndef MOTION_DETECTION_H
#define MOTION_DETECTION_H

#include <opencv2/opencv.hpp>
#define _CV cv::


void max_difference_3channels(const _CV Mat &a, const _CV Mat &b, _CV Mat &res);
void motion_detection(const _CV Mat &frame_now, const _CV Mat &frame_last, _CV Mat &dst, float thresh = 15.0);

#endif // MOTION_DETECTION_H
