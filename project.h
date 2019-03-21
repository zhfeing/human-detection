#ifndef PROJECT_H
#define PROJECT_H

#include <opencv2/opencv.hpp>
#include <vector>

#define _CV cv::
#define _STD std::

class Bad_MatSizeIncompatible{ };

void check_size_compatibility(const _CV Mat &a, const _CV Mat &b);

typedef struct Result
{
    int x1;
    int y1;
    int x2;
    int y2;
    bool operator()(const Result &lhs, const Result &rhs)
    {
        int s1 = (lhs.x2 - lhs.x1)*(lhs.y2 - lhs.y1);
        int s2 = (rhs.x2 - rhs.x1)*(rhs.y2 - rhs.y1);
        return s1 > s2;
    }
}Result;

_STD vector<Result> get_people_pos(const _CV Mat &src);
bool overlay(const Result &r1, const Result &r2);

#endif // PROJECT_H
