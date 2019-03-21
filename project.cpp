#include <project.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <set>
#include <fstream>

using namespace std;
using namespace cv;

void check_size_compatibility(const Mat &a, const Mat &b)
{
    if(a.size != b.size)
        throw Bad_MatSizeIncompatible();
}


vector<Result> get_people_pos(const Mat &src)
{
    Mat Imglabels, Imgstats, Imgcentriods;
    int Imglabelnum = connectedComponentsWithStats(src, Imglabels, Imgstats, Imgcentriods);
    Imglabels.convertTo(Imglabels, CV_32S);


    vector<Result> result;

    if(Imglabelnum == 1)    // no people
        return result;

    Mat done_record(src.size(), CV_8U, Scalar::all(0));

    set<int> ids;
    ids.insert(0);

    for(int i = 0; i < src.rows; i++)
    {
        for(int j = 0; j < src.cols; j++)
        {
            if(ids.find(Imglabels.at<int>(i, j)) != ids.end())    // checked or no people
            {
                done_record.at<uint8_t>(i, j) = 255;
                continue;
            }
            else if(done_record.at<uint8_t>(i, j) != 0)
            {
                continue;
            }

            done_record.at<uint8_t>(i, j) = 255;

            int id = Imglabels.at<int>(i, j);
            ids.insert(id);

            int left = Imgstats.at<int>(id, CC_STAT_LEFT);
            int top = Imgstats.at<int>(id, CC_STAT_TOP);
            int width = Imgstats.at<int>(id, CC_STAT_WIDTH);
            int height = Imgstats.at<int>(id, CC_STAT_HEIGHT);
            int right = left + width;
            int bottom = top + height;

            int tight_left = left;
            int tight_right = right;
            int tight_top = top;
            int tight_bottom = bottom;

            const float x_scale = 2;
            const float y_scale = 2;

            while(true) // find all unmarked block
            {
                bool find_unmark = false;

                left = max(tight_left - width/2.0*(x_scale - 1), 0.0) + 0.5;
                top = max(tight_top - height/2.0*(y_scale - 1), 0.0) + 0.5;
                right = tight_left + width*x_scale + 0.5;
                bottom = tight_top + height*y_scale + 0.5;

                for(int i = top; i < bottom && i < src.rows; i++)
                {
                    for(int j = left; j < right && j < src.cols; j++)
                    {
                        if(!done_record.at<uint8_t>(i, j) && ids.find(Imglabels.at<int>(i, j)) == ids.end())
                            // new id
                        {
                            int new_id = Imglabels.at<int>(i, j);
                            ids.insert(new_id);
                            find_unmark = true;

                            // update bounding box
                            tight_left = min(Imgstats.at<int>(new_id, CC_STAT_LEFT), tight_left);
                            tight_top = min(Imgstats.at<int>(new_id, CC_STAT_TOP), tight_top);
                            tight_right = max(tight_right, Imgstats.at<int>(new_id, CC_STAT_LEFT) +
                                            Imgstats.at<int>(new_id, CC_STAT_WIDTH));
                            tight_bottom = max(tight_bottom, Imgstats.at<int>(new_id, CC_STAT_TOP) +
                                             Imgstats.at<int>(new_id, CC_STAT_HEIGHT));

                        }
                        done_record.at<uint8_t>(i, j) = 255;
                    }
                }

                if(!find_unmark)
                {
                    break;
                }
            }
            Result r;
            r.x1 = tight_left;
            r.y1 = tight_top;
            r.x2 = tight_right;
            r.y2 = tight_bottom;
            result.push_back(r);
        }
    }
}
bool overlay(const Result &r1, const Result &r2)
{
    if (r1.x2  > r2.x1 &&
        r2.x2  > r1.x1 &&
        r1.y2  > r2.y1 &&
        r2.y2  > r1.y1
       )
        return true;
    else
        return false;
}
