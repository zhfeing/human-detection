#include <project.h>
#include <motion_detection.h>
#include <mix_gauss_detection.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <algorithm>

using namespace std;
using namespace cv;

int main()
{
    string file_name = "level_1";
    VideoCapture cap(file_name + ".mp4");

    if (!cap.isOpened())  // if not success, exit program
    {
        cout << "Cannot open the vedio" << endl;
        return -1;
    }

    Size size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    VideoWriter writer(file_name + "_result.mp4",
                       VideoWriter::fourcc('M', 'P', '4', 'V'),
                       cap.get(CV_CAP_PROP_FPS), size);

    int frame_id = 0;
    Mat frame, last_frame, blur_frame;
    Size frame_size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT));
    cout << "frame size: " << frame_size << endl;

    Mat outcome;

    while(true)
    {
        frame_id++;
        cap >> frame;
        if(frame.empty())
            break;
        frame.copyTo(outcome);

        blur(frame, blur_frame, Size(5, 5));

        if(frame_id == 1)
        {
            blur_frame.copyTo(last_frame);
        }

        // motion detection
        Mat dynamic_frame;
        motion_detection(blur_frame, last_frame, dynamic_frame, 10);

        vector<Result> res = get_people_pos(dynamic_frame);
        sort(res.begin(), res.end(), Result());

        res.resize(max(res.size()/2.0, 15.0));
        vector<bool> keep(15, true);
        for(int i = 1; i < res.size(); i++)
        {
            for(int j = 0; j < i; j++)
            {
                if(keep[j] && overlay(res[j], res[i]))
                {
                    keep[i] = false;
                    continue;
                }

            }
        }

        for(int i = 0; i < res.size() && i < res.size() && i < 5; i++)
        {
            if(res[i].x2 - res[i].x1 < 30 || res[i].y2 - res[i].y1 < 30 || !keep[i] ||
               res[i].x2 - res[i].x1 > frame_size.width*2.0/3.0)
                continue;
            rectangle(outcome, Rect(Point(res[i].x1, res[i].y1), Point(res[i].x2, res[i].y2)),
                        Scalar(0, 0, 255), 3, 4, 0 );
        }
        writer << outcome;
        imshow("input", dynamic_frame);
        imshow("res", outcome);
        // filter in time domain
        double alpha = 0.7;
        last_frame = blur_frame*alpha + last_frame*(1 - alpha);

        int key = waitKey(1);
        if(key == 27)       // esc
            break;
    }

}

//int main()
//{
//    Mat m = imread("b.jpg", IMREAD_GRAYSCALE);
//    threshold(m, m, 100, 255, CV_8U);
//    namedWindow("fuck", WINDOW_NORMAL);
//    namedWindow("out", WINDOW_NORMAL);
//    imshow("fuck", m);

//    ofstream file("fff");
//    file << m;
//    file.close();

//    vector<Result> res = get_people_pos(m);

//    sort(res.begin(), res.end(), Result());


//    Mat o = Mat::zeros(m.size(), CV_8UC3);

//    for(int i = 0; i < res.size() && i < 5; i++)
//    {
//        rectangle(o, Rect(Point(res[i].x1, res[i].y1), Point(res[i].x2, res[i].y2)),
//                    Scalar(0, 255, 0), 1, 1, 0 );
//        cout << Rect(Point(res[i].x1, res[i].y1), Point(res[i].x2, res[i].y2)) << endl;

//        imshow("out", o);
//        waitKey();
//    }
//    imshow("out", o);
//    waitKey();
//    cout << res.size() << endl;
//}
