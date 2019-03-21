#ifndef MIX_GAUSS_DETECTION_H
#define MIX_GAUSS_DETECTION_H

#include <opencv2/opencv.hpp>
#include <vector>

#define _CV cv::
#define _STD std::

class Bad_UseWithoutInitial{};
class Bad_SizeNotFit{};

class GaussModule
{
    // all Mat in this module is one channel float Mat
public:
    GaussModule(int rows, int cols);
    void init_parameters(const _CV Mat &mu, const _CV Mat &sigma_square);
    void update_parameter(const _CV Mat &x_n, const _CV Mat &alpha);
    _CV Mat pr(const _CV Mat x);
    _CV Mat confidence_level(const _CV Mat x);
    _CV Mat within_three_sigma(const _CV Mat x);    // return float matrix and >0 means within 3 sigma
private:
    void check_init();
    void check_size(const cv::Mat &src);
private:
    _CV Mat mu;
    _CV Mat mx_square;
    _CV Mat sigma_square;
    int rows, cols;
    bool init;
    static const float PI;
};

class MixGaussModule
{
public:
    MixGaussModule(int rows, int cols);
    void init_people(const _CV Mat &mu, const _CV Mat &sigma_square);
    void init_background(const _CV Mat &mu, const _CV Mat &sigma_square);
    void update_module(const _CV Mat x_n);
    _CV Mat type();
    _CV Mat confidence();
public:
    static const uint8_t BG;
    static const uint8_t PR;
    static const uint8_t UK;

private:
    float increase_p(float p);
private:
    int rows, cols;
    _CV Mat p;          // probability that a pixel is background
    GaussModule m_p;    // people module
    GaussModule m_b;    // background module
    _CV Mat confidence_level;         // confidence level of people
    _CV Mat ty;         // one channel int matrix 0: BG, 1: PR, 2: UK
    _CV Mat alpha_p;    // update rate matrix
    _CV Mat alpha_b;    // update rate matrix
};

class MixGaussDetection
{
public:
    MixGaussDetection(int rows, int cols);
    void init_parameters(const _CV Mat &x_0);
    void update_module(const _CV Mat &x_n);
    _CV Mat get_pred();
private:
    int rows, cols;
    bool init;
    MixGaussModule R_module, G_module, B_module;
    _CV Mat pred;
};

#endif // MIX_GAUSS_DETECTION_H
