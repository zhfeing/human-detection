#include <mix_gauss_detection.h>
#include <iostream>
#include <vector>
#include <algorithm>

// class GaussModule
const float GaussModule::PI = 3.1415926f;
GaussModule::GaussModule(int rows, int cols)
    :rows(rows), cols(cols), init(false)
{
}
void GaussModule::init_parameters(const _CV Mat &mu, const _CV Mat &sigma_square)
{
    check_size(mu);
    check_size(sigma_square);
    mu.copyTo(this->mu);
    sigma_square.copyTo(this->sigma_square);
//    _STD cout << mx_square.type() << " " << mu.type() << _STD endl;
    mx_square = mu.mul(mu) + sigma_square;
    init = true;
}
void GaussModule::update_parameter(const _CV Mat &x_n, const _CV Mat &alpha)
{
    check_init();
    check_size(x_n);
    check_size(alpha);
    mu = mu.mul(alpha) + x_n.mul(1 - alpha);
    mx_square = mx_square.mul(alpha) + x_n.mul(x_n).mul(1 - alpha);
    sigma_square = _CV max(mx_square - mu.mul(mu), 1);
}
_CV Mat GaussModule::pr(const _CV Mat x)
{
    check_init();
    check_size(x);

    _CV Mat t1;     // 1/(2*sigma_square)
    _CV divide(1.0f, 2*sigma_square, t1);

    _CV Mat t2;
    _CV sqrt(t1/PI, t2);

    _CV Mat t3;
    _CV exp(-t1.mul((x - mu).mul(x - mu)), t3);

    _CV Mat res = t2.mul(t3);
    res.convertTo(res, CV_32F);
    return res;
}
_CV Mat GaussModule::confidence_level(const _CV Mat x)
{
    check_init();

    _CV Mat t1;     // 1/(2*sigma_square)
    _CV divide(1.0f, 2*sigma_square, t1);

    _CV Mat t2;
    _CV exp(-t1.mul((x - mu).mul(x - mu)), t2);
    t2.convertTo(t2, CV_32F);
    return t2;
}
_CV Mat GaussModule::within_three_sigma(const _CV Mat x)
{
    _CV Mat sigma;
    _CV sqrt(sigma_square, sigma);
    _CV Mat res = 3*sigma - cv::abs(x - mu);
    res.convertTo(res, CV_32F);
    return res;
}
inline void GaussModule::check_init()
{
    if(!init)
        throw Bad_UseWithoutInitial();
}
inline void GaussModule::check_size(const _CV Mat &src)
{
    if(src.rows != rows || src.cols != cols)
        throw Bad_SizeNotFit();
}

// class MixGaussModule

const uint8_t MixGaussModule::BG = 0;
const uint8_t MixGaussModule::PR = 1;
const uint8_t MixGaussModule::UK = 2;

MixGaussModule::MixGaussModule(int rows, int cols)
    :rows(rows), cols(cols),
     p(rows, cols, CV_32F, _CV Scalar::all(0.5f)),
     m_p(rows, cols), m_b(rows, cols),
     confidence_level(rows, cols, CV_32F),
     ty(rows, cols, CV_8U, _CV Scalar::all(0)),
     alpha_p(rows, cols, CV_32F),
     alpha_b(rows, cols, CV_32F)
{
}
void MixGaussModule::init_people(const _CV Mat &mu, const _CV Mat &sigma_square)
{
    m_p.init_parameters(mu, sigma_square);
}
void MixGaussModule::init_background(const _CV Mat &mu, const _CV Mat &sigma_square)
{
    m_b.init_parameters(mu, sigma_square);
}
void MixGaussModule::update_module(const _CV Mat x_n)
{
    confidence_level = m_p.confidence_level(x_n);

    _CV Mat isBackground = p.mul(m_b.pr(x_n)) - m_p.pr(x_n).mul(1 - p);
    _CV Mat bg_within_three_sigma = m_b.within_three_sigma(x_n);
    _CV Mat pr_within_three_sigma = m_p.within_three_sigma(x_n);

    _CV MatConstIterator_<float> iter_isBackground = isBackground.begin<float>();
    _CV MatConstIterator_<float> iter_bg_within_three_sigma = bg_within_three_sigma.begin<float>();
    _CV MatConstIterator_<float> iter_pr_within_three_sigma = pr_within_three_sigma.begin<float>();
//    _CV MatConstIterator_<float> iter_x_n = x_n.begin<float>();

    _CV MatIterator_<float> iter_alpha_b = alpha_b.begin<float>();
    _CV MatIterator_<float> iter_alpha_p = alpha_p.begin<float>();
    _CV MatIterator_<float> iter_p = p.begin<float>();
    _CV MatIterator_<float> iter_confidence_level = confidence_level.begin<float>();
    _CV MatIterator_<uint8_t> iter_ty = ty.begin<uint8_t>();

    const _CV MatConstIterator_<float> isBackground_end = isBackground.end<float>();


    while(iter_isBackground != isBackground_end)
    {
        if(*iter_isBackground > 0 && *iter_bg_within_three_sigma > 0)
        {
            *iter_p = increase_p(*iter_p);
            *iter_ty = BG;
            *iter_confidence_level = 0;
            *iter_alpha_p = 1;
            *iter_alpha_b = 0;
        }
        else
        {
            if(*iter_pr_within_three_sigma > 0)
            {
                *iter_p = 1 - increase_p(*iter_p);
                *iter_ty = PR;
                // already has confidence level
                *iter_alpha_p = 1;
                *iter_alpha_b = 0;
            }
            else
            {
                *iter_ty = UK;
                *iter_confidence_level = 0;
                *iter_alpha_p = 1;
                *iter_alpha_b = 0;
            }
        }

        iter_isBackground++;
        iter_bg_within_three_sigma++;
        iter_pr_within_three_sigma++;
//        iter_x_n++;
        iter_alpha_b++;
        iter_alpha_p++;
        iter_p++;
        iter_confidence_level++;
        iter_ty++;
    }
    m_b.update_parameter(x_n, alpha_b);
    m_p.update_parameter(x_n, alpha_p);
}
float MixGaussModule::increase_p(float p)
{
    return sqrt(1 - (p - 1)*(p - 1));
}
_CV Mat MixGaussModule::type()
{
    return ty.clone();
}
_CV Mat MixGaussModule::confidence()
{
    return confidence_level.clone();
}

// class MixGaussDetection

MixGaussDetection::MixGaussDetection(int rows, int cols)
    :rows(rows), cols(cols), init(false),
     R_module(rows, cols), G_module(rows, cols), B_module(rows, cols),
     pred(rows, cols, CV_32F, _CV Scalar::all(0))
{
}
void MixGaussDetection::init_parameters(const _CV Mat &x_0)
{
    _STD vector<_CV Mat> RGB_pic;
    cv::split(x_0, RGB_pic);
    B_module.init_background(RGB_pic[0], _CV Mat(rows, cols, CV_32F, _CV Scalar::all(2.5f)));
    B_module.init_people(_CV Mat(rows, cols, CV_32F, _CV Scalar::all(0.0f)),
                         _CV Mat(rows, cols, CV_32F, _CV Scalar::all(0.001f)));
    G_module.init_background(RGB_pic[1], _CV Mat(rows, cols, CV_32F, _CV Scalar::all(2.5f)));
    G_module.init_people(_CV Mat(rows, cols, CV_32F, _CV Scalar::all(0.0f)),
                         _CV Mat(rows, cols, CV_32F, _CV Scalar::all(0.001f)));
    R_module.init_background(RGB_pic[2], _CV Mat(rows, cols, CV_32F, _CV Scalar::all(2.5f)));
    R_module.init_people(_CV Mat(rows, cols, CV_32F, _CV Scalar::all(0.0f)),
                         _CV Mat(rows, cols, CV_32F, _CV Scalar::all(0.001f)));
    init = true;
}
void MixGaussDetection::update_module(const _CV Mat &x_n)
{
    _STD vector<_CV Mat> RGB_pic;
    cv::split(x_n, RGB_pic);
    B_module.update_module(RGB_pic[0]);
    G_module.update_module(RGB_pic[1]);
    R_module.update_module(RGB_pic[2]);

    _CV Mat B_type = B_module.type();
    _CV Mat G_type = G_module.type();
    _CV Mat R_type = R_module.type();

    _CV Mat B_conf = B_module.confidence();
    _CV Mat G_conf = G_module.confidence();
    _CV Mat R_conf = R_module.confidence();

    _CV Mat people = (B_type == _CV Mat(rows, cols, CV_8U, _CV Scalar::all(MixGaussModule::PR))) +
                     (G_type == _CV Mat(rows, cols, CV_8U, _CV Scalar::all(MixGaussModule::PR))) +
                     (R_type == _CV Mat(rows, cols, CV_8U, _CV Scalar::all(MixGaussModule::PR)));
    cv::threshold(people, pred, 0, 255, CV_THRESH_BINARY);
//    pred = people/3.0*255;
}
_CV Mat MixGaussDetection::get_pred()
{
    return pred.clone();
}
