#include <mix_gauss.h>
#include <cmath>
#include <iostream>

using namespace std;

// class GaussModule
const float GaussModule::Pi = 3.1415926535898;

GaussModule::GaussModule(float mu, float sigma_square)
    :mu(mu), sigma_square(sigma_square), mx_square(mu*mu + sigma_square)
{
}
inline float GaussModule::pr(float x)
{
    return exp(-(x - mu)*(x - mu)/(2*sigma_square))/sqrt(2*Pi*sigma_square);
}
inline float GaussModule::confidence_level(float x)
{
    return exp(-(x - mu)*(x - mu)/(2*sigma_square));
}
inline void GaussModule::update_module(float x_n, float alpha)
{
    update_mu(x_n, alpha);
    update_sigma(x_n, alpha);
//    cout << "mu " << mu << "\t sigma" << sigma_square << endl;
}
inline void GaussModule::update_mu(float x_n, float alpha)
{
    mu = alpha*mu + (1 - alpha)*x_n;
}
inline void GaussModule::update_sigma(float x_n, float alpha)
{
    mx_square = alpha*mx_square + (1 - alpha)*x_n*x_n;
    sigma_square = fmax(mx_square - mu*mu, 0.5);
}
inline bool GaussModule::in_three_sigma_region(float x)
{
    if(fabs(x - mu) < 3*sqrt(sigma_square))
        return true;
    else
        return false;
}

// class MixGauss

MixGauss::MixGauss()
    :p(0.5), people_initial(false), background_initial(false), ty(BG)
{
}
MixGauss::~MixGauss()
{
    if(people_initial)
        delete m_p;
    if(background_initial)
        delete m_b;
}
void MixGauss::update_module(float x_n)
{
//    if(!people_initial || !background_initial)
//        throw Bad_UseWithoutInitial();

    if(/*is_background(x_n) && m_b->in_three_sigma_region(x_n)*/true)
    {
        //p = increase_p();
        ty = BG;
        m_b->update_module(x_n, 0.8);     // set momentum small
        confidence = m_b->confidence_level(x_n);
    }
    else
    {
//        if(m_p->in_three_sigma_region(x_n))
//        {
            //p = 1 - increase_p();
            ty = PR;
            confidence = m_p->confidence_level(x_n);
//        }
//        else
//        {
//            ty = UK;
//            confidence = 0;
//        }
        m_p->update_module(x_n, 0.95);
        m_b->update_module(x_n, 0.99);     // set momentum large
    }
}
inline bool MixGauss::is_background(float x)
{
    return p*m_b->pr(x) > (1 - p)*m_p->pr(x);
}
inline float MixGauss::confidence_level()
{
    return confidence;
}
void MixGauss::initial_background(float mu, float sigma_square)
{
    if(!background_initial)
    {
        m_b = new GaussModule(mu, sigma_square);
        background_initial = true;
    }
}
void MixGauss::initial_people(float mu, float sigma_square)
{
    if(!people_initial)
    {
        m_p = new GaussModule(mu, sigma_square);
        people_initial = true;
    }
}
float MixGauss::increase_p()
{
    return sqrt(1 - (1 - p)*(1 - p));
}
MixGauss::Type MixGauss::type()
{
    return ty;
}
