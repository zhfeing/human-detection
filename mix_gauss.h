#ifndef MIX_GAUSS_H
#define MIX_GAUSS_H

class Bad_UseWithoutInitial{};

class GaussModule
{
public:
    GaussModule(float mu, float sigma_square);
    float pr(float x);
    float confidence_level(float x);
    void update_module(float x_n, float alpha);
    bool in_three_sigma_region(float x);
private:
    void update_mu(float x_n, float alpha);
    void update_sigma(float x_n, float alpha);
private:
    float mu, sigma_square, mx_square;
    const static float Pi;
};

class MixGauss
{    
public:
    enum Type{BG, PR, UK};  // background, person, unknow
public:
    MixGauss();
    ~MixGauss();
    void update_module(float x_n);
    void initial_background(float mu, float sigma_square);
    void initial_people(float mu, float sigma_square);
    float confidence_level();
    Type type();
private:
    float increase_p();
    bool is_background(float x);
private:
    float p;           // probability that a the module is pure background
    bool people_initial;
    bool background_initial;
    GaussModule *m_b;   // background module
    GaussModule *m_p;   // people module
    Type ty;
    float confidence;
};

#endif // MIX_GAUSS_H
