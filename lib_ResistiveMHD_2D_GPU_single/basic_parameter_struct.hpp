#ifndef BASIC_PARAMETER_STRUCT_H
#define BASIC_PARAMETER_STRUCT_H


struct BasicParameter
{
    float rho;
    float u;
    float v;
    float w;
    float bX; 
    float bY;
    float bZ;
    float p;

    __host__ __device__
    BasicParameter() : 
        rho(0.0f), 
        u(0.0f), 
        v(0.0f), 
        w(0.0f), 
        bX(0.0f), 
        bY(0.0f), 
        bZ(0.0f), 
        p(0.0f)
        {}
};

#endif
