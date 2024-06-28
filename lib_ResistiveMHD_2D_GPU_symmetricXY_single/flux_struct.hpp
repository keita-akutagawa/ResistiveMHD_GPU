#ifndef Flux_STRUCT_H
#define Flux_STRUCT_H


struct Flux
{
    float f0;
    float f1;
    float f2;
    float f3;
    float f4;
    float f5;
    float f6;
    float f7;

    __host__ __device__
    Flux() : 
        f0(0.0f), 
        f1(0.0f),
        f2(0.0f),
        f3(0.0f),
        f4(0.0f),
        f5(0.0f),
        f6(0.0f),
        f7(0.0f)
        {}
};

#endif