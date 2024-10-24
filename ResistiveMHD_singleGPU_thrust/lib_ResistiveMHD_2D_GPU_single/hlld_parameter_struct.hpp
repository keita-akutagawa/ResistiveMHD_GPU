#ifndef HLLD_PARAMETER_STRUCT_H
#define HLLD_PARAMETER_STRUCT_H


struct HLLDParameter
{
    float pTL;
    float pTR;
    float eL;
    float eR;
    float csL;
    float csR;
    float caL;
    float caR;
    float vaL;
    float vaR;
    float cfL;
    float cfR;

    float SL;
    float SR;
    float SM;

    float rho1L;
    float rho1R;
    float u1L;
    float u1R;
    float v1L;
    float v1R;
    float w1L;
    float w1R;
    float bY1L;
    float bY1R;
    float bZ1L;
    float bZ1R;
    float e1L;
    float e1R; 
    float pT1L;
    float pT1R;

    float S1L;
    float S1R;

    float rho2L;
    float rho2R;
    float u2;
    float v2;
    float w2; 
    float bY2; 
    float bZ2;
    float e2L;
    float e2R;
    float pT2L;
    float pT2R;
    

    __host__ __device__
    HLLDParameter() :
        pTL(0.0f),
        pTR(0.0f),
        eL(0.0f),
        eR(0.0f),
        csL(0.0f),
        csR(0.0f),
        caL(0.0f),
        caR(0.0f),
        vaL(0.0f),
        vaR(0.0f),
        cfL(0.0f),
        cfR(0.0f),

        SL(0.0f),
        SR(0.0f),
        SM(0.0f),

        rho1L(0.0f),
        rho1R(0.0f),
        u1L(0.0f),
        u1R(0.0f),
        v1L(0.0f),
        v1R(0.0f),
        w1L(0.0f),
        w1R(0.0f),
        bY1L(0.0f),
        bY1R(0.0f),
        bZ1L(0.0f),
        bZ1R(0.0f),
        e1L(0.0f),
        e1R(0.0f), 
        pT1L(0.0f),
        pT1R(0.0f),

        S1L(0.0f),
        S1R(0.0f),

        rho2L(0.0f),
        rho2R(0.0f),
        u2(0.0f),
        v2(0.0f),
        w2(0.0f), 
        bY2(0.0f), 
        bZ2(0.0f),
        e2L(0.0f),
        e2R(0.0f),
        pT2L(0.0f),
        pT2R(0.0f)
        {}
};

#endif
