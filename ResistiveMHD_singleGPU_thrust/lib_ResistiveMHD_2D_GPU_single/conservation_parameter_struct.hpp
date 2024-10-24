#ifndef CONSERVATION_PARAMETER_STRUCT_H
#define CONSERVATION_PARAMETER_STRUCT_H


struct ConservationParameter
{
    float rho;
    float rhoU;
    float rhoV;
    float rhoW;
    float bX; 
    float bY;
    float bZ;
    float e;

    __host__ __device__
    ConservationParameter() : 
        rho(0.0f), 
        rhoU(0.0f), 
        rhoV(0.0f), 
        rhoW(0.0f), 
        bX(0.0f), 
        bY(0.0f), 
        bZ(0.0f), 
        e(0.0f)
        {}
};

#endif
