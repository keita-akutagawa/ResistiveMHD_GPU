#ifndef CONSERVATION_PARAMETER_STRUCT_H
#define CONSERVATION_PARAMETER_STRUCT_H


struct ConservationParameter
{
    double rho;
    double rhoU;
    double rhoV;
    double rhoW;
    double bX; 
    double bY;
    double bZ;
    double e;
    double psi; 

    __host__ __device__
    ConservationParameter() : 
        rho(0.0), 
        rhoU(0.0), 
        rhoV(0.0), 
        rhoW(0.0), 
        bX(0.0), 
        bY(0.0), 
        bZ(0.0), 
        e(0.0), 
        psi(0.0)
        {}
    
    __host__ __device__
    ConservationParameter& operator=(const ConservationParameter& other)
    {
        if (this != &other) {
            rho = other.rho; 
            rhoU = other.rhoU; 
            rhoV = other.rhoV; 
            rhoW = other.rhoW; 
            bX = other.bX; 
            bY = other.bY; 
            bZ = other.bZ; 
            e = other.e; 
            psi = other.psi;
        }
        return *this;
    }
};

#endif
