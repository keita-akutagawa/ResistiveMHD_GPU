#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <thrust/extrema.h>
#include "const.hpp"
#include "flux_solver.hpp"
#include "ct.hpp"
#include "boundary.hpp"
#include "mpi.hpp"


class ResistiveMHD2D
{
private:
    MPIInfo mPIInfo; 
    MPIInfo* device_mPIInfo; 

    FluxSolver fluxSolver;
    
    thrust::device_vector<Flux> fluxF;
    thrust::device_vector<Flux> fluxG;
    thrust::device_vector<ConservationParameter> U;
    thrust::device_vector<ConservationParameter> UBar;
    thrust::device_vector<double> dtVector;
    thrust::device_vector<double> bXOld;
    thrust::device_vector<double> bYOld;
    thrust::device_vector<double> tmpVector;
    thrust::host_vector<ConservationParameter> hU;

    Boundary boundary;
    CT ct;

public:
    ResistiveMHD2D(MPIInfo& mPIInfo);

    virtual void initializeU(); 

    void oneStepRK2();

    void oneStepRK2_periodicXWallY();

    void oneStepRK2_periodicXSymmetricY();

    void oneStepRK2_flareXWallY(); //左は磁力線が刺さる壁。右は対称境界

    void save(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    thrust::device_vector<ConservationParameter>& getU();

    void calculateDt();

    bool checkCalculationIsCrashed();

private:
    void shiftUToCenterForCT(
        thrust::device_vector<ConservationParameter>& U
    );

    void backUToCenterHalfForCT(
        thrust::device_vector<ConservationParameter>& U
    );
};



