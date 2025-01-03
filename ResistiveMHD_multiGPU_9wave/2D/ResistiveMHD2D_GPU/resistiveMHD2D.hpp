#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <string>
#include "flux_solver.hpp"
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

public:
    ResistiveMHD2D(MPIInfo& mPIInfo);

    virtual void initializeU(); 

    void oneStepRK2();

    void oneStepRK2_periodicXWallY();

    void oneStepRK2_periodicXSymmetricY();

    void save(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    thrust::device_vector<ConservationParameter>& getU();

    void calculateDt();

    bool checkCalculationIsCrashed();

private:

};



