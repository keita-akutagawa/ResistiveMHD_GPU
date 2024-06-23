#include "flux_solver.hpp"


FluxSolver::FluxSolver()
    : flux(nx * ny)
{
}


thrust::device_vector<Flux> FluxSolver::getFluxF(
    const thrust::device_vector<ConservationParameter>& U
)
{
    hLLD.calculateFluxF(U);
    flux = hLLD.getFlux();

    addResistiveTermToFluxF(U);

    return flux;
}


thrust::device_vector<Flux> FluxSolver::getFluxG(
    const thrust::device_vector<ConservationParameter>& U
)
{
    hLLD.calculateFluxG(U);
    flux = hLLD.getFlux();
    
    addResistiveTermToFluxG(U);

    return flux;
}


