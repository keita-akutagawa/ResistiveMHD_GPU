#include "flux_solver.hpp"


FluxSolver::FluxSolver()
    : flux(nx * ny), 
      currentX(nx * ny), 
      currentY(nx * ny), 
      currentZ(nx * ny)
{
}


thrust::device_vector<Flux> FluxSolver::getFluxF(
    const thrust::device_vector<ConservationParameter>& U
)
{
    hLLD.calculateFluxF(U);
    flux = hLLD.getFlux();

    return flux;
}


thrust::device_vector<Flux> FluxSolver::getFluxG(
    const thrust::device_vector<ConservationParameter>& U
)
{
    hLLD.calculateFluxG(U);
    flux = hLLD.getFlux();

    return flux;
}


