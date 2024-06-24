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


__global__ void addResistiveTermToFluxF_kernel(
    const ConservationParameter* U, Flux* flux)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < device_nx - 1) && (0 < j) && (j < device_ny - 1)) {
        double xPosition = i * device_dx, yPosition = j * device_dy;

        double bXPlus1, bXMinus1, bYPlus1, bY, bYMinus1, bZPlus1, bZ, bZMinus1;
        double currentY, currentZ;
        double eta;

        bZPlus1 = U[j + (i + 1) * device_ny].bZ;
        bZMinus1 = U[j + (i - 1) * device_ny].bZ;
        currentY = -(bZPlus1 - bZMinus1) / (2.0 * device_dx);

        bYPlus1 = U[j + (i + 1) * device_ny].bY;
        bYMinus1 = U[j + (i - 1) * device_ny].bY;
        bXPlus1 = U[j + 1 + i * device_ny].bX;
        bXMinus1 = U[j - 1 + i * device_ny].bX;
        currentZ = (bYPlus1 - bYMinus1) / (2.0 * device_dx)
                 - (bXPlus1 - bXMinus1) / (2.0 * device_dy);
        
        eta = FluxSolver::getEta(xPosition, yPosition);
  
        flux[j + i * device_ny].f5 -= eta * currentZ;
        flux[j + i * device_ny].f6 += eta * currentY;

        bY = U[j + i * device_ny].bY;
        bZ = U[j + i * device_ny].bZ;
        flux[j + i * device_ny].f7 += eta * (currentY * bZ - currentZ * bY);
    }
}

void FluxSolver::addResistiveTermToFluxF(
    const thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    addResistiveTermToFluxF_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(flux.data())
    );

    cudaDeviceSynchronize();
}


__global__ void addResistiveTermToFluxG_kernel(
    const ConservationParameter* U, Flux* flux)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < device_nx - 1) && (0 < j) && (j < device_ny - 1)) {
        double xPosition = i * device_dx, yPosition = j * device_dy;

        double bXPlus1, bX, bXMinus1, bYPlus1, bYMinus1, bZPlus1, bZ, bZMinus1;
        double currentX, currentZ;
        double eta;

        bZPlus1 = U[j + 1 + i * device_ny].bZ;
        bZMinus1 = U[j - 1 + i * device_ny].bZ;
        currentX = (bZPlus1 - bZMinus1) / (2.0 * device_dy);

        bYPlus1 = U[j + (i + 1) * device_ny].bY;
        bYMinus1 = U[j + (i - 1) * device_ny].bY;
        bXPlus1 = U[j + 1 + i * device_ny].bX;
        bXMinus1 = U[j - 1 + i * device_ny].bX;
        currentZ = (bYPlus1 - bYMinus1) / (2.0 * device_dx)
                 - (bXPlus1 - bXMinus1) / (2.0 * device_dy);
        
        eta = FluxSolver::getEta(xPosition, yPosition);
  
        flux[j + i * device_ny].f4 += eta * currentZ;
        flux[j + i * device_ny].f6 -= eta * currentX;

        bX = U[j + i * device_ny].bX;
        bZ = U[j + i * device_ny].bZ;
        flux[j + i * device_ny].f7 += eta * (currentZ * bX - currentX * bZ);
    }
}

void FluxSolver::addResistiveTermToFluxG(
    const thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    addResistiveTermToFluxG_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(flux.data())
    );

    cudaDeviceSynchronize();
}
