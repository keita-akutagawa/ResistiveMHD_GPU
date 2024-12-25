#include "flux_solver.hpp"


FluxSolver::FluxSolver(MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo), 
      hLLD(mPIInfo)
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
    const ConservationParameter* U, Flux* flux, 
    int localSizeX, int localSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX - 1) && (0 < j) && (j < localSizeY - 1)) {
        int index = j + i * localSizeY;

        double xPosition = (i + 0.5) * device_dx, yPosition = j * device_dy;
        double jY, jZ;
        double eta, etaJY, etaJZ, etaJYBZ, etaJZBY;

        jY = -(U[index + localSizeY].bZ - U[index - localSizeY].bZ) / (2.0 * device_dx);
        jZ = (U[index + localSizeY].bY - U[index - localSizeY].bY) / (2.0 * device_dx)
           - (U[index + 1].bX - U[index - 1].bX) / (2.0 * device_dy);
           
        eta = getEta(xPosition, yPosition);
        etaJY = eta * jY; 
        etaJZ = eta * jZ;
        etaJYBZ = etaJY * U[index].bZ;
        etaJZBY = etaJZ * U[index].bY;
  
        flux[index].f5 -= etaJZ;
        flux[index].f6 += etaJY;
        flux[index].f7 += etaJYBZ - etaJZBY;
    }
}

void FluxSolver::addResistiveTermToFluxF(
    const thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    addResistiveTermToFluxF_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(flux.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
    );
    cudaDeviceSynchronize();
}


__global__ void addResistiveTermToFluxG_kernel(
    const ConservationParameter* U, Flux* flux, 
    int localSizeX, int localSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX - 1) && (0 < j) && (j < localSizeY - 1)) {
        int index = j + i * localSizeY;

        double xPosition = i * device_dx, yPosition = (j + 0.5) * device_dy;
        double jX, jZ;
        double eta, etaJX, etaJZ, etaJZBX, etaJXBZ;

        jX = (U[index + 1].bZ - U[index - 1].bZ) / (2.0 * device_dy);
        jZ = (U[index + localSizeY].bY - U[index - localSizeY].bY) / (2.0 * device_dx)
           - (U[index + 1].bX - U[index - 1].bX) / (2.0 * device_dy); 
        
        eta = getEta(xPosition, yPosition);
        etaJX = eta * jX;
        etaJZ = eta * jZ;
        etaJXBZ = etaJX * U[index].bZ;
        etaJZBX = etaJZ * U[index].bX; 
  
        flux[index].f4 += etaJZ;
        flux[index].f6 -= etaJX;
        flux[index].f7 += etaJZBX - etaJXBZ;
    }
}

void FluxSolver::addResistiveTermToFluxG(
    const thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    addResistiveTermToFluxG_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(flux.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
    );
    cudaDeviceSynchronize();
}
