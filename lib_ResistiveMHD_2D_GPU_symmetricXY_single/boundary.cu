#include "boundary.hpp"


__global__
void symmetricBoundaryX2nd_kernel(ConservationParameter* U)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < device_ny) {
        U[j + 0 * device_ny].rho  = U[j + 5 * device_ny].rho;
        U[j + 0 * device_ny].rhoU = U[j + 5 * device_ny].rhoU;
        U[j + 0 * device_ny].rhoV = U[j + 5 * device_ny].rhoV;
        U[j + 0 * device_ny].rhoW = U[j + 5 * device_ny].rhoW;
        U[j + 0 * device_ny].bY   = U[j + 5 * device_ny].bY;
        U[j + 0 * device_ny].bZ   = U[j + 5 * device_ny].bZ;
        U[j + 0 * device_ny].e    = U[j + 5 * device_ny].e;
        U[j + 1 * device_ny].rho  = U[j + 4 * device_ny].rho;
        U[j + 1 * device_ny].rhoU = U[j + 4 * device_ny].rhoU;
        U[j + 1 * device_ny].rhoV = U[j + 4 * device_ny].rhoV;
        U[j + 1 * device_ny].rhoW = U[j + 4 * device_ny].rhoW;
        U[j + 1 * device_ny].bY   = U[j + 4 * device_ny].bY;
        U[j + 1 * device_ny].bZ   = U[j + 4 * device_ny].bZ;
        U[j + 1 * device_ny].e    = U[j + 4 * device_ny].e;
        U[j + 2 * device_ny].rho  = U[j + 3 * device_ny].rho;
        U[j + 2 * device_ny].rhoU = U[j + 3 * device_ny].rhoU;
        U[j + 2 * device_ny].rhoV = U[j + 3 * device_ny].rhoV;
        U[j + 2 * device_ny].rhoW = U[j + 3 * device_ny].rhoW;
        U[j + 2 * device_ny].bY   = U[j + 3 * device_ny].bY;
        U[j + 2 * device_ny].bZ   = U[j + 3 * device_ny].bZ;
        U[j + 2 * device_ny].e    = U[j + 3 * device_ny].e;


        U[j + (device_nx - 1) * device_ny].rho  = U[j + (device_nx - 6) * device_ny].rho;
        U[j + (device_nx - 1) * device_ny].rhoU = U[j + (device_nx - 6) * device_ny].rhoU;
        U[j + (device_nx - 1) * device_ny].rhoV = U[j + (device_nx - 6) * device_ny].rhoV;
        U[j + (device_nx - 1) * device_ny].rhoW = U[j + (device_nx - 6) * device_ny].rhoW;
        U[j + (device_nx - 1) * device_ny].bY   = U[j + (device_nx - 6) * device_ny].bY;
        U[j + (device_nx - 1) * device_ny].bZ   = U[j + (device_nx - 6) * device_ny].bZ;
        U[j + (device_nx - 1) * device_ny].e    = U[j + (device_nx - 6) * device_ny].e;
        U[j + (device_nx - 2) * device_ny].rho  = U[j + (device_nx - 5) * device_ny].rho;
        U[j + (device_nx - 2) * device_ny].rhoU = U[j + (device_nx - 5) * device_ny].rhoU;
        U[j + (device_nx - 2) * device_ny].rhoV = U[j + (device_nx - 5) * device_ny].rhoV;
        U[j + (device_nx - 2) * device_ny].rhoW = U[j + (device_nx - 5) * device_ny].rhoW;
        U[j + (device_nx - 2) * device_ny].bY   = U[j + (device_nx - 5) * device_ny].bY;
        U[j + (device_nx - 2) * device_ny].bZ   = U[j + (device_nx - 5) * device_ny].bZ;
        U[j + (device_nx - 2) * device_ny].e    = U[j + (device_nx - 5) * device_ny].e;
        U[j + (device_nx - 3) * device_ny].rho  = U[j + (device_nx - 4) * device_ny].rho;
        U[j + (device_nx - 3) * device_ny].rhoU = U[j + (device_nx - 4) * device_ny].rhoU;
        U[j + (device_nx - 3) * device_ny].rhoV = U[j + (device_nx - 4) * device_ny].rhoV;
        U[j + (device_nx - 3) * device_ny].rhoW = U[j + (device_nx - 4) * device_ny].rhoW;
        U[j + (device_nx - 3) * device_ny].bY   = U[j + (device_nx - 4) * device_ny].bY;
        U[j + (device_nx - 3) * device_ny].bZ   = U[j + (device_nx - 4) * device_ny].bZ;
        U[j + (device_nx - 3) * device_ny].e    = U[j + (device_nx - 4) * device_ny].e;
    }
}

__global__
void symmetricBoundaryX2ndBX_kernel(ConservationParameter* U)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (0 < j && j < device_ny) {
        U[j + 2 * device_ny].bX = U[j + 3 * device_ny].bX 
                                + (U[j + 3 * device_ny].bY - U[j - 1 + 3 * device_ny].bY) / device_dy * device_dx;
        U[j + 1 * device_ny].bX = U[j + 2 * device_ny].bX 
                                + (U[j + 2 * device_ny].bY - U[j - 1 + 2 * device_ny].bY) / device_dy * device_dx;
        U[j + 0 * device_ny].bX = U[j + 1 * device_ny].bX 
                                + (U[j + 1 * device_ny].bY - U[j - 1 + 1 * device_ny].bY) / device_dy * device_dx;
        
        U[j + (device_nx - 3) * device_ny].bX = U[j + (device_nx - 4) * device_ny].bX 
                                              - (U[j + (device_nx - 3) * device_ny].bY - U[j - 1 + (device_nx - 3) * device_ny].bY) / device_dy * device_dx;
        U[j + (device_nx - 2) * device_ny].bX = U[j + (device_nx - 3) * device_ny].bX 
                                              - (U[j + (device_nx - 2) * device_ny].bY - U[j - 1 + (device_nx - 2) * device_ny].bY) / device_dy * device_dx;
        U[j + (device_nx - 1) * device_ny].bX = U[j + (device_nx - 2) * device_ny].bX 
                                              - (U[j + (device_nx - 1) * device_ny].bY - U[j - 1 + (device_nx - 1) * device_ny].bY) / device_dy * device_dx;

    }

    if (j == 0) {
        U[j + 2 * device_ny].bX = U[j + 3 * device_ny].bX;
        U[j + 1 * device_ny].bX = U[j + 2 * device_ny].bX;
        U[j + 0 * device_ny].bX = U[j + 1 * device_ny].bX;
        
        U[j + (device_nx - 3) * device_ny].bX = U[j + (device_nx - 4) * device_ny].bX;
        U[j + (device_nx - 2) * device_ny].bX = U[j + (device_nx - 3) * device_ny].bX;
        U[j + (device_nx - 1) * device_ny].bX = U[j + (device_nx - 2) * device_ny].bX;

    }

}

void Boundary::symmetricBoundaryX2nd(
    thrust::device_vector<ConservationParameter>& U
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (ny + threadsPerBlock - 1) / threadsPerBlock;

    symmetricBoundaryX2nd_kernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(U.data()));

    cudaDeviceSynchronize();

    symmetricBoundaryX2ndBX_kernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(U.data()));

    cudaDeviceSynchronize();
}


__global__
void symmetricBoundaryY2nd_kernel(ConservationParameter* U)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < device_nx) {
        U[0 + i * device_ny].rho  = U[5 + i * device_ny].rho;
        U[0 + i * device_ny].rhoU = U[5 + i * device_ny].rhoU;
        U[0 + i * device_ny].rhoV = U[5 + i * device_ny].rhoV;
        U[0 + i * device_ny].rhoW = U[5 + i * device_ny].rhoW;
        U[0 + i * device_ny].bX   = U[5 + i * device_ny].bX;
        U[0 + i * device_ny].bZ   = U[5 + i * device_ny].bZ;
        U[0 + i * device_ny].e    = U[5 + i * device_ny].e;
        U[1 + i * device_ny].rho  = U[4 + i * device_ny].rho;
        U[1 + i * device_ny].rhoU = U[4 + i * device_ny].rhoU;
        U[1 + i * device_ny].rhoV = U[4 + i * device_ny].rhoV;
        U[1 + i * device_ny].rhoW = U[4 + i * device_ny].rhoW;
        U[1 + i * device_ny].bX   = U[4 + i * device_ny].bX;
        U[1 + i * device_ny].bZ   = U[4 + i * device_ny].bZ;
        U[1 + i * device_ny].e    = U[4 + i * device_ny].e;
        U[2 + i * device_ny].rho  = U[3 + i * device_ny].rho;
        U[2 + i * device_ny].rhoU = U[3 + i * device_ny].rhoU;
        U[2 + i * device_ny].rhoV = U[3 + i * device_ny].rhoV;
        U[2 + i * device_ny].rhoW = U[3 + i * device_ny].rhoW;
        U[2 + i * device_ny].bX   = U[3 + i * device_ny].bX;
        U[2 + i * device_ny].bZ   = U[3 + i * device_ny].bZ;
        U[2 + i * device_ny].e    = U[3 + i * device_ny].e;
        

        U[device_ny - 1 + i * device_ny].rho  = U[device_ny - 6 + i * device_ny].rho;
        U[device_ny - 1 + i * device_ny].rhoU = U[device_ny - 6 + i * device_ny].rhoU;
        U[device_ny - 1 + i * device_ny].rhoV = U[device_ny - 6 + i * device_ny].rhoV;
        U[device_ny - 1 + i * device_ny].rhoW = U[device_ny - 6 + i * device_ny].rhoW;
        U[device_ny - 1 + i * device_ny].bX   = U[device_ny - 6 + i * device_ny].bX;
        U[device_ny - 1 + i * device_ny].bZ   = U[device_ny - 6 + i * device_ny].bZ;
        U[device_ny - 1 + i * device_ny].e    = U[device_ny - 6 + i * device_ny].e;
        U[device_ny - 2 + i * device_ny].rho  = U[device_ny - 5 + i * device_ny].rho;
        U[device_ny - 2 + i * device_ny].rhoU = U[device_ny - 5 + i * device_ny].rhoU;
        U[device_ny - 2 + i * device_ny].rhoV = U[device_ny - 5 + i * device_ny].rhoV;
        U[device_ny - 2 + i * device_ny].rhoW = U[device_ny - 5 + i * device_ny].rhoW;
        U[device_ny - 2 + i * device_ny].bX   = U[device_ny - 5 + i * device_ny].bX;
        U[device_ny - 2 + i * device_ny].bZ   = U[device_ny - 5 + i * device_ny].bZ;
        U[device_ny - 2 + i * device_ny].e    = U[device_ny - 5 + i * device_ny].e;
        U[device_ny - 3 + i * device_ny].rho  = U[device_ny - 4 + i * device_ny].rho;
        U[device_ny - 3 + i * device_ny].rhoU = U[device_ny - 4 + i * device_ny].rhoU;
        U[device_ny - 3 + i * device_ny].rhoV = U[device_ny - 4 + i * device_ny].rhoV;
        U[device_ny - 3 + i * device_ny].rhoW = U[device_ny - 4 + i * device_ny].rhoW;
        U[device_ny - 3 + i * device_ny].bX   = U[device_ny - 4 + i * device_ny].bX;
        U[device_ny - 3 + i * device_ny].bZ   = U[device_ny - 4 + i * device_ny].bZ;
        U[device_ny - 3 + i * device_ny].e    = U[device_ny - 4 + i * device_ny].e;
    }
}

__global__
void symmetricBoundaryY2ndBY_kernel(ConservationParameter* U)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (0 < i && i < device_nx) {
        U[2 + i * device_ny].bY = U[3 + i * device_ny].bY
                                + (U[3 + i * device_ny].bX - U[3 + (i - 1) * device_ny].bX) / device_dx * device_dy;
        U[1 + i * device_ny].bY = U[2 + i * device_ny].bY
                                + (U[2 + i * device_ny].bX - U[2 + (i - 1) * device_ny].bX) / device_dx * device_dy;
        U[0 + i * device_ny].bY = U[1 + i * device_ny].bY
                                + (U[1 + i * device_ny].bX - U[1 + (i - 1) * device_ny].bX) / device_dx * device_dy;
        
        U[device_ny - 3 + i * device_ny].bY = U[device_ny - 4 + i * device_ny].bY
                                            - (U[device_ny - 3 + i * device_ny].bX - U[device_ny - 3 + (i - 1) * device_ny].bX) / device_dx * device_dy;
        U[device_ny - 2 + i * device_ny].bY = U[device_ny - 3 + i * device_ny].bY
                                            - (U[device_ny - 2 + i * device_ny].bX - U[device_ny - 2 + (i - 1) * device_ny].bX) / device_dx * device_dy;
        U[device_ny - 1 + i * device_ny].bY = U[device_ny - 2 + i * device_ny].bY
                                            - (U[device_ny - 1 + i * device_ny].bX - U[device_ny - 1 + (i - 1) * device_ny].bX) / device_dx * device_dy;
    }

    if (i == 0) {
        U[2 + i * device_ny].bY = U[3 + i * device_ny].bY;
        U[1 + i * device_ny].bY = U[2 + i * device_ny].bY;
        U[0 + i * device_ny].bY = U[1 + i * device_ny].bY;
        
        U[device_ny - 3 + i * device_ny].bY = U[device_ny - 4 + i * device_ny].bY;
        U[device_ny - 2 + i * device_ny].bY = U[device_ny - 3 + i * device_ny].bY;
        U[device_ny - 1 + i * device_ny].bY = U[device_ny - 2 + i * device_ny].bY;
    }
}


void Boundary::symmetricBoundaryY2nd(
    thrust::device_vector<ConservationParameter>& U
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (nx + threadsPerBlock - 1) / threadsPerBlock;

    symmetricBoundaryY2nd_kernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(U.data()));

    cudaDeviceSynchronize();

    symmetricBoundaryY2ndBY_kernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(U.data()));

    cudaDeviceSynchronize();
}

//////////////////////////////////////////////////

__global__
void periodicBoundaryX2nd_kernel(ConservationParameter* U)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < device_ny) {
        U[j + 0 * device_ny].rho  = U[j + (device_nx-6) * device_ny].rho;
        U[j + 0 * device_ny].rhoU = U[j + (device_nx-6) * device_ny].rhoU;
        U[j + 0 * device_ny].rhoV = U[j + (device_nx-6) * device_ny].rhoV;
        U[j + 0 * device_ny].rhoW = U[j + (device_nx-6) * device_ny].rhoW;
        U[j + 0 * device_ny].bX   = U[j + (device_nx-6) * device_ny].bX;
        U[j + 0 * device_ny].bY   = U[j + (device_nx-6) * device_ny].bY;
        U[j + 0 * device_ny].bZ   = U[j + (device_nx-6) * device_ny].bZ;
        U[j + 0 * device_ny].e    = U[j + (device_nx-6) * device_ny].e;
        U[j + 1 * device_ny].rho  = U[j + (device_nx-5) * device_ny].rho;
        U[j + 1 * device_ny].rhoU = U[j + (device_nx-5) * device_ny].rhoU;
        U[j + 1 * device_ny].rhoV = U[j + (device_nx-5) * device_ny].rhoV;
        U[j + 1 * device_ny].rhoW = U[j + (device_nx-5) * device_ny].rhoW;
        U[j + 1 * device_ny].bX   = U[j + (device_nx-5) * device_ny].bX;
        U[j + 1 * device_ny].bY   = U[j + (device_nx-5) * device_ny].bY;
        U[j + 1 * device_ny].bZ   = U[j + (device_nx-5) * device_ny].bZ;
        U[j + 1 * device_ny].e    = U[j + (device_nx-5) * device_ny].e;
        U[j + 2 * device_ny].rho  = U[j + (device_nx-4) * device_ny].rho;
        U[j + 2 * device_ny].rhoU = U[j + (device_nx-4) * device_ny].rhoU;
        U[j + 2 * device_ny].rhoV = U[j + (device_nx-4) * device_ny].rhoV;
        U[j + 2 * device_ny].rhoW = U[j + (device_nx-4) * device_ny].rhoW;
        U[j + 2 * device_ny].bX   = U[j + (device_nx-4) * device_ny].bX;
        U[j + 2 * device_ny].bY   = U[j + (device_nx-4) * device_ny].bY;
        U[j + 2 * device_ny].bZ   = U[j + (device_nx-4) * device_ny].bZ;
        U[j + 2 * device_ny].e    = U[j + (device_nx-4) * device_ny].e;

        U[j + (device_nx-3) * device_ny].rho  = U[j + 3 * device_ny].rho;
        U[j + (device_nx-3) * device_ny].rhoU = U[j + 3 * device_ny].rhoU;
        U[j + (device_nx-3) * device_ny].rhoV = U[j + 3 * device_ny].rhoV;
        U[j + (device_nx-3) * device_ny].rhoW = U[j + 3 * device_ny].rhoW;
        U[j + (device_nx-3) * device_ny].bX   = U[j + 3 * device_ny].bX;
        U[j + (device_nx-3) * device_ny].bY   = U[j + 3 * device_ny].bY;
        U[j + (device_nx-3) * device_ny].bZ   = U[j + 3 * device_ny].bZ;
        U[j + (device_nx-3) * device_ny].e    = U[j + 3 * device_ny].e;
        U[j + (device_nx-2) * device_ny].rho  = U[j + 4 * device_ny].rho;
        U[j + (device_nx-2) * device_ny].rhoU = U[j + 4 * device_ny].rhoU;
        U[j + (device_nx-2) * device_ny].rhoV = U[j + 4 * device_ny].rhoV;
        U[j + (device_nx-2) * device_ny].rhoW = U[j + 4 * device_ny].rhoW;
        U[j + (device_nx-2) * device_ny].bX   = U[j + 4 * device_ny].bX;
        U[j + (device_nx-2) * device_ny].bY   = U[j + 4 * device_ny].bY;
        U[j + (device_nx-2) * device_ny].bZ   = U[j + 4 * device_ny].bZ;
        U[j + (device_nx-2) * device_ny].e    = U[j + 4 * device_ny].e;
        U[j + (device_nx-1) * device_ny].rho  = U[j + 5 * device_ny].rho;
        U[j + (device_nx-1) * device_ny].rhoU = U[j + 5 * device_ny].rhoU;
        U[j + (device_nx-1) * device_ny].rhoV = U[j + 5 * device_ny].rhoV;
        U[j + (device_nx-1) * device_ny].rhoW = U[j + 5 * device_ny].rhoW;
        U[j + (device_nx-1) * device_ny].bX   = U[j + 5 * device_ny].bX;
        U[j + (device_nx-1) * device_ny].bY   = U[j + 5 * device_ny].bY;
        U[j + (device_nx-1) * device_ny].bZ   = U[j + 5 * device_ny].bZ;
        U[j + (device_nx-1) * device_ny].e    = U[j + 5 * device_ny].e;
    }
}

void Boundary::periodicBoundaryX2nd(
    thrust::device_vector<ConservationParameter>& U
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (ny + threadsPerBlock - 1) / threadsPerBlock;

    periodicBoundaryX2nd_kernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(U.data()));

    cudaDeviceSynchronize();
}


__global__
void periodicBoundaryY2nd_kernel(ConservationParameter* U)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < device_nx) {
        U[0 + i * device_ny].rho  = U[device_ny-6 + i * device_ny].rho;
        U[0 + i * device_ny].rhoU = U[device_ny-6 + i * device_ny].rhoU;
        U[0 + i * device_ny].rhoV = U[device_ny-6 + i * device_ny].rhoV;
        U[0 + i * device_ny].rhoW = U[device_ny-6 + i * device_ny].rhoW;
        U[0 + i * device_ny].bX   = U[device_ny-6 + i * device_ny].bX;
        U[0 + i * device_ny].bY   = U[device_ny-6 + i * device_ny].bY;
        U[0 + i * device_ny].bZ   = U[device_ny-6 + i * device_ny].bZ;
        U[0 + i * device_ny].e    = U[device_ny-6 + i * device_ny].e;
        U[1 + i * device_ny].rho  = U[device_ny-5 + i * device_ny].rho;
        U[1 + i * device_ny].rhoU = U[device_ny-5 + i * device_ny].rhoU;
        U[1 + i * device_ny].rhoV = U[device_ny-5 + i * device_ny].rhoV;
        U[1 + i * device_ny].rhoW = U[device_ny-5 + i * device_ny].rhoW;
        U[1 + i * device_ny].bX   = U[device_ny-5 + i * device_ny].bX;
        U[1 + i * device_ny].bY   = U[device_ny-5 + i * device_ny].bY;
        U[1 + i * device_ny].bZ   = U[device_ny-5 + i * device_ny].bZ;
        U[1 + i * device_ny].e    = U[device_ny-5 + i * device_ny].e;
        U[2 + i * device_ny].rho  = U[device_ny-4 + i * device_ny].rho;
        U[2 + i * device_ny].rhoU = U[device_ny-4 + i * device_ny].rhoU;
        U[2 + i * device_ny].rhoV = U[device_ny-4 + i * device_ny].rhoV;
        U[2 + i * device_ny].rhoW = U[device_ny-4 + i * device_ny].rhoW;
        U[2 + i * device_ny].bX   = U[device_ny-4 + i * device_ny].bX;
        U[2 + i * device_ny].bY   = U[device_ny-4 + i * device_ny].bY;
        U[2 + i * device_ny].bZ   = U[device_ny-4 + i * device_ny].bZ;
        U[2 + i * device_ny].e    = U[device_ny-4 + i * device_ny].e;

        U[device_ny-3 + i * device_ny].rho  = U[3 + i * device_ny].rho;
        U[device_ny-3 + i * device_ny].rhoU = U[3 + i * device_ny].rhoU;
        U[device_ny-3 + i * device_ny].rhoV = U[3 + i * device_ny].rhoV;
        U[device_ny-3 + i * device_ny].rhoW = U[3 + i * device_ny].rhoW;
        U[device_ny-3 + i * device_ny].bX   = U[3 + i * device_ny].bX;
        U[device_ny-3 + i * device_ny].bY   = U[3 + i * device_ny].bY;
        U[device_ny-3 + i * device_ny].bZ   = U[3 + i * device_ny].bZ;
        U[device_ny-3 + i * device_ny].e    = U[3 + i * device_ny].e;
        U[device_ny-2 + i * device_ny].rho  = U[4 + i * device_ny].rho;
        U[device_ny-2 + i * device_ny].rhoU = U[4 + i * device_ny].rhoU;
        U[device_ny-2 + i * device_ny].rhoV = U[4 + i * device_ny].rhoV;
        U[device_ny-2 + i * device_ny].rhoW = U[4 + i * device_ny].rhoW;
        U[device_ny-2 + i * device_ny].bX   = U[4 + i * device_ny].bX;
        U[device_ny-2 + i * device_ny].bY   = U[4 + i * device_ny].bY;
        U[device_ny-2 + i * device_ny].bZ   = U[4 + i * device_ny].bZ;
        U[device_ny-2 + i * device_ny].e    = U[4 + i * device_ny].e;
        U[device_ny-1 + i * device_ny].rho  = U[5 + i * device_ny].rho;
        U[device_ny-1 + i * device_ny].rhoU = U[5 + i * device_ny].rhoU;
        U[device_ny-1 + i * device_ny].rhoV = U[5 + i * device_ny].rhoV;
        U[device_ny-1 + i * device_ny].rhoW = U[5 + i * device_ny].rhoW;
        U[device_ny-1 + i * device_ny].bX   = U[5 + i * device_ny].bX;
        U[device_ny-1 + i * device_ny].bY   = U[5 + i * device_ny].bY;
        U[device_ny-1 + i * device_ny].bZ   = U[5 + i * device_ny].bZ;
        U[device_ny-1 + i * device_ny].e    = U[5 + i * device_ny].e;
    }
}

void Boundary::periodicBoundaryY2nd(
    thrust::device_vector<ConservationParameter>& U
)
{
    int threadsPerBlock = 256; 
    int blocksPerGrid = (nx + threadsPerBlock - 1) / threadsPerBlock; 

    periodicBoundaryY2nd_kernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(U.data()));

    cudaDeviceSynchronize();
}


