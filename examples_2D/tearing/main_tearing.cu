#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "../../lib_ResistiveMHD_2D_GPU_single/const.hpp"
#include "../../lib_ResistiveMHD_2D_GPU_single/resistiveMHD_2D.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>


std::string directoryname = "results_tearing_1e-6";
std::string filenameWithoutStep = "tearing";
std::ofstream logfile("results_tearing_1e-6/log_tearing.txt");

const int totalStep = 20000;
const int recordStep = 500;
float totalTime = 0.0f;


const float EPS = 1.0e-20f;
const float PI = 3.141592653f;

const float gamma_mhd = 5.0f / 3.0f;

const float sheatThickness = 1.0f;
const float betaUpstream = 2.0f;
const float rho0 = 1.0f;
const float b0 = 1.0f;
const float p0 = b0 * b0 / 2.0f;
const float VA = b0 / sqrt(rho0);
const float alfvenTime = sheatThickness / VA;

const float eta0 = 0.0f;
const float eta1 = 1e-6f * pow(2.0f * PI, 4.0f);
float eta = eta0 + eta1;
const float triggerRatio = 0.01f;

const float xmin = 0.0f;
const float xmax = 5.0f * 2.0f * PI * sheatThickness / pow(eta, 1.0f / 4.0f);
const float dx = sheatThickness / 16.0f;
const int nx = int((xmax - xmin) / dx);
const float ymin = 0.0f;
const float ymax = 20.0f * sheatThickness;
const float dy = sheatThickness / 16.0f;
const int ny = int((ymax - ymin) / dy);

const float CFL = 0.7f;
float dt = 0.0f;

__constant__ float device_EPS;
__constant__ float device_PI;

__constant__ float device_dx;
__constant__ float device_xmin;
__constant__ float device_xmax;
__constant__ int device_nx;

__constant__ float device_dy;
__constant__ float device_ymin;
__constant__ float device_ymax;
__constant__ int device_ny;

__constant__ float device_CFL;
__constant__ float device_gamma_mhd;

__device__ float device_dt;
__device__ float device_totalTime;

__constant__ float device_sheatThickness;
__constant__ float device_betaUpstream;
__constant__ float device_rho0;
__constant__ float device_b0;
__constant__ float device_p0;
__constant__ float device_VA;
__constant__ float device_alfvenTime;

__constant__ float device_eta0;
__constant__ float device_eta1;
__device__ float device_eta;

__constant__ float device_triggerRatio;

int step;
__device__ int device_step;


__global__ void initializeU_kernel(ConservationParameter* U) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx && j < device_ny) {

        float rho, u, v, w, bX, bY, bZ, e, p;
        float bXHalf, bYHalf;
        float x = i * device_dx, y = j * device_dy;
        float xHalf = (i + 0.5f) * device_dx, yHalf = (j + 0.5f) * device_dy;
        float xCenter = 0.5f * (device_xmax - device_xmin), yCenter = 0.5f * (device_ymax - device_ymin);
        float xi, phi, kmax;
        curandState state; 
        curand_init(0, j + device_ny * i, 0, &state);
        phi = 1.0f * device_PI * (curand_uniform(&state) - 1.0f);
        xi = 0.1f * (y - yCenter) / device_sheatThickness;
        kmax = pow(device_eta, 1.0f / 4.0f) / device_sheatThickness;
        
        rho = device_rho0;
        u = device_triggerRatio * device_VA * (2.0f * xi * tanh(xi) - pow(1.0f / cosh(xi), 2.0f))
          * exp(-pow(xi, 2.0f)) / kmax * sin(kmax * i * device_dx + phi);
        v = device_triggerRatio * device_VA * tanh(xi) * exp(-pow(xi, 2.0f)) * cos(kmax * i * device_dx + phi);
        w = 0.0f;
        bX = device_b0 * tanh((y - yCenter) / device_sheatThickness);
        bXHalf = device_b0 * tanh((y - yCenter) / device_sheatThickness);
        bY = 0.0f;
        bYHalf = 0.0f;
        bZ = 0.0f;
        p = device_p0 * (device_betaUpstream + pow(cosh((y - yCenter) / device_sheatThickness), -2));
        e = p / (device_gamma_mhd - 1.0f)
          + 0.5f * rho * (u * u + v * v + w * w)
          + 0.5f * (bX * bX + bY * bY + bZ * bZ);
        
        U[j + i * device_ny].rho  = rho;
        U[j + i * device_ny].rhoU = rho * u;
        U[j + i * device_ny].rhoV = rho * v;
        U[j + i * device_ny].rhoW = rho * w;
        U[j + i * device_ny].bX   = bXHalf;
        U[j + i * device_ny].bY   = bYHalf;
        U[j + i * device_ny].bZ   = bZ;
        U[j + i * device_ny].e    = e;
    }
}

void ResistiveMHD2D::initializeU()
{
    cudaMemcpyToSymbol(device_sheatThickness, &sheatThickness, sizeof(float));
    cudaMemcpyToSymbol(device_betaUpstream, &betaUpstream, sizeof(float));
    cudaMemcpyToSymbol(device_rho0, &rho0, sizeof(float));
    cudaMemcpyToSymbol(device_b0, &b0, sizeof(float));
    cudaMemcpyToSymbol(device_p0, &p0, sizeof(float));
    cudaMemcpyToSymbol(device_VA, &VA, sizeof(float));
    cudaMemcpyToSymbol(device_alfvenTime, &alfvenTime, sizeof(float));
    cudaMemcpyToSymbol(device_eta0, &eta0, sizeof(float));
    cudaMemcpyToSymbol(device_eta1, &eta1, sizeof(float));
    cudaMemcpyToSymbol(device_eta, &eta, sizeof(float));
    cudaMemcpyToSymbol(device_triggerRatio, &triggerRatio, sizeof(float));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializeU_kernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(U.data()));

    cudaDeviceSynchronize();

    boundary.symmetricBoundaryX2nd(U);
    boundary.symmetricBoundaryY2nd(U);
}


__device__
inline float getEta(float xPosition, float yPosition)
{
    float eta;

    eta = device_eta;
    
    return eta;
}


__global__ void addResistiveTermToFluxF_kernel(
    const ConservationParameter* U, Flux* flux)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < device_nx - 2) && (0 < j) && (j < device_ny - 2)) {
        float xPosition = i * device_dx, yPosition = j * device_dy;
        float xPositionPlus1 = (i + 1) * device_dx;

        float jY, jZ;
        float eta;
        float etaJY, etaJYPlus1, etaJZ, etaJZPlus1;
        float etaJYBZ, etaJYBZPlus1, etaJZBY, etaJZBYPlus1;

        jY = -(U[j + (i + 1) * device_ny].bZ - U[j + (i - 1) * device_ny].bZ) / (2.0f * device_dx);
        jZ = (U[j + (i + 1) * device_ny].bY - U[j + (i - 1) * device_ny].bY) / (2.0f * device_dx)
           - (U[j + 1 + i * device_ny].bX - U[j - 1 + i * device_ny].bX) / (2.0f * device_dy);
    
        eta = getEta(xPosition, yPosition);
        etaJY = eta * jY; 
        etaJZ = eta * jZ;
        etaJYBZ = etaJY * U[j + i * device_ny].bZ;
        etaJZBY = etaJZ * U[j + i * device_ny].bY;

        jY = -(U[j + (i + 2) * device_ny].bZ - U[j + i * device_ny].bZ) / (2.0f * device_dx);
        jZ = (U[j + (i + 2) * device_ny].bY - U[j + i * device_ny].bY) / (2.0f * device_dx)
           - (U[j + 1 + (i + 1) * device_ny].bX - U[j - 1 + (i + 1) * device_ny].bX) / (2.0f * device_dy);
        
        eta = getEta(xPositionPlus1, yPosition);
        etaJYPlus1 = eta * jY; 
        etaJZPlus1 = eta * jZ;
        etaJYBZPlus1 = etaJYPlus1 * U[j + (i + 1) * device_ny].bZ;
        etaJZBYPlus1 = etaJZPlus1 * U[j + (i + 1) * device_ny].bY;
  
        flux[j + i * device_ny].f5 -= 0.5f * (etaJZ + etaJZPlus1);
        flux[j + i * device_ny].f6 += 0.5f * (etaJY + etaJYPlus1);
        flux[j + i * device_ny].f7 += 0.5f * (etaJYBZ + etaJYBZPlus1)
                                    - 0.5f * (etaJZBY + etaJZBYPlus1);
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

    if ((0 < i) && (i < device_nx - 2) && (0 < j) && (j < device_ny - 2)) {
        float xPosition = i * device_dx, yPosition = j * device_dy;
        float yPositionPlus1 = (j + 1) * device_dy;

        float jX, jZ;
        float eta;
        float etaJX, etaJXPlus1, etaJZ, etaJZPlus1;
        float etaJZBX, etaJZBXPlus1, etaJXBZ, etaJXBZPlus1;

        jX = (U[j + 1 + i * device_ny].bZ - U[j - 1 + i * device_ny].bZ) / (2.0f * device_dy);
        jZ = (U[j + (i + 1) * device_ny].bY - U[j + (i - 1) * device_ny].bY) / (2.0f * device_dx)
           - (U[j + 1 + i * device_ny].bX - U[j - 1 + i * device_ny].bX) / (2.0f * device_dy);
        
        eta = getEta(xPosition, yPosition);
        etaJX = eta * jX;
        etaJZ = eta * jZ;
        etaJXBZ = etaJX * U[j + i * device_ny].bZ;
        etaJZBX = etaJZ * U[j + i * device_ny].bX;

        jX = (U[j + 2 + i * device_ny].bZ - U[j + i * device_ny].bZ) / (2.0f * device_dy);
        jZ = (U[j + 1 + (i + 1) * device_ny].bY - U[j + 1 + (i - 1) * device_ny].bY) / (2.0f * device_dx)
           - (U[j + 2 + i * device_ny].bX - U[j + i * device_ny].bX) / (2.0f * device_dy);
        
        eta = getEta(xPosition, yPositionPlus1);
        etaJXPlus1 = eta * jX;
        etaJZPlus1 = eta * jZ;
        etaJXBZPlus1 = etaJXPlus1 * U[j + 1 + i * device_ny].bZ;
        etaJZBXPlus1 = etaJZPlus1 * U[j + 1 + i * device_ny].bX;
  
        flux[j + i * device_ny].f4 += 0.5f * (etaJZ + etaJZPlus1);
        flux[j + i * device_ny].f6 -= 0.5f * (etaJX + etaJXPlus1);
        flux[j + i * device_ny].f7 += 0.5f * (etaJZBX + etaJZBXPlus1)
                                    - 0.5f * (etaJXBZ + etaJXBZPlus1);
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



//////////////////////////////////////////////////


int main()
{
    initializeDeviceConstants();

    ResistiveMHD2D resistiveMHD2D;

    resistiveMHD2D.initializeU();

    for (step = 0; step < totalStep+1; step++) {
        cudaMemcpyToSymbol(device_totalTime, &totalTime, sizeof(float));

        if (step % recordStep == 0) {
            resistiveMHD2D.save(directoryname, filenameWithoutStep, step);
            logfile << std::to_string(step) << ","
                    << std::setprecision(4) << totalTime
                    << std::endl;
            std::cout << std::to_string(step) << " step done : total time is "
                      << std::setprecision(4) << totalTime
                      << std::endl;
        }
        
        resistiveMHD2D.oneStepRK2PeriodicXSymmetricY();

        if (resistiveMHD2D.checkCalculationIsCrashed()) {
            std::cout << "Calculation stopped! : " << step << " steps" << std::endl;
            return 0;
        }

        totalTime += dt;

        if (step == 0) {
            size_t free_mem = 0;
            size_t total_mem = 0;
            cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);

            std::cout << "Free memory: " << free_mem / (1024 * 1024 * 1024) << " GB" << std::endl;
            std::cout << "Total memory: " << total_mem / (1024 * 1024 * 1024) << " GB" << std::endl;
        }
    }
    
    return 0;
}


