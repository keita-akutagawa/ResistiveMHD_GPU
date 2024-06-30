#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "../../lib_ResistiveMHD_2D_GPU_symmetricXY_single/const.hpp"
#include "../../lib_ResistiveMHD_2D_GPU_symmetricXY_single/resistiveMHD_2D.hpp"
#include <cuda_runtime.h>


std::string directoryname = "results_plasmoid_eta=10000_grid=16";
std::string filenameWithoutStep = "plasmoid";
std::ofstream logfile("results_plasmoid_eta=10000_grid=16/log_plasmoid.txt");

const float EPS = 1.0e-20f;
const float PI = 3.141592653f;

const float gamma_mhd = 5.0f / 3.0f;

const float sheat_thickness = 1.0f;
const float betaUpstream = 2.0f;
const float rho0 = 1.0f;
const float b0 = 1.0f;
const float p0 = b0 * b0 / 2.0f;
const float VA = b0 / sqrt(rho0);
const float alfvenTime = sheat_thickness / VA;

const float eta0 = 1.0 / 100.0f;
const float eta1 = 1.0 / 10000.0f;
float eta = eta0 + eta1;
const float triggerRatio = 0.0f;

const float xmin = 0.0f;
const float xmax = 300.0f * sheat_thickness;
const float dx = sheat_thickness / 16.0f;
const int nx = int((xmax - xmin) / dx);
const float ymin = 0.0f;
const float ymax = 30.0f * sheat_thickness;
const float dy = sheat_thickness / 16.0f;
const int ny = int((ymax - ymin) / dy);

const float CFL = 0.7f;
float dt = 0.0f;
const int totalStep = 100000;
const int recordStep = 400;
float totalTime = 0.0f;

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

__constant__ float device_sheat_thickness;
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


__global__ void poyntingFluxY2nd_kernel(ConservationParameter* U) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx && j < device_ny) {
        float flowRatio = 0.1f;

        if (j < 10) {
            float rho, u, v, w, bX ,bY, bZ, e, p;
            rho = device_betaUpstream * device_rho0;
            u = 0.0; 
            v = flowRatio * device_VA * (1.0 - exp(-device_totalTime / 1000.0f)); 
            w = 0.0;
            bX = -device_b0; 
            bY = 0.0; 
            bZ = 0.0;
            p = device_p0 * device_betaUpstream;
            e = p / (device_gamma_mhd - 1.0) + 0.5 * rho * (u * u + v * v + w * w)
              + 0.5 * (bX * bX + bY * bY + bZ * bZ);

            U[j + i * device_ny].rho  = rho;
            U[j + i * device_ny].rhoU = rho * u;
            U[j + i * device_ny].rhoV = rho * v;
            U[j + i * device_ny].rhoW = rho * w;
            U[j + i * device_ny].bX   = bX;
            U[j + i * device_ny].bY   = bY;
            U[j + i * device_ny].bZ   = bZ;
            U[j + i * device_ny].e    = e;
        }

        if (j > device_ny - 11) {
            float rho, u, v, w, bX ,bY, bZ, e, p;
            rho = device_betaUpstream * device_rho0;
            u = 0.0; 
            v = -flowRatio * device_VA * (1.0 - exp(-device_totalTime / 1000.0f)); 
            w = 0.0;
            bX = device_b0;
            bY = 0.0; 
            bZ = 0.0;
            p = device_p0 * device_betaUpstream;
            e = p / (device_gamma_mhd - 1.0) + 0.5 * rho * (u * u + v * v + w * w)
              + 0.5 * (bX * bX + bY * bY + bZ * bZ);

            U[j + i * device_ny].rho  = rho;
            U[j + i * device_ny].rhoU = rho * u;
            U[j + i * device_ny].rhoV = rho * v;
            U[j + i * device_ny].rhoW = rho * w;
            U[j + i * device_ny].bX   = bX;
            U[j + i * device_ny].bY   = bY;
            U[j + i * device_ny].bZ   = bZ;
            U[j + i * device_ny].e    = e;
        }

    }
}

void Boundary::poyntingFluxY2nd(thrust::device_vector<ConservationParameter>& U)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    poyntingFluxY2nd_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data())
    );

    cudaDeviceSynchronize();
}


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
        
        rho = device_rho0 * (device_betaUpstream + pow(cosh((y - yCenter) / device_sheat_thickness), -2));
        u = 0.0f;
        v = 0.0f;
        w = 0.0f;
        bX = device_b0 * tanh((y - yCenter) / device_sheat_thickness);
        bXHalf = device_b0 * tanh((y - yCenter) / device_sheat_thickness);
        bY = 0.0;
        bYHalf = 0.0;
        bZ = 0.0f;
        p = device_p0 * (device_betaUpstream + pow(cosh((y - yCenter) / device_sheat_thickness), -2));
        e = p / (device_gamma_mhd - 1.0f)
          + 0.5f * rho * (u * u + v * v + w * w)
          + 0.5f * (bX * bX + bY * bY + bZ * bZ);
        
        bX += - device_b0 * device_triggerRatio * (y - yCenter) / device_sheat_thickness
            * exp(-(pow(x - xCenter, 2) + pow(y - yCenter, 2))
            / pow(2.0f * device_sheat_thickness, 2));
        bXHalf += - device_b0 * device_triggerRatio * (y - yCenter) / device_sheat_thickness
                * exp(-(pow(xHalf - xCenter, 2) + pow(y - yCenter, 2))
                / pow(2.0f * device_sheat_thickness, 2));
        bY += device_b0 * device_triggerRatio * (x - xCenter) / device_sheat_thickness
            * exp(-(pow(x - xCenter, 2) + pow(y - yCenter, 2))
            / pow(2.0f * device_sheat_thickness, 2));
        bYHalf += device_b0 * device_triggerRatio * (x - xCenter) / device_sheat_thickness
                * exp(-(pow(x - xCenter, 2) + pow(yHalf - yCenter, 2))
                / pow(2.0f * device_sheat_thickness, 2));
        
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
    cudaMemcpyToSymbol(device_sheat_thickness, &sheat_thickness, sizeof(float));
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
inline float getEta(float& xPosition, float& yPosition)
{
    float eta;

    eta = device_eta0 * pow(cosh(sqrt(
          pow(xPosition - 0.5f * (device_xmax - device_xmin), 2)
        + pow(yPosition - 0.5f * (device_ymax - device_ymin), 2)
        )), -2)
        * exp(-(static_cast<float>(device_totalTime) / (10.0f * device_alfvenTime)))
        + device_eta1;
    
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
        
        resistiveMHD2D.oneStepRK2();

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


