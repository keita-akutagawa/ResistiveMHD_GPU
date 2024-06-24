#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "../../lib_ResistiveMHD_2D_GPU_symmetricXY/const.hpp"
#include "../../lib_ResistiveMHD_2D_GPU_symmetricXY/resistiveMHD_2D.hpp"


std::string directoryname = "results";
std::string filenameWithoutStep = "Petscheck";
std::ofstream logfile("log_Petscheck.txt");

const double EPS = 1e-20;
const double PI = 3.141592653589793;

const double gamma_mhd = 5.0 / 3.0;

const double sheat_thickness = 1.0;
const double betaUpstream = 0.2;
const double rho0 = 1.0;
const double b0 = 1.0;
const double p0 = b0 * b0 / 2.0;

const double eta0 = 1.0 / 1000.0;
const double eta1 = 1.0 / 60.0;

const double xmin = 0.0;
const double xmax = 100.0;
const double dx = sheat_thickness / 10.0;
const int nx = int((xmax - xmin) / dx);
const double ymin = 0.0;
const double ymax = 20.0;
const double dy = sheat_thickness / 10.0;
const int ny = int((ymax - ymin) / dy);

const double CFL = 0.7;
double dt = 0.0;
const int totalStep = 10000;
const int recordStep = 100;
double totalTime = 0.0;

__constant__ double device_EPS;
__constant__ double device_PI;

__constant__ double device_dx;
__constant__ double device_xmin;
__constant__ double device_xmax;
__constant__ int device_nx;

__constant__ double device_dy;
__constant__ double device_ymin;
__constant__ double device_ymax;
__constant__ int device_ny;

__constant__ double device_CFL;
__constant__ double device_gamma_mhd;

__device__ double device_dt;

__constant__ double device_sheat_thickness;
__constant__ double device_betaUpstream;
__constant__ double device_rho0;
__constant__ double device_b0;
__constant__ double device_p0;

__constant__ double device_eta0;
__constant__ double device_eta1;


__global__ void initializeU_kernel(ConservationParameter* U) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx && j < device_ny) {
        double rho, u, v, w, bX, bY, bZ, e, p;
        double y = j * device_dy;
        
        rho = device_rho0 * (device_betaUpstream + pow(cosh((y - 0.5 * device_ymax) / device_sheat_thickness), -2));
        u = 0.0;
        v = 0.0;
        w = 0.0;
        bX = device_b0 * tanh((y - 0.5 * device_ymax) / device_sheat_thickness);
        bY = 0.0;
        bZ = 0.0;
        p = device_p0 * (device_betaUpstream + pow(cosh((y - 0.5 * device_ymax) / device_sheat_thickness), -2));
        e = p / (device_gamma_mhd - 1.0)
          + 0.5 * rho * (u * u + v * v + w * w)
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

void ResistiveMHD2D::initializeU()
{
    cudaMemcpyToSymbol(device_sheat_thickness, &sheat_thickness, sizeof(double));
    cudaMemcpyToSymbol(device_betaUpstream, &betaUpstream, sizeof(double));
    cudaMemcpyToSymbol(device_rho0, &rho0, sizeof(double));
    cudaMemcpyToSymbol(device_b0, &b0, sizeof(double));
    cudaMemcpyToSymbol(device_p0, &p0, sizeof(double));
    cudaMemcpyToSymbol(device_eta0, &eta0, sizeof(double));
    cudaMemcpyToSymbol(device_eta1, &eta1, sizeof(double));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializeU_kernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(U.data()));

    cudaDeviceSynchronize();
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
        
        eta = device_eta0 
            + (device_eta1 - device_eta0)
            * pow(cosh(sqrt(
                pow(xPosition - 0.5 * (device_xmax - device_xmin), 2)
              + pow(yPosition - 0.5 * (device_ymax - device_ymin), 2)
            )), -2);
  
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
        
        eta = device_eta0 
            + (device_eta1 - device_eta0)
            * pow(cosh(sqrt(
                pow(xPosition - 0.5 * (device_xmax - device_xmin), 2)
              + pow(yPosition - 0.5 * (device_ymax - device_ymin), 2
                ))), -2);
  
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


//////////////////////////////////////////////////


int main()
{
    initializeDeviceConstants();

    ResistiveMHD2D resistiveMHD2D;

    resistiveMHD2D.initializeU();

    for (int step = 0; step < totalStep+1; step++) {
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
    }
    
    return 0;
}


