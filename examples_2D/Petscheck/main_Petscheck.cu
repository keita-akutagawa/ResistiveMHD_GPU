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

const double shear_thickness = 1.0;
const double beta = 2.0;
const double rho0 = 1.0;
const double b0 = 1.0;
const double p0 = beta * b0 * b0 / 2.0;
const double v0 = sqrt(b0 * b0 / rho0 + gamma_mhd * p0 / rho0);

const double xmin = 0.0;
const double xmax = 2.0 * PI * shear_thickness / 0.4;
const double dx = shear_thickness / 32.0;
const int nx = int((xmax - xmin) / dx);
const double ymin = 0.0;
const double ymax = 2.0 * 10.0 * shear_thickness;
const double dy = shear_thickness / 32.0;
const int ny = int((ymax - ymin) / dy);

const double xCenter = (xmax - xmin) / 2.0;
const double yCenter = (ymax - ymin) / 2.0;

const double CFL = 0.7;
double dt = 0.0;
const int totalStep = 30000;
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

__constant__ double device_xCenter;
__constant__ double device_yCenter;

__constant__ double device_CFL;
__constant__ double device_gamma_mhd;

__device__ double device_dt;

__constant__ double device_shear_thickness;
__constant__ double device_beta;
__constant__ double device_rho0;
__constant__ double device_b0;
__constant__ double device_p0;
__constant__ double device_v0;


__global__ void initializeU_kernel(ConservationParameter* U) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx && j < device_ny) {
        double rho, u, v, w, bX, bY, bZ, e, p;
        
        
        
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
    cudaMemcpyToSymbol(device_xCenter, &xCenter, sizeof(double));
    cudaMemcpyToSymbol(device_yCenter, &yCenter, sizeof(double));
    cudaMemcpyToSymbol(device_shear_thickness, &shear_thickness, sizeof(double));
    cudaMemcpyToSymbol(device_beta, &beta, sizeof(double));
    cudaMemcpyToSymbol(device_rho0, &rho0, sizeof(double));
    cudaMemcpyToSymbol(device_b0, &b0, sizeof(double));
    cudaMemcpyToSymbol(device_p0, &p0, sizeof(double));
    cudaMemcpyToSymbol(device_v0, &v0, sizeof(double));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializeU_kernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(U.data()));

    cudaDeviceSynchronize();
}


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


