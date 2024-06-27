#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "../../lib_ResistiveMHD_2D_GPU_symmetricXY/const.hpp"
#include "../../lib_ResistiveMHD_2D_GPU_symmetricXY/resistiveMHD_2D.hpp"


std::string directoryname = "results";
std::string filenameWithoutStep = "plasmoid";
std::ofstream logfile("results/log_plasmoid.txt");

const double EPS = 1e-20;
const double PI = 3.141592653589793;

const double gamma_mhd = 5.0 / 3.0;

const double sheat_thickness = 1.0;
const double betaUpstream = 0.5;
const double rho0 = 1.0;
const double b0 = 1.0;
const double p0 = b0 * b0 / 2.0;
const double VA = b0 / sqrt(rho0);
const double alfvenTime = sheat_thickness / VA;

const double eta0 = 1.0 / 100.0;
const double eta1 = 1.0 / 100.0;
double eta = eta0 + eta1;
const double triggerRatio = 0.0;

const double xmin = 0.0;
const double xmax = 400.0;
const double dx = sheat_thickness / 8.0;
const int nx = int((xmax - xmin) / dx);
const double ymin = 0.0;
const double ymax = 40.0;
const double dy = sheat_thickness / 8.0;
const int ny = int((ymax - ymin) / dy);

const double CFL = 0.7;
double dt = 0.0;
const int totalStep = 1000000;
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
__device__ double device_totalTime;

__constant__ double device_sheat_thickness;
__constant__ double device_betaUpstream;
__constant__ double device_rho0;
__constant__ double device_b0;
__constant__ double device_p0;
__constant__ double device_VA;
__constant__ double device_alfvenTime;

__constant__ double device_eta0;
__constant__ double device_eta1;
__device__ double device_eta;

__constant__ double device_triggerRatio;

int step;
__device__ int device_step;


__global__ void initializeU_kernel(ConservationParameter* U) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx && j < device_ny) {
        double rho, u, v, w, bX, bY, bZ, e, p;
        double bXHalf, bYHalf;
        double x = i * device_dx, y = j * device_dy;
        double xHalf = (i + 0.5) * device_dx, yHalf = (j + 0.5) * device_dy;
        double xCenter = 0.5 * (device_xmax - device_xmin), yCenter = 0.5 * (device_ymax - device_ymin);
        
        rho = device_rho0 * (device_betaUpstream + pow(cosh((y - yCenter) / device_sheat_thickness), -2));
        u = 0.0;
        v = 0.0;
        w = 0.0;
        bX = device_b0 * tanh((y - yCenter) / device_sheat_thickness)
           - device_b0 * device_triggerRatio * (y - yCenter) / device_sheat_thickness
           * exp(-(pow(x - xCenter, 2) + pow(y - yCenter, 2))
           / pow(2.0 * device_sheat_thickness, 2));
        bXHalf = device_b0 * tanh((y - yCenter) / device_sheat_thickness)
               - device_b0 * device_triggerRatio * (y - yCenter) / device_sheat_thickness
               * exp(-(pow(xHalf - xCenter, 2) + pow(y - yCenter, 2))
               / pow(2.0 * device_sheat_thickness, 2));
        bY = device_b0 * device_triggerRatio * (x - xCenter) / device_sheat_thickness
           * exp(-(pow(x - xCenter, 2) + pow(y - yCenter, 2))
           / pow(2.0 * device_sheat_thickness, 2));
        bYHalf = device_b0 * device_triggerRatio * (x - xCenter) / device_sheat_thickness
               * exp(-(pow(x - xCenter, 2) + pow(yHalf - yCenter, 2))
               / pow(2.0 * device_sheat_thickness, 2));
        bZ = 0.0;
        p = device_p0 * (device_betaUpstream + pow(cosh((y - yCenter) / device_sheat_thickness), -2));
        e = p / (device_gamma_mhd - 1.0)
          + 0.5 * rho * (u * u + v * v + w * w)
          + 0.5 * (bX * bX + bY * bY + bZ * bZ);
        
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
    cudaMemcpyToSymbol(device_sheat_thickness, &sheat_thickness, sizeof(double));
    cudaMemcpyToSymbol(device_betaUpstream, &betaUpstream, sizeof(double));
    cudaMemcpyToSymbol(device_rho0, &rho0, sizeof(double));
    cudaMemcpyToSymbol(device_b0, &b0, sizeof(double));
    cudaMemcpyToSymbol(device_p0, &p0, sizeof(double));
    cudaMemcpyToSymbol(device_VA, &VA, sizeof(double));
    cudaMemcpyToSymbol(device_alfvenTime, &alfvenTime, sizeof(double));
    cudaMemcpyToSymbol(device_eta0, &eta0, sizeof(double));
    cudaMemcpyToSymbol(device_eta1, &eta1, sizeof(double));
    cudaMemcpyToSymbol(device_eta, &eta, sizeof(double));
    cudaMemcpyToSymbol(device_triggerRatio, &triggerRatio, sizeof(double));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializeU_kernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(U.data()));

    cudaDeviceSynchronize();

    boundary.symmetricBoundaryX2nd(U);
    boundary.symmetricBoundaryY2nd(U);
}


__device__
inline double getEta(double xPosition, double yPosition)
{
    double eta;

    eta = device_eta0 * pow(cosh(sqrt(
          pow(xPosition - 0.5 * (device_xmax - device_xmin), 2)
        + pow(yPosition - 0.5 * (device_ymax - device_ymin), 2)
        )), -2)
        * exp(-(static_cast<double>(device_totalTime) / (1.0 * device_alfvenTime)))
        + device_eta1;
    
    return eta;
}


__global__ void addResistiveTermToFluxF_kernel(
    const ConservationParameter* U, Flux* flux)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < device_nx - 2) && (0 < j) && (j < device_ny - 2)) {
        double xPosition = i * device_dx, yPosition = j * device_dy;
        double xPositionPlus1 = (i + 1) * device_dx;

        double jY, jZ;
        double eta;
        double etaJY, etaJYPlus1, etaJZ, etaJZPlus1;
        double etaJYBZ, etaJYBZPlus1, etaJZBY, etaJZBYPlus1;

        jY = -(U[j + (i + 1) * device_ny].bZ - U[j + (i - 1) * device_ny].bZ) / (2.0 * device_dx);
        jZ = (U[j + (i + 1) * device_ny].bY - U[j + (i - 1) * device_ny].bY) / (2.0 * device_dx)
           - (U[j + 1 + i * device_ny].bX - U[j - 1 + i * device_ny].bX) / (2.0 * device_dy);
    
        eta = getEta(xPosition, yPosition);
        etaJY = eta * jY; 
        etaJZ = eta * jZ;
        etaJYBZ = etaJY * U[j + i * device_ny].bZ;
        etaJZBY = etaJZ * U[j + i * device_ny].bY;

        jY = -(U[j + (i + 2) * device_ny].bZ - U[j + i * device_ny].bZ) / (2.0 * device_dx);
        jZ = (U[j + (i + 2) * device_ny].bY - U[j + i * device_ny].bY) / (2.0 * device_dx)
           - (U[j + 1 + (i + 1) * device_ny].bX - U[j - 1 + (i + 1) * device_ny].bX) / (2.0 * device_dy);
        
        eta = getEta(xPositionPlus1, yPosition);
        etaJYPlus1 = eta * jY; 
        etaJZPlus1 = eta * jZ;
        etaJYBZPlus1 = etaJYPlus1 * U[j + (i + 1) * device_ny].bZ;
        etaJZBYPlus1 = etaJZPlus1 * U[j + (i + 1) * device_ny].bY;
  
        flux[j + i * device_ny].f5 -= 0.5 * (etaJZ + etaJZPlus1);
        flux[j + i * device_ny].f6 += 0.5 * (etaJY + etaJYPlus1);
        flux[j + i * device_ny].f7 += 0.5 * (etaJYBZ + etaJYBZPlus1)
                                    - 0.5 * (etaJZBY + etaJZBYPlus1);
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
        double xPosition = i * device_dx, yPosition = j * device_dy;
        double yPositionPlus1 = (j + 1) * device_dy;

        double jX, jZ;
        double eta;
        double etaJX, etaJXPlus1, etaJZ, etaJZPlus1;
        double etaJZBX, etaJZBXPlus1, etaJXBZ, etaJXBZPlus1;

        jX = (U[j + 1 + i * device_ny].bZ - U[j - 1 + i * device_ny].bZ) / (2.0 * device_dy);
        jZ = (U[j + (i + 1) * device_ny].bY - U[j + (i - 1) * device_ny].bY) / (2.0 * device_dx)
           - (U[j + 1 + i * device_ny].bX - U[j - 1 + i * device_ny].bX) / (2.0 * device_dy);
        
        eta = getEta(xPosition, yPosition);
        etaJX = eta * jX;
        etaJZ = eta * jZ;
        etaJXBZ = etaJX * U[j + i * device_ny].bZ;
        etaJZBX = etaJZ * U[j + i * device_ny].bX;

        jX = (U[j + 2 + i * device_ny].bZ - U[j + i * device_ny].bZ) / (2.0 * device_dy);
        jZ = (U[j + 1 + (i + 1) * device_ny].bY - U[j + 1 + (i - 1) * device_ny].bY) / (2.0 * device_dx)
           - (U[j + 2 + i * device_ny].bX - U[j + i * device_ny].bX) / (2.0 * device_dy);
        
        eta = getEta(xPosition, yPositionPlus1);
        etaJXPlus1 = eta * jX;
        etaJZPlus1 = eta * jZ;
        etaJXBZPlus1 = etaJXPlus1 * U[j + 1 + i * device_ny].bZ;
        etaJZBXPlus1 = etaJZPlus1 * U[j + 1 + i * device_ny].bX;
  
        flux[j + i * device_ny].f4 += 0.5 * (etaJZ + etaJZPlus1);
        flux[j + i * device_ny].f6 -= 0.5 * (etaJX + etaJXPlus1);
        flux[j + i * device_ny].f7 += 0.5 * (etaJZBX + etaJZBXPlus1)
                                    - 0.5 * (etaJXBZ + etaJXBZPlus1);
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
        cudaMemcpyToSymbol(device_totalTime, &totalTime, sizeof(int));

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


