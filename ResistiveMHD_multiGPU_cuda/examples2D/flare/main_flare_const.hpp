#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "../ResistiveMHD2D_GPU/get_eta.hpp"
#include "../ResistiveMHD2D_GPU/const.hpp"
#include "../ResistiveMHD2D_GPU/resistiveMHD2D.hpp"
#include <cuda_runtime.h>
#include <mpi.h>


std::string directoryname = "/cfca-work/akutagawakt/ResistiveMHD_cuda/results_flare";
std::string filenameWithoutStep = "flare";
std::ofstream logfile("/cfca-work/akutagawakt/ResistiveMHD_cuda/results_flare/log_flare.txt");

const int buffer = 3;

const double EPS = 1e-20;
const double PI = 3.141592653589793;

const double gamma_mhd = 5.0 / 3.0;

const double sheatThickness = 1.0;
const double betaUpstream = 1.0;
const double rho0 = 1.0;
const double b0 = 1.0;
const double p0 = b0 * b0 / 2.0;
const double VA = b0 / sqrt(rho0);
const double alfvenTime = sheatThickness / VA;

const double eta = 1.0 / 1000.0;
const double triggerRatio = 0.1;
const double xPointPosition = 20.0;

const double xmin = 0.0;
const double xmax = 200.0;
const double dx = sheatThickness / 8.0;
const int nx = int((xmax - xmin) / dx);
const double ymin = 0.0;
const double ymax = 50.0;
const double dy = sheatThickness / 8.0;
const int ny = int((ymax - ymin) / dy);

const double CFL = 0.7;
double dt = 0.0;
const int totalStep = 100000;
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

__constant__ double device_sheatThickness;
__constant__ double device_betaUpstream;
__constant__ double device_rho0;
__constant__ double device_b0;
__constant__ double device_p0;
__constant__ double device_VA;
__constant__ double device_alfvenTime;

__device__ double device_eta;

__constant__ double device_triggerRatio;
__constant__ double device_xPointPosition;

int step;
__device__ int device_step;

