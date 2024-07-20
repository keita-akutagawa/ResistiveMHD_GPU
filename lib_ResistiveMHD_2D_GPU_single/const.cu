#include "const.hpp"


void initializeDeviceConstants() {
    cudaMemcpyToSymbol(device_EPS, &EPS, sizeof(float));
    cudaMemcpyToSymbol(device_PI, &PI, sizeof(float));

    cudaMemcpyToSymbol(device_dx, &dx, sizeof(float));
    cudaMemcpyToSymbol(device_xmin, &xmin, sizeof(float));
    cudaMemcpyToSymbol(device_xmax, &xmax, sizeof(float));
    cudaMemcpyToSymbol(device_nx, &nx, sizeof(int));

    cudaMemcpyToSymbol(device_dy, &dy, sizeof(float));
    cudaMemcpyToSymbol(device_ymin, &ymin, sizeof(float));
    cudaMemcpyToSymbol(device_ymax, &ymax, sizeof(float));
    cudaMemcpyToSymbol(device_ny, &ny, sizeof(int));

    cudaMemcpyToSymbol(device_CFL, &CFL, sizeof(float));
    cudaMemcpyToSymbol(device_gamma_mhd, &gamma_mhd, sizeof(float));

    cudaMemcpyToSymbol(device_dt, &dt, sizeof(float));

    cudaMemcpyToSymbol(device_eta, &eta, sizeof(float));
}
