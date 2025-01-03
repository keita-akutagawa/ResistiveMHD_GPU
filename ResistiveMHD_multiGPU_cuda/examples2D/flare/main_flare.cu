#include "main_flare_const.hpp"


__global__ void initializeU_kernel(
    ConservationParameter* U, 
    MPIInfo* device_mPIInfo
) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx && j < device_ny) {
        MPIInfo mPIInfo = *device_mPIInfo;

        if (mPIInfo.isInside(i, j)) {
            int index = mPIInfo.globalToLocal(i, j);

            double rho, u, v, w, bX, bY, bZ, e, p;
            double bXHalf, bYHalf;
            double x = i * device_dx, y = j * device_dy;
            double xHalf = (i + 0.5) * device_dx, yHalf = (j + 0.5) * device_dy;
            double yCenter = 0.5 * (device_ymax - device_ymin);
            double coef = 1.0;
            
            rho    = device_rho0 * (sqrt(device_betaUpstream) + pow(cosh((y - yCenter) / device_sheatThickness), -2));
            u      = 0.0;
            v      = 0.0;
            w      = 0.0;
            bX     = device_b0 * tanh((y - yCenter) / device_sheatThickness)
                   - device_b0 * device_triggerRatio * (y - yCenter) / pow(coef, 2) / device_sheatThickness
                   * exp(-(pow((x - device_xPointPosition) / coef, 2) + pow((y - yCenter) / coef, 2))
                   / pow(2.0 * device_sheatThickness, 2));
            bXHalf = device_b0 * tanh((y - yCenter) / device_sheatThickness)
                   - device_b0 * device_triggerRatio * (y - yCenter) / pow(coef, 2) / device_sheatThickness
                   * exp(-(pow((xHalf - device_xPointPosition) / coef, 2) + pow((y - yCenter) / coef, 2))
                   / pow(2.0 * device_sheatThickness, 2));
            bY     = device_b0 * device_triggerRatio * (x - device_xPointPosition) / pow(coef, 2) / device_sheatThickness
                   * exp(-(pow((x - device_xPointPosition) / coef, 2) + pow((y - yCenter) / coef, 2))
                   / pow(2.0 * device_sheatThickness, 2));
            bYHalf = device_b0 * device_triggerRatio * (x - device_xPointPosition) / pow(coef, 2) / device_sheatThickness
                   * exp(-(pow((x - device_xPointPosition) / coef, 2) + pow((yHalf - yCenter) / coef, 2))
                   / pow(2.0 * device_sheatThickness, 2));
            bZ     = 0.0;
            p      = device_p0 * (device_betaUpstream + pow(cosh((y - yCenter) / device_sheatThickness), -2));
            e      = p / (device_gamma_mhd - 1.0)
                   + 0.5 * rho * (u * u + v * v + w * w)
                   + 0.5 * (bX * bX + bY * bY + bZ * bZ);
            
            U[index].rho  = rho;
            U[index].rhoU = rho * u;
            U[index].rhoV = rho * v;
            U[index].rhoW = rho * w;
            U[index].bX   = bXHalf;
            U[index].bY   = bYHalf;
            U[index].bZ   = bZ;
            U[index].e    = e;
        }
    }
}

void ResistiveMHD2D::initializeU()
{
    cudaMemcpyToSymbol(device_sheatThickness, &sheatThickness, sizeof(double));
    cudaMemcpyToSymbol(device_betaUpstream, &betaUpstream, sizeof(double));
    cudaMemcpyToSymbol(device_rho0, &rho0, sizeof(double));
    cudaMemcpyToSymbol(device_b0, &b0, sizeof(double));
    cudaMemcpyToSymbol(device_p0, &p0, sizeof(double));
    cudaMemcpyToSymbol(device_VA, &VA, sizeof(double));
    cudaMemcpyToSymbol(device_alfvenTime, &alfvenTime, sizeof(double));
    cudaMemcpyToSymbol(device_eta, &eta, sizeof(double));
    cudaMemcpyToSymbol(device_triggerRatio, &triggerRatio, sizeof(double));
    cudaMemcpyToSymbol(device_xPointPosition, &xPointPosition, sizeof(double));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializeU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        device_mPIInfo
    );
    cudaDeviceSynchronize();

    sendrecv_U(U, mPIInfo);
    boundary.flareBoundaryX2nd_U(U);
    boundary.wallBoundaryY2nd_U(U);
    MPI_Barrier(MPI_COMM_WORLD);
}


__device__
double getEta(double& xPosition, double& yPosition)
{
    double etaLocal;

    etaLocal = device_eta;
    
    return etaLocal;
}



//////////////////////////////////////////////////


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    MPIInfo mPIInfo;
    setupInfo(mPIInfo, buffer);

    if (mPIInfo.rank == 0) {
        std::cout << mPIInfo.gridX << "," << mPIInfo.gridY << std::endl;
        logfile   << mPIInfo.gridX << "," << mPIInfo.gridY << std::endl;
    }

    cudaSetDevice(mPIInfo.rank);

    initializeDeviceConstants();


    ResistiveMHD2D resistiveMHD2D(mPIInfo);
    MPI_Barrier(MPI_COMM_WORLD);

    resistiveMHD2D.initializeU();

    if (mPIInfo.rank == 0) {
        size_t free_mem = 0;
        size_t total_mem = 0;
        cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);

        std::cout << "Free memory: " << free_mem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Total memory: " << total_mem / (1024 * 1024) << " MB" << std::endl;
    }

    for (step = 0; step < totalStep+1; step++) {
        MPI_Barrier(MPI_COMM_WORLD);

        cudaMemcpyToSymbol(device_totalTime, &totalTime, sizeof(double));

        if (step % recordStep == 0) {
            if (mPIInfo.rank == 0) {
                logfile << std::to_string(step) << ","
                        << std::setprecision(4) << totalTime
                        << std::endl;
                std::cout << std::to_string(step) << " step done : total time is "
                        << std::setprecision(4) << totalTime
                        << std::endl;
            }
            resistiveMHD2D.save(directoryname, filenameWithoutStep, step);
        }
        
        resistiveMHD2D.oneStepRK2_flareXWallY();

        if (resistiveMHD2D.checkCalculationIsCrashed()) {
            std::cout << "Calculation stopped! : " << step << " steps" << std::endl;
            return 0;
        }

        if (mPIInfo.rank == 0) {
            totalTime += dt;
        }
    }
    
    MPI_Finalize();

    if (mPIInfo.rank == 0) {
        std::cout << "program was completed!" << std::endl;
    }

    return 0;
}


