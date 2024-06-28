#ifndef CONST_STRUCT_H
#define CONST_STRUCT_H

extern const float EPS;
extern const float PI;

extern const float dx;
extern const float xmin;
extern const float xmax;
extern const int nx;

extern const float dy;
extern const float ymin;
extern const float ymax;
extern const int ny;

extern const float CFL;
extern const float gamma_mhd;

extern float dt;

extern const int totalStep;
extern float totalTime;

extern float eta;


extern __constant__ float device_EPS;
extern __constant__ float device_PI;

extern __constant__ float device_dx;
extern __constant__ float device_xmin;
extern __constant__ float device_xmax;
extern __constant__ int device_nx;

extern __constant__ float device_dy;
extern __constant__ float device_ymin;
extern __constant__ float device_ymax;
extern __constant__ int device_ny;

extern __constant__ float device_CFL;
extern __constant__ float device_gamma_mhd;

extern __device__ float device_dt;

extern __device__ float device_eta;


void initializeDeviceConstants();

#endif


