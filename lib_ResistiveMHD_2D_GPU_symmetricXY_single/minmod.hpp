#include <thrust/device_vector.h>
#include <cmath>
#include "const.hpp"


struct MinMod
{
    __device__
    float operator()(const float& x, const float& y) const
    {
        int sign_x = (x > 0) - (x < 0);
        float abs_x = std::abs(x);

        return sign_x * thrust::max(thrust::min(abs_x, sign_x * y), device_EPS);
    }
};


