/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of drop out gen mask
 */


#ifndef _AICPU_TEST_ADD_KERNELS_H_
#define _AICPU_TEST_ADD_KERNELS_H_

#include "cpu_kernel.h"
#include "Eigen/Core"

namespace aicpu {
struct tensorShapeDesc {
    int64_t batch = 0;
    int64_t channels = 0;
    int64_t height = 0;
    int64_t width = 0;
};

struct extraParameters {
    int32_t strideH = 0;
    int32_t strideW = 0;
    int32_t padUp = 0;
    int32_t padDown = 0;
    int32_t padLeft = 0;
    int32_t padRight = 0;
    int32_t ksizeX = 0;
    int32_t ksizeY = 0;
    int32_t dilationsH = 0;
    int32_t dilationsW = 0;
};

class DeformableOffsetsCpuKernel : public CpuKernel {
public:
    DeformableOffsetsCpuKernel() = default;
    ~DeformableOffsetsCpuKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    uint32_t GetInputParam(CpuKernelContext &ctx);
    uint32_t ParseInputParam();
    uint32_t CheckInputParam();

    template <typename T>
    uint32_t DoCompute(Eigen::half *inputDataX, T *inputDataOffsets, Eigen::half *inputDataY);

    template <typename T>
    uint32_t ComputePosition(const Eigen::half *inputX, const T *inputOffset, Eigen::half *inputY, int currentAxis);

    template <typename T>
    uint32_t ComputeResult(const Eigen::half *inputX, const T *inputOffset, Eigen::half *inputY, int xSrc,
                           int ySrc, int currentAxis);

    uint32_t BilinearInterpolate(Eigen::half &out, const Eigen::half *in, int c_axis, float h, float w);

private:
    Tensor *xTensor;
    Tensor *offsetsTensor;
    Tensor *yTensor;
    // Move step size for convolution calculation
    std::vector<int64_t> stride_list_;
    // Specify the number of layers filled with 0 around the input x feature map
    std::vector<int64_t> pads_list_;
    // Convolution kernel size
    std::vector<int64_t> ksize_list_;
    // Used to change the size of the convolution kernel
    std::vector<int64_t> dilation_list_;
    // Specify the type of input x
    std::string data_format_;
    int deformable_groups_;
    tensorShapeDesc x_;
    tensorShapeDesc offset_;
    tensorShapeDesc y_;
    extraParameters param_;
};
}
#endif
