#include "eigen_tensor.h"

namespace aicpu {

    const Tensor *EigenTensor::GetTensor()
    {
        return tensor_;
    }

}