/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cast_kernels.h"

#include <memory.h>
#include <cfloat>
#include <ctime>

#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *CAST = "Cast";
}

namespace aicpu {
template <typename T, typename S>
uint32_t CastTask(Tensor *&xTensor, Tensor *&yTensor, int64_t &start,
                  int64_t &end) {
  T *inptr = static_cast<T *>(xTensor->GetData());
  S *outptr = static_cast<S *>(yTensor->GetData());
  Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> inputMap(
      (inptr + start), 1, (end - start));
  const auto &input = Eigen::Tensor<T, 2, Eigen::RowMajor>(inputMap);
  Eigen::TensorMap<Eigen::Tensor<S, 2, Eigen::RowMajor>> output(
      (outptr + start), 1, (end - start));
  output = input.template cast<S>();
  return KERNEL_STATUS_OK;
}

void CastCpuKernel::SetMap() {
  calls_[DT_INT8][DT_INT8] = CastTask<int8_t, int8_t>;
  calls_[DT_INT8][DT_INT16] = CastTask<int8_t, int16_t>;
  calls_[DT_INT8][DT_INT32] = CastTask<int8_t, int32_t>;
  calls_[DT_INT8][DT_INT64] = CastTask<int8_t, int64_t>;
  calls_[DT_INT8][DT_FLOAT16] = CastTask<int8_t, Eigen::half>;
  calls_[DT_INT8][DT_FLOAT] = CastTask<int8_t, float>;
  calls_[DT_INT8][DT_DOUBLE] = CastTask<int8_t, double>;
  calls_[DT_INT8][DT_UINT8] = CastTask<int8_t, uint8_t>;
  calls_[DT_INT8][DT_UINT16] = CastTask<int8_t, uint16_t>;
  calls_[DT_INT8][DT_UINT32] = CastTask<int8_t, uint32_t>;
  calls_[DT_INT8][DT_UINT64] = CastTask<int8_t, uint64_t>;
  calls_[DT_INT8][DT_BOOL] = CastTask<int8_t, bool>;
  calls_[DT_INT16][DT_INT8] = CastTask<int16_t, int8_t>;
  calls_[DT_INT16][DT_INT16] = CastTask<int16_t, int16_t>;
  calls_[DT_INT16][DT_INT32] = CastTask<int16_t, int32_t>;
  calls_[DT_INT16][DT_INT64] = CastTask<int16_t, int64_t>;
  calls_[DT_INT16][DT_FLOAT16] = CastTask<int16_t, Eigen::half>;
  calls_[DT_INT16][DT_FLOAT] = CastTask<int16_t, float>;
  calls_[DT_INT16][DT_DOUBLE] = CastTask<int16_t, double>;
  calls_[DT_INT16][DT_UINT8] = CastTask<int16_t, uint8_t>;
  calls_[DT_INT16][DT_UINT16] = CastTask<int16_t, uint16_t>;
  calls_[DT_INT16][DT_UINT32] = CastTask<int16_t, uint32_t>;
  calls_[DT_INT16][DT_UINT64] = CastTask<int16_t, uint64_t>;
  calls_[DT_INT16][DT_BOOL] = CastTask<int16_t, bool>;
  calls_[DT_INT32][DT_INT8] = CastTask<int32_t, int8_t>;
  calls_[DT_INT32][DT_INT16] = CastTask<int32_t, int16_t>;
  calls_[DT_INT32][DT_INT32] = CastTask<int32_t, int32_t>;
  calls_[DT_INT32][DT_INT64] = CastTask<int32_t, int64_t>;
  calls_[DT_INT32][DT_FLOAT16] = CastTask<int32_t, Eigen::half>;
  calls_[DT_INT32][DT_FLOAT] = CastTask<int32_t, float>;
  calls_[DT_INT32][DT_DOUBLE] = CastTask<int32_t, double>;
  calls_[DT_INT32][DT_UINT8] = CastTask<int32_t, uint8_t>;
  calls_[DT_INT32][DT_UINT16] = CastTask<int32_t, uint16_t>;
  calls_[DT_INT32][DT_UINT32] = CastTask<int32_t, uint32_t>;
  calls_[DT_INT32][DT_UINT64] = CastTask<int32_t, uint64_t>;
  calls_[DT_INT32][DT_BOOL] = CastTask<int32_t, bool>;
  calls_[DT_INT64][DT_INT8] = CastTask<int64_t, int8_t>;
  calls_[DT_INT64][DT_INT16] = CastTask<int64_t, int16_t>;
  calls_[DT_INT64][DT_INT32] = CastTask<int64_t, int32_t>;
  calls_[DT_INT64][DT_INT64] = CastTask<int64_t, int64_t>;
  calls_[DT_INT64][DT_FLOAT16] = CastTask<int64_t, Eigen::half>;
  calls_[DT_INT64][DT_FLOAT] = CastTask<int64_t, float>;
  calls_[DT_INT64][DT_DOUBLE] = CastTask<int64_t, double>;
  calls_[DT_INT64][DT_UINT8] = CastTask<int64_t, uint8_t>;
  calls_[DT_INT64][DT_UINT16] = CastTask<int64_t, uint16_t>;
  calls_[DT_INT64][DT_UINT32] = CastTask<int64_t, uint32_t>;
  calls_[DT_INT64][DT_UINT64] = CastTask<int64_t, uint64_t>;
  calls_[DT_INT64][DT_BOOL] = CastTask<int64_t, bool>;
  calls_[DT_FLOAT16][DT_INT8] = CastTask<Eigen::half, int8_t>;
  calls_[DT_FLOAT16][DT_INT16] = CastTask<Eigen::half, int16_t>;
  calls_[DT_FLOAT16][DT_INT32] = CastTask<Eigen::half, int32_t>;
  calls_[DT_FLOAT16][DT_INT64] = CastTask<Eigen::half, int64_t>;
  calls_[DT_FLOAT16][DT_FLOAT16] = CastTask<Eigen::half, Eigen::half>;
  calls_[DT_FLOAT16][DT_FLOAT] = CastTask<Eigen::half, float>;
  calls_[DT_FLOAT16][DT_DOUBLE] = CastTask<Eigen::half, double>;
  calls_[DT_FLOAT16][DT_UINT8] = CastTask<Eigen::half, uint8_t>;
  calls_[DT_FLOAT16][DT_UINT16] = CastTask<Eigen::half, uint16_t>;
  calls_[DT_FLOAT16][DT_UINT32] = CastTask<Eigen::half, uint32_t>;
  calls_[DT_FLOAT16][DT_UINT64] = CastTask<Eigen::half, uint64_t>;
  calls_[DT_FLOAT16][DT_BOOL] = CastTask<Eigen::half, bool>;
  calls_[DT_FLOAT][DT_INT8] = CastTask<float, int8_t>;
  calls_[DT_FLOAT][DT_INT16] = CastTask<float, int16_t>;
  calls_[DT_FLOAT][DT_INT32] = CastTask<float, int32_t>;
  calls_[DT_FLOAT][DT_INT64] = CastTask<float, int64_t>;
  calls_[DT_FLOAT][DT_FLOAT16] = CastTask<float, Eigen::half>;
  calls_[DT_FLOAT][DT_FLOAT] = CastTask<float, float>;
  calls_[DT_FLOAT][DT_DOUBLE] = CastTask<float, double>;
  calls_[DT_FLOAT][DT_UINT8] = CastTask<float, uint8_t>;
  calls_[DT_FLOAT][DT_UINT16] = CastTask<float, uint16_t>;
  calls_[DT_FLOAT][DT_UINT32] = CastTask<float, uint32_t>;
  calls_[DT_FLOAT][DT_UINT64] = CastTask<float, uint64_t>;
  calls_[DT_FLOAT][DT_BOOL] = CastTask<float, bool>;
  calls_[DT_DOUBLE][DT_INT8] = CastTask<double, int8_t>;
  calls_[DT_DOUBLE][DT_INT16] = CastTask<double, int16_t>;
  calls_[DT_DOUBLE][DT_INT32] = CastTask<double, int32_t>;
  calls_[DT_DOUBLE][DT_INT64] = CastTask<double, int64_t>;
  calls_[DT_DOUBLE][DT_FLOAT16] = CastTask<double, Eigen::half>;
  calls_[DT_DOUBLE][DT_FLOAT] = CastTask<double, float>;
  calls_[DT_DOUBLE][DT_DOUBLE] = CastTask<double, double>;
  calls_[DT_DOUBLE][DT_UINT8] = CastTask<double, uint8_t>;
  calls_[DT_DOUBLE][DT_UINT16] = CastTask<double, uint16_t>;
  calls_[DT_DOUBLE][DT_UINT32] = CastTask<double, uint32_t>;
  calls_[DT_DOUBLE][DT_UINT64] = CastTask<double, uint64_t>;
  calls_[DT_DOUBLE][DT_BOOL] = CastTask<double, bool>;
  calls_[DT_UINT8][DT_INT8] = CastTask<uint8_t, int8_t>;
  calls_[DT_UINT8][DT_INT16] = CastTask<uint8_t, int16_t>;
  calls_[DT_UINT8][DT_INT32] = CastTask<uint8_t, int32_t>;
  calls_[DT_UINT8][DT_INT64] = CastTask<uint8_t, int64_t>;
  calls_[DT_UINT8][DT_FLOAT16] = CastTask<uint8_t, Eigen::half>;
  calls_[DT_UINT8][DT_FLOAT] = CastTask<uint8_t, float>;
  calls_[DT_UINT8][DT_DOUBLE] = CastTask<uint8_t, double>;
  calls_[DT_UINT8][DT_UINT8] = CastTask<uint8_t, uint8_t>;
  calls_[DT_UINT8][DT_UINT16] = CastTask<uint8_t, uint16_t>;
  calls_[DT_UINT8][DT_UINT32] = CastTask<uint8_t, uint32_t>;
  calls_[DT_UINT8][DT_UINT64] = CastTask<uint8_t, uint64_t>;
  calls_[DT_UINT8][DT_BOOL] = CastTask<uint8_t, bool>;
  calls_[DT_UINT16][DT_INT8] = CastTask<uint16_t, int8_t>;
  calls_[DT_UINT16][DT_INT16] = CastTask<uint16_t, int16_t>;
  calls_[DT_UINT16][DT_INT32] = CastTask<uint16_t, int32_t>;
  calls_[DT_UINT16][DT_INT64] = CastTask<uint16_t, int64_t>;
  calls_[DT_UINT16][DT_FLOAT16] = CastTask<uint16_t, Eigen::half>;
  calls_[DT_UINT16][DT_FLOAT] = CastTask<uint16_t, float>;
  calls_[DT_UINT16][DT_DOUBLE] = CastTask<uint16_t, double>;
  calls_[DT_UINT16][DT_UINT8] = CastTask<uint16_t, uint8_t>;
  calls_[DT_UINT16][DT_UINT16] = CastTask<uint16_t, uint16_t>;
  calls_[DT_UINT16][DT_UINT32] = CastTask<uint16_t, uint32_t>;
  calls_[DT_UINT16][DT_UINT64] = CastTask<uint16_t, uint64_t>;
  calls_[DT_UINT16][DT_BOOL] = CastTask<uint16_t, bool>;
  calls_[DT_UINT32][DT_INT8] = CastTask<uint32_t, int8_t>;
  calls_[DT_UINT32][DT_INT16] = CastTask<uint32_t, int16_t>;
  calls_[DT_UINT32][DT_INT32] = CastTask<uint32_t, int32_t>;
  calls_[DT_UINT32][DT_INT64] = CastTask<uint32_t, int64_t>;
  calls_[DT_UINT32][DT_FLOAT16] = CastTask<uint32_t, Eigen::half>;
  calls_[DT_UINT32][DT_FLOAT] = CastTask<uint32_t, float>;
  calls_[DT_UINT32][DT_DOUBLE] = CastTask<uint32_t, double>;
  calls_[DT_UINT32][DT_UINT8] = CastTask<uint32_t, uint8_t>;
  calls_[DT_UINT32][DT_UINT16] = CastTask<uint32_t, uint16_t>;
  calls_[DT_UINT32][DT_UINT32] = CastTask<uint32_t, uint32_t>;
  calls_[DT_UINT32][DT_UINT64] = CastTask<uint32_t, uint64_t>;
  calls_[DT_UINT32][DT_BOOL] = CastTask<uint32_t, bool>;
  calls_[DT_UINT64][DT_INT8] = CastTask<uint64_t, int8_t>;
  calls_[DT_UINT64][DT_INT16] = CastTask<uint64_t, int16_t>;
  calls_[DT_UINT64][DT_INT32] = CastTask<uint64_t, int32_t>;
  calls_[DT_UINT64][DT_INT64] = CastTask<uint64_t, int64_t>;
  calls_[DT_UINT64][DT_FLOAT16] = CastTask<uint64_t, Eigen::half>;
  calls_[DT_UINT64][DT_FLOAT] = CastTask<uint64_t, float>;
  calls_[DT_UINT64][DT_DOUBLE] = CastTask<uint64_t, double>;
  calls_[DT_UINT64][DT_UINT8] = CastTask<uint64_t, uint8_t>;
  calls_[DT_UINT64][DT_UINT16] = CastTask<uint64_t, uint16_t>;
  calls_[DT_UINT64][DT_UINT32] = CastTask<uint64_t, uint32_t>;
  calls_[DT_UINT64][DT_UINT64] = CastTask<uint64_t, uint64_t>;
  calls_[DT_UINT64][DT_BOOL] = CastTask<uint64_t, bool>;
  calls_[DT_BOOL][DT_INT8] = CastTask<bool, int8_t>;
  calls_[DT_BOOL][DT_INT16] = CastTask<bool, int16_t>;
  calls_[DT_BOOL][DT_INT32] = CastTask<bool, int32_t>;
  calls_[DT_BOOL][DT_INT64] = CastTask<bool, int64_t>;
  calls_[DT_BOOL][DT_FLOAT16] = CastTask<bool, Eigen::half>;
  calls_[DT_BOOL][DT_FLOAT] = CastTask<bool, float>;
  calls_[DT_BOOL][DT_DOUBLE] = CastTask<bool, double>;
  calls_[DT_BOOL][DT_UINT8] = CastTask<bool, uint8_t>;
  calls_[DT_BOOL][DT_UINT16] = CastTask<bool, uint16_t>;
  calls_[DT_BOOL][DT_UINT32] = CastTask<bool, uint32_t>;
  calls_[DT_BOOL][DT_UINT64] = CastTask<bool, uint64_t>;
  calls_[DT_BOOL][DT_BOOL] = CastTask<bool, bool>;
}

uint32_t CastCpuKernel::TransferType(int64_t start, int64_t end) {
  if (calls_.find(xDataType_) == calls_.end()) {
    KERNEL_LOG_ERROR(
        "CastCpuKernel::CastCpuKernel op don't support input tensor types: %s",
        typeid(xDataType_).name());
    return KERNEL_STATUS_PARAM_INVALID;
  } else if (calls_[xDataType_].find(yDataType_) == calls_[xDataType_].end()) {
    KERNEL_LOG_ERROR(
        "CastCpuKernel::CaseKernel op don't support output tensor types: %s",
        typeid(yDataType_).name());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return calls_[xDataType_][yDataType_](xTensor_, yTensor_, start, end);
}

uint32_t CastCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("CastCpuKernel::Compute begin.");
  xTensor_ = ctx.Input(0);
  if (xTensor_ == nullptr) {
    KERNEL_LOG_ERROR("CastCpuKernel::GetInputAndCheck: get input tensor failed");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  yTensor_ = ctx.Output(0);
  if (yTensor_ == nullptr) {
    KERNEL_LOG_ERROR("CastCpuKernel::GetInputAndCheck: get output tensor failed");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  xDataSize_ = xTensor_->GetDataSize();
  yDataSize_ = yTensor_->GetDataSize();
  if (xDataSize_ < 1 || yDataSize_ < 1) {
    KERNEL_LOG_ERROR("CastCpuKernel::inpudata size:%lld or output size:%lld < 1",
                     xDataSize_, yDataSize_);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  KERNEL_LOG_INFO("CastCpuKernel:: input size:%lld, output size:%lld", xDataSize_,
                  yDataSize_);
  xDataType_ = DataType(xTensor_->GetDataType());
  yDataType_ = DataType(yTensor_->GetDataType());
  KERNEL_LOG_INFO("CastCpuKernel::Cast input type:%d, out type:%d", xDataType_,
                  yDataType_);
  int xTypeSize = GetSizeByDataType(static_cast<DataType>(xDataType_));
  int yTypeSize = GetSizeByDataType(static_cast<DataType>(yDataType_));
  KERNEL_LOG_INFO("CastCpuKernel::Cast input type size:%d, out type size:%d",
                  xTypeSize, yTypeSize);
  xDataSize_ = xDataSize_ / xTypeSize;
  yDataSize_ = yDataSize_ / yTypeSize;
  KERNEL_LOG_INFO("CastCpuKernel::inputdata length:%lld, output length:%lld",
                  xDataSize_, yDataSize_);
  if (xDataSize_ > yDataSize_) {
    xDataSize_ = yDataSize_;
  }
  uint32_t minCoreNum = 1;
  int64_t maxCoreNum =
      std::max(minCoreNum, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
  if (maxCoreNum > xDataSize_) {
    maxCoreNum = xDataSize_;
  }
  SetMap();
  aicpu::CpuKernelUtils::ParallelFor(
      ctx, xDataSize_, xDataSize_ / maxCoreNum,
      [&](int64_t start, int64_t end) {
        uint32_t result = TransferType(start, end);
        if (result == KERNEL_STATUS_PARAM_INVALID) {
          KERNEL_LOG_ERROR("CastCpuKernel::TransferType failed");
          return KERNEL_STATUS_PARAM_INVALID;
        }
        return KERNEL_STATUS_OK;
      });
  calls_.clear();
  yDataSize_ = yTensor_->GetDataSize();
  KERNEL_LOG_INFO("CastCpuKernel::Cast output size:%lld.", yDataSize_);
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(CAST, CastCpuKernel);
}  // namespace aicpu
