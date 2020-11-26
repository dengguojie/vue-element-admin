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

#include "less_kernels.h"

#include <securec.h>
#include <stdint.h>
#include <vector>
#include "Eigen/Core"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace {
const char *LESS = "Less";
}

namespace aicpu {
uint32_t LessCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("LessCpuKernel::Compute start!! ");

  uint32_t res = GetInputAndCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("LessCpuKernel::Compute fail!! ");
    return res;
  }

  switch (x_dtype_) {
    case DT_FLOAT16:
      res = DoCompute<Eigen::half>();
      break;
    case DT_FLOAT:
      res = DoCompute<float>();
      break;
    case DT_DOUBLE:
      res = DoCompute<double>();
      break;
    case DT_UINT8:
      res = DoCompute<uint8_t>();
      break;
    case DT_INT8:
      res = DoCompute<int8_t>();
      break;
    case DT_UINT16:
      res = DoCompute<uint16_t>();
      break;
    case DT_INT16:
      res = DoCompute<int16_t>();
      break;
    case DT_INT32:
      res = DoCompute<int>();
      break;
    case DT_INT64:
      res = DoCompute<int64_t>();
      break;
    case DT_UINT32:
      res = DoCompute<uint32_t>();
      break;
    case DT_UINT64:
      res = DoCompute<uint64_t>();
      break;
    default: {
      KERNEL_LOG_ERROR("Less op don't support input tensor types: %d",
                       x_dtype_);
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  if (res != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("LessCpuKernel::Compute fail!! ");
    return res;
  }
  KERNEL_LOG_INFO("LessCpuKernel::Compute end!! ");
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t LessCpuKernel::DoCompute() {
  KERNEL_LOG_INFO("LessCpuKernel::DoCompute start!! ");

  std::shared_ptr<TensorShape> y_shape = y_->GetTensorShape();
  const int64_t y_rank = y_shape->GetDims();
  KERNEL_LOG_INFO("input dim size is %lld(y_rank) ", y_rank);

  switch (y_rank) {
    case 0:
      return KERNEL_STATUS_PARAM_INVALID;
      break;
    case 1:
      return DoRealCompute<T, 1>();
      break;
    case 2:
      return DoRealCompute<T, 2>();
      break;
    case 3:
      return DoRealCompute<T, 3>();
      break;
    case 4:
      return DoRealCompute<T, 4>();
      break;
    case 5:
      return DoRealCompute<T, 5>();
      break;
    case 6:
      return DoRealCompute<T, 6>();
      break;
    case 7:
      return DoRealCompute<T, 7>();
      break;
    default:
      KERNEL_LOG_ERROR("Don't support %d dims", y_rank);
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T, const int32_t rank>
uint32_t LessCpuKernel::DoRealCompute() {
  KERNEL_LOG_INFO("LessCpuKernel::DoRealCompute begin!! ");
  auto input_x1 = reinterpret_cast<T *>(x1_->GetData());
  auto input_x2 = reinterpret_cast<T *>(x2_->GetData());
  auto input_y = reinterpret_cast<bool *>(y_->GetData());

  std::shared_ptr<TensorShape> x1_shape = x1_->GetTensorShape();
  std::shared_ptr<TensorShape> x2_shape = x2_->GetTensorShape();
  std::shared_ptr<TensorShape> y_shape = y_->GetTensorShape();
  std::vector<int64_t> x1_dimsize = GetDimSize(x1_shape);
  std::vector<int64_t> x2_dimsize = GetDimSize(x2_shape);
  std::vector<int64_t> y_dimsize = GetDimSize(y_shape);

  while (int64_t(x1_dimsize.size()) < rank) {
    x1_dimsize.insert(x1_dimsize.begin(), 1);
  }
  while (int64_t(x2_dimsize.size()) < rank) {
    x2_dimsize.insert(x2_dimsize.begin(), 1);
  }

  for (int i = 0; i < rank; i++) {
    if (x1_dimsize[i] != x2_dimsize[i] && x1_dimsize[i] != 1 &&
        x2_dimsize[i] != 1) {
      KERNEL_LOG_ERROR("The x1 shape and the x2 shape can't broadcast.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  size_t x1_size = GetSize(x1_dimsize);
  size_t x2_size = GetSize(x2_dimsize);
  size_t y_size = GetSize(y_dimsize);

  Eigen::TensorMap<Eigen::Tensor<T, 1>> x1_map(input_x1, x1_size);
  Eigen::TensorMap<Eigen::Tensor<T, 1>> x2_map(input_x2, x2_size);
  Eigen::TensorMap<Eigen::Tensor<bool, 1>> y_map(input_y, y_size);
  Eigen::DSizes<Eigen::DenseIndex, rank> x1_pad;
  Eigen::DSizes<Eigen::DenseIndex, rank> x2_pad;
  Eigen::array<int, rank> x1_bcast;
  Eigen::array<int, rank> x2_bcast;
  for (int j = 0; j < rank; j++) {
    x1_pad[j] = x1_dimsize[j];
    x2_pad[j] = x2_dimsize[j];
    x1_bcast[j] = x1_dimsize[j] == y_dimsize[j] ? 1 : y_dimsize[j];
    x2_bcast[j] = x2_dimsize[j] == y_dimsize[j] ? 1 : y_dimsize[j];
  }
  Eigen::Tensor<T, rank> x1_br = x1_map.reshape(x1_pad).broadcast(x1_bcast);
  Eigen::Tensor<T, rank> x2_br = x2_map.reshape(x2_pad).broadcast(x2_bcast);
  Eigen::TensorMap<Eigen::Tensor<T, 1>> x1_br_map(x1_br.data(), y_size);
  Eigen::TensorMap<Eigen::Tensor<T, 1>> x2_br_map(x2_br.data(), y_size);
  const auto &x1 = Eigen::Tensor<T, 1>(x1_br_map);
  const auto &x2 = Eigen::Tensor<T, 1>(x2_br_map);
  Eigen::TensorMap<Eigen::Tensor<bool, 1>> y((bool *)input_y, y_size);
  for (size_t r = 0; r < y_size; ++r) {
    if (x1(r) < x2(r)) {
      y(r) = true;
    } else {
      y(r) = false;
    }
    KERNEL_LOG_INFO("LessCpuKernel::DoCompute y[%d] = %d . ", r, y(r));
  }

  KERNEL_LOG_INFO("LessCpuKernel::DoRealCompute end!! ");
  return KERNEL_STATUS_OK;
}

std::vector<int64_t> LessCpuKernel::GetDimSize(
    std::shared_ptr<TensorShape> input_shape) {
  std::vector<int64_t> dimsize;
  for (int j = 0; j < input_shape->GetDims(); j++) {
    dimsize.push_back(input_shape->GetDimSize(j));
  }
  return dimsize;
}

size_t LessCpuKernel::GetSize(std::vector<int64_t> dim_size) {
  size_t size = 1;
  for (size_t i = 0; i < dim_size.size(); ++i) {
    size *= dim_size[i];
  }
  return size;
}

uint32_t LessCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("LessCpuKernel::GetInputAndCheck start!! ");
  uint32_t res = KERNEL_STATUS_OK;
  // get x1
  x1_ = ctx.Input(0);
  if (x1_ == nullptr) {
    KERNEL_LOG_ERROR("get input:0 failed");
    res = KERNEL_STATUS_PARAM_INVALID;
  }

  x_dtype_ = static_cast<DataType>(x1_->GetDataType());

  // get x2
  x2_ = ctx.Input(1);
  if (x2_ == nullptr) {
    KERNEL_LOG_ERROR("get input:1 failed");
    res = KERNEL_STATUS_PARAM_INVALID;
  }

  DataType x2_DType = static_cast<DataType>(x2_->GetDataType());
  if (x2_DType != x_dtype_) {
    KERNEL_LOG_ERROR(
        "Input parameter error: The data type of x2 need be same with x1.");
    res = KERNEL_STATUS_PARAM_INVALID;
  }

  // get y
  y_ = ctx.Output(0);
  if (y_ == nullptr) {
    KERNEL_LOG_ERROR("get output:0 failed");
    res = KERNEL_STATUS_PARAM_INVALID;
  }

  KERNEL_LOG_INFO("LessCpuKernel::GetInputAndCheck end!! ");
  return res;
}

REGISTER_CPU_KERNEL(LESS, LessCpuKernel);
}  // namespace aicpu
