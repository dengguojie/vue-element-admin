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

#include "realdiv_kernels.h"

#include <securec.h>
#include <stdint.h>
#include <vector>

#include "Eigen/Dense"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace {
const char *REALDIV = "RealDiv";           // op name
const size_t REALDIV_OUTPUT_DESC_NUM = 1;  // output dims
const size_t REALDIV_INPUT_NUM = 2;        // input dims
}  // namespace

namespace aicpu {
uint32_t RealDivKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("RealDivKernel::Compute begin.");

  if ((ctx.GetInputsSize() != REALDIV_INPUT_NUM) ||
      (ctx.GetOutputsSize() != REALDIV_OUTPUT_DESC_NUM)) {
    KERNEL_LOG_ERROR(
        "Unexpected RealDiv node, node input size: %zu, node output size: %zu, "
        "node name: %s",
        ctx.GetInputsSize(), ctx.GetOutputsSize(), ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *x = ctx.Input(0);
  if (x == nullptr) {
    KERNEL_LOG_ERROR("RealDiv first input x is null");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *y = ctx.Input(1);
  if (y == nullptr) {
    KERNEL_LOG_ERROR("RealDiv second input y is null");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *z = ctx.Output(0);
  if (z == nullptr) {
    KERNEL_LOG_ERROR("RealDiv output z is null");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  DataType dataType = DataType(x->GetDataType());
  KERNEL_LOG_INFO("RealDiv input type:%d", dataType);
  uint32_t ret = ComputeDiffType(x, y, z, dataType);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }

  KERNEL_LOG_INFO("RealDivKernel::Compute end.");
  return KERNEL_STATUS_OK;
}

uint32_t RealDivKernel::ComputeDiffType(Tensor *x, Tensor *y, Tensor *z,
                                        DataType dataType) {
  switch (dataType) {
    case DT_FLOAT16:
      return ComputeRealdiv<Eigen::half>(x, y, z);
      break;
    case DT_FLOAT:
      return ComputeRealdiv<float>(x, y, z);
      break;
    case DT_DOUBLE:
      return ComputeRealdiv<double>(x, y, z);
      break;
    case DT_UINT8:
      return ComputeRealdiv<uint8_t>(x, y, z);
      break;
    case DT_INT8:
      return ComputeRealdiv<int8_t>(x, y, z);
      break;
    case DT_UINT16:
      return ComputeRealdiv<uint16_t>(x, y, z);
      break;
    case DT_INT16:
      return ComputeRealdiv<int16_t>(x, y, z);
      break;
    case DT_INT32:
      return ComputeRealdiv<int32_t>(x, y, z);
      break;
    case DT_INT64:
      return ComputeRealdiv<int64_t>(x, y, z);
      break;
    default:
      KERNEL_LOG_ERROR("RealDiv invalid input type:%d", dataType);
      return KERNEL_STATUS_PARAM_INVALID;
      break;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t RealDivKernel::ComputeRealdiv(Tensor *x, Tensor *y, Tensor *z) {
  auto xAddr = x->GetData();
  auto yAddr = y->GetData();
  auto zAddr = z->GetData();
  auto xShape = x->GetTensorShape();
  auto yShape = y->GetTensorShape();
  auto zShape = z->GetTensorShape();

  std::vector<int64_t> xDimSize;
  for (int j = 0; j < xShape->GetDims(); j++) {
    xDimSize.push_back(xShape->GetDimSize(j));
  }
  std::vector<int64_t> yDimSize;
  for (int j = 0; j < yShape->GetDims(); j++) {
    yDimSize.push_back(yShape->GetDimSize(j));
  }
  std::vector<int64_t> zDimSize;
  for (int j = 0; j < zShape->GetDims(); j++) {
    zDimSize.push_back(zShape->GetDimSize(j));
  }

  int64_t dim = zShape->GetDims();
  while (int64_t(xDimSize.size()) < dim) {
    xDimSize.insert(xDimSize.begin(), 1);
  }
  while (int64_t(yDimSize.size()) < dim) {
    yDimSize.insert(yDimSize.begin(), 1);
  }
  for (int j = 0; j < dim; j++) {
    if (xDimSize[j] != yDimSize[j] && xDimSize[j] != 1 && yDimSize[j] != 1) {
      KERNEL_LOG_ERROR("The xShape and the yShape can't broadcast.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  uint32_t ret = ComputeDiffShape<T>(dim, (T *)xAddr, (T *)yAddr, (T *)zAddr,
                                     xDimSize, yDimSize, zDimSize);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }

  KERNEL_LOG_INFO("RealDivKernel::Compute success.");
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t RealDivKernel::ComputeDiffShape(int64_t dim, T *xAddr, T *yAddr,
                                         T *zAddr,
                                         std::vector<int64_t> &xDimSize,
                                         std::vector<int64_t> &yDimSize,
                                         std::vector<int64_t> &zDimSize) {
  switch (dim) {
    case 0:
      *((T *)zAddr) = *((T *)xAddr) / *((T *)yAddr);
      break;
    case 1:
      DoCompute<T, 1>(xAddr, yAddr, zAddr, xDimSize, yDimSize, zDimSize);
      break;
    case 2:
      DoCompute<T, 2>(xAddr, yAddr, zAddr, xDimSize, yDimSize, zDimSize);
      break;
    case 3:
      DoCompute<T, 3>(xAddr, yAddr, zAddr, xDimSize, yDimSize, zDimSize);
      break;
    case 4:
      DoCompute<T, 4>(xAddr, yAddr, zAddr, xDimSize, yDimSize, zDimSize);
      break;
    case 5:
      DoCompute<T, 5>(xAddr, yAddr, zAddr, xDimSize, yDimSize, zDimSize);
      break;
    case 6:
      DoCompute<T, 6>(xAddr, yAddr, zAddr, xDimSize, yDimSize, zDimSize);
      break;
    case 7:
      DoCompute<T, 7>(xAddr, yAddr, zAddr, xDimSize, yDimSize, zDimSize);
      break;
    default:
      KERNEL_LOG_ERROR("Don't support %d dims", dim);
      return KERNEL_STATUS_PARAM_INVALID;
      break;
  }
  return KERNEL_STATUS_OK;
}

template <typename T, int32_t dim>
void RealDivKernel::DoCompute(T *xAddr, T *yAddr, T *zAddr,
                              std::vector<int64_t> &xDimSize,
                              std::vector<int64_t> &yDimSize,
                              std::vector<int64_t> &zDimSize) {
  int64_t xDataSize = 1;
  for (int j = 0; j < int(xDimSize.size()); j++) {
    xDataSize *= xDimSize[j];
  }
  int64_t yDataSize = 1;
  for (int j = 0; j < int(yDimSize.size()); j++) {
    yDataSize *= yDimSize[j];
  }
  int64_t zDataSize = 1;
  for (int j = 0; j < int(zDimSize.size()); j++) {
    zDataSize *= zDimSize[j];
  }
  Eigen::TensorMap<Eigen::Tensor<T, 1>> mapX(xAddr, xDataSize);
  Eigen::TensorMap<Eigen::Tensor<T, 1>> mapY(yAddr, yDataSize);
  Eigen::TensorMap<Eigen::Tensor<T, 1>> mapZ(zAddr, zDataSize);
  Eigen::DSizes<Eigen::DenseIndex, dim> xPad;
  Eigen::DSizes<Eigen::DenseIndex, dim> yPad;
  Eigen::DSizes<Eigen::DenseIndex, dim> zPad;
  Eigen::array<int, dim> xBcast;
  Eigen::array<int, dim> yBcast;
  for (int j = 0; j < dim; j++) {
    xPad[j] = xDimSize[j];
    yPad[j] = yDimSize[j];
    zPad[j] = zDimSize[j];
    xBcast[j] = xDimSize[j] == zDimSize[j] ? 1 : zDimSize[j];
    yBcast[j] = yDimSize[j] == zDimSize[j] ? 1 : zDimSize[j];
  }
  mapZ.reshape(zPad) = mapX.reshape(xPad).broadcast(xBcast) /
                       mapY.reshape(yPad).broadcast(yBcast);
}

REGISTER_CPU_KERNEL(REALDIV, RealDivKernel);
}  // namespace aicpu
