/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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

#include "realdiv.h"

#include <stdint.h>
#include <vector>

#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "kernel_util.h"
#include "log.h"
#include "securec.h"
#include "status.h"

namespace {
const char *K_REALDIV = "RealDiv";           // op name
const size_t K_REALDIV_OUTPUT_DESC_NUM = 1;  // output size
const size_t K_REALDIV_INPUT_NUM = 2;        // input size
}  // namespace

namespace aicpu {
uint32_t RealDivKernel::Compute(CpuKernelContext &ctx) {
  if (ctx.GetInputsSize() != K_REALDIV_INPUT_NUM) {
    KERNEL_LOG_ERROR("Ceil node input size should be [%zu], but get [%zu]",
                     K_REALDIV_INPUT_NUM, ctx.GetInputsSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (ctx.GetOutputsSize() != K_REALDIV_OUTPUT_DESC_NUM) {
    KERNEL_LOG_ERROR("Ceil node output size should be [%zu], but get [%zu]",
                     K_REALDIV_OUTPUT_DESC_NUM, ctx.GetOutputsSize());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *x1 = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(x1, KERNEL_STATUS_PARAM_INVALID,
                       "Get input[0], name[x1] failed");
  Tensor *x2 = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(x2, KERNEL_STATUS_PARAM_INVALID,
                       "Get input[1], name[x2] failed");
  Tensor *y = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(y, KERNEL_STATUS_PARAM_INVALID,
                       "Get output[0], name[y] failed");
  DataType data_type = DataType(x1->GetDataType());
  uint32_t ret = ComputeDiffType(x1, x2, y, data_type, ctx);
  if (ret != KERNEL_STATUS_OK) {
    return ret;
  }
  return KERNEL_STATUS_OK;
}

uint32_t RealDivKernel::ComputeDiffType(Tensor *x, Tensor *y, Tensor *z,
                                        DataType data_type,
                                        CpuKernelContext &ctx) {
  switch (data_type) {
    case DT_FLOAT16:
      return ComputeRealdiv<Eigen::half>(x, y, z, ctx);
      break;
    case DT_FLOAT:
      return ComputeRealdiv<float>(x, y, z, ctx);
      break;
    case DT_DOUBLE:
      return ComputeRealdiv<double>(x, y, z, ctx);
      break;
    case DT_UINT8:
      return ComputeRealdiv<uint8_t>(x, y, z, ctx);
      break;
    case DT_INT8:
      return ComputeRealdiv<int8_t>(x, y, z, ctx);
      break;
    case DT_UINT16:
      return ComputeRealdiv<uint16_t>(x, y, z, ctx);
      break;
    case DT_INT16:
      return ComputeRealdiv<int16_t>(x, y, z, ctx);
      break;
    case DT_INT32:
      return ComputeRealdiv<int32_t>(x, y, z, ctx);
      break;
    case DT_INT64:
      return ComputeRealdiv<int64_t>(x, y, z, ctx);
      break;
    default:
      KERNEL_LOG_ERROR("RealDiv invalid input type[%s]",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
      break;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t RealDivKernel::ComputeRealdiv(Tensor *x, Tensor *y, Tensor *z,
                                       CpuKernelContext &ctx) {
  auto x_addr = x->GetData();
  auto y_addr = y->GetData();
  auto z_addr = z->GetData();
  auto x_shape = x->GetTensorShape();
  auto y_shape = y->GetTensorShape();
  auto z_shape = z->GetTensorShape();

  std::vector<int64_t> x_dim_size;
  for (int j = 0; j < x_shape->GetDims(); j++) {
    x_dim_size.push_back(x_shape->GetDimSize(j));
  }
  std::vector<int64_t> y_dim_size;
  for (int j = 0; j < y_shape->GetDims(); j++) {
    y_dim_size.push_back(y_shape->GetDimSize(j));
  }
  std::vector<int64_t> z_dim_size;
  for (int j = 0; j < z_shape->GetDims(); j++) {
    z_dim_size.push_back(z_shape->GetDimSize(j));
  }

  int64_t dim = z_shape->GetDims();
  while (int64_t(x_dim_size.size()) < dim) {
    x_dim_size.insert(x_dim_size.begin(), 1);
  }
  while (int64_t(y_dim_size.size()) < dim) {
    y_dim_size.insert(y_dim_size.begin(), 1);
  }
  for (int j = 0; j < dim; j++) {
    if (x_dim_size[j] != y_dim_size[j] && x_dim_size[j] != 1 &&
        y_dim_size[j] != 1) {
      KERNEL_LOG_ERROR("The x1_shape and x2_shape can't broadcast.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  uint32_t ret = ComputeDiffShape<T>(dim, (T *)x_addr, (T *)y_addr, (T *)z_addr,
                                     x_dim_size, y_dim_size, z_dim_size, ctx);
  if (ret != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("ComputeDiffShape function failed");
    return ret;
  }

  KERNEL_LOG_INFO("RealDivKernel::Compute success.");
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t RealDivKernel::ComputeDiffShape(int64_t dim, T *x_addr, T *y_addr,
                                         T *z_addr,
                                         std::vector<int64_t> &x_dim_size,
                                         std::vector<int64_t> &y_dim_size,
                                         std::vector<int64_t> &z_dim_size,
                                         CpuKernelContext &ctx) {
  switch (dim) {
    case 0:
      *((T *)z_addr) = *((T *)x_addr) / *((T *)y_addr);
      break;
    case 1:
      return DoCompute<T, 1>(x_addr, y_addr, z_addr, x_dim_size, y_dim_size,
                             z_dim_size, ctx);
    case 2:
      return DoCompute<T, 2>(x_addr, y_addr, z_addr, x_dim_size, y_dim_size,
                             z_dim_size, ctx);
    case 3:
      return DoCompute<T, 3>(x_addr, y_addr, z_addr, x_dim_size, y_dim_size,
                             z_dim_size, ctx);
    case 4:
      return DoCompute<T, 4>(x_addr, y_addr, z_addr, x_dim_size, y_dim_size,
                             z_dim_size, ctx);
    case 5:
      return DoCompute<T, 5>(x_addr, y_addr, z_addr, x_dim_size, y_dim_size,
                             z_dim_size, ctx);
    case 6:
      return DoCompute<T, 6>(x_addr, y_addr, z_addr, x_dim_size, y_dim_size,
                             z_dim_size, ctx);
    case 7:
      return DoCompute<T, 7>(x_addr, y_addr, z_addr, x_dim_size, y_dim_size,
                             z_dim_size, ctx);
    case 8:
      return DoCompute<T, 8>(x_addr, y_addr, z_addr, x_dim_size, y_dim_size,
                             z_dim_size, ctx);
    default:
      KERNEL_LOG_ERROR("RealDiv op don't support [%d] dims", dim);
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T, int32_t dim>
uint32_t RealDivKernel::DoCompute(T *x_addr, T *y_addr, T *z_addr,
                                  std::vector<int64_t> &x_dim_size,
                                  std::vector<int64_t> &y_dim_size,
                                  std::vector<int64_t> &z_dim_size,
                                  CpuKernelContext &ctx) {
  int64_t x_data_size = 1;
  for (int j = 0; j < int(x_dim_size.size()); j++) {
    x_data_size *= x_dim_size[j];
  }
  int64_t y_data_size = 1;
  for (int j = 0; j < int(y_dim_size.size()); j++) {
    y_data_size *= y_dim_size[j];
  }
  int64_t z_data_size = 1;
  for (int j = 0; j < int(z_dim_size.size()); j++) {
    z_data_size *= z_dim_size[j];
  }
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> map_x(x_addr, x_data_size);
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> map_y(y_addr, y_data_size);
  Eigen::TensorMap<Eigen::Tensor<T, 1>> map_z(z_addr, z_data_size);
  Eigen::DSizes<Eigen::DenseIndex, dim> x_pad;
  Eigen::DSizes<Eigen::DenseIndex, dim> y_pad;
  Eigen::array<int, dim> x_bcast;
  Eigen::array<int, dim> y_bcast;
  for (int j = 0; j < dim; j++) {
    x_pad[j] = x_dim_size[j];
    y_pad[j] = y_dim_size[j];
    x_bcast[j] = x_dim_size[j] == z_dim_size[j] ? 1 : z_dim_size[j];
    y_bcast[j] = y_dim_size[j] == z_dim_size[j] ? 1 : z_dim_size[j];
  }
  Eigen::Tensor<T, dim, Eigen::RowMajor> map_x_broad = map_x.reshape(x_pad).broadcast(x_bcast);
  Eigen::Tensor<T, dim, Eigen::RowMajor> map_y_broad = map_y.reshape(y_pad).broadcast(y_bcast);
  auto shard_realdiv = [&](size_t start, size_t end) {
    Eigen::TensorMap<Eigen::Tensor<T, 1>> map_x_shard(
        map_x_broad.data() + start, end - start);
    Eigen::TensorMap<Eigen::Tensor<T, 1>> map_y_shard(
        map_y_broad.data() + start, end - start);
    Eigen::TensorMap<Eigen::Tensor<T, 1>> map_z_shard(map_z.data() + start,
                                                      end - start);
    map_z_shard = map_x_shard / map_y_shard;
  };
  uint32_t ret =
      CpuKernelUtils::ParallelFor(ctx, z_data_size, 1, shard_realdiv);
  if (ret != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("CpuKernelUtils::ParallelFor failed");
    return ret;
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(K_REALDIV, RealDivKernel);
}  // namespace aicpu
