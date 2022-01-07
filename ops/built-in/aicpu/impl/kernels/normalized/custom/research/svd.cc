/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "svd.h"

#include <algorithm>
#include "Eigen/SVD"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 3;
const uint32_t kInputNum = 1;
const char *kSvd = "Svd";
constexpr int64_t kParallelDataNumsSize = 2048;

#define SVD_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                     \
    uint32_t result = SvdCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                 \
      KERNEL_LOG_ERROR("Svd kernel compute failed."); \
      return result;                                  \
    }                                                 \
    break;                                            \
  }
}  // namespace

namespace aicpu {
uint32_t SvdCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "[%s] check input and output failed.", kSvd);
  KERNEL_HANDLE_ERROR(SvdCheck(ctx), "[%s] check params failed.", kSvd);
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    SVD_COMPUTE_CASE(DT_FLOAT, float, ctx)
    SVD_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("Svd kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t SvdCpuKernel::SvdCheck(CpuKernelContext &ctx) {
  KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input data failed.")
  KERNEL_CHECK_NULLPTR(ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get output 0 data failed")
  KERNEL_CHECK_NULLPTR(ctx.Output(1)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get output 1 data failed")
  KERNEL_CHECK_NULLPTR(ctx.Output(2)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get output 2 data failed")
  KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetTensorShape(),
                       KERNEL_STATUS_PARAM_INVALID,
                       "Get input tensor shape failed.")
  std::vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((shape_x.size() > 1), KERNEL_STATUS_PARAM_INVALID,
                     "Input x must be at least rank 2.")

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SvdCpuKernel::SvdCompute(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_s = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto output_u = reinterpret_cast<T *>(ctx.Output(1)->GetData());
  auto output_v = reinterpret_cast<T *>(ctx.Output(2)->GetData());
  auto attr_full_matrices = ctx.GetAttr("full_matrices");
  auto attr_compute_uv = ctx.GetAttr("compute_uv");
  bool full_matrices =
      (attr_full_matrices == nullptr) ? false : attr_full_matrices->GetBool();
  bool compute_uv =
      (attr_compute_uv == nullptr) ? true : attr_compute_uv->GetBool();

  std::vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  size_t shape_size = shape_x.size();
  int64_t m = shape_x[shape_size - 2];
  int64_t n = shape_x[shape_size - 1];
  int64_t p = std::min(m, n);
  int64_t size_mn = m * n;
  int64_t size_nn = n * n;
  int64_t size_mm = m * m;
  int64_t size_mp = m * p;
  int64_t size_np = n * p;

  bool empty = (m == 0 || n == 0);
  int options = 0;  // Don't compute singular vectors;
  if (compute_uv) {
    options = full_matrices ? Eigen::ComputeFullU | Eigen::ComputeFullV
                            : Eigen::ComputeThinU | Eigen::ComputeThinV;
  }
  size_t martix_num = 1;
  for (size_t i = 0; i < shape_x.size() - 2; i++) {
    martix_num = martix_num * shape_x.at(i);
  }

  if (martix_num > 0) {
    int64_t data_size = ctx.Input(0)->NumElements() * sizeof(T);
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        MartixXd;

    if (data_size <= kParallelDataNumsSize) {
      for (size_t i = 0; i < martix_num; i++) {
        Eigen::Map<MartixXd> martix_x(input_x + i * size_mn, m, n);
        if (!empty) {
          Eigen::BDCSVD<MartixXd> svd(martix_x, options);
          Eigen::Map<MartixXd> martix_s(output_s + i * p, p, 1);
          martix_s = svd.singularValues().template cast<T>();
          if (compute_uv) {
            if (full_matrices) {
              Eigen::Map<MartixXd> martix_u(output_u + i * size_mm, m, m);
              Eigen::Map<MartixXd> martix_v(output_v + i * size_nn, n, n);
              martix_u = svd.matrixU();
              martix_v = svd.matrixV();
            } else {
              Eigen::Map<MartixXd> martix_u(output_u + i * size_mp, m, p);
              Eigen::Map<MartixXd> martix_v(output_v + i * size_np, n, p);
              martix_u = svd.matrixU();
              martix_v = svd.matrixV();
            }
          }
        } else if (compute_uv && full_matrices) {
          // For an empty matrix where only one dimension is zero, we still set
          // U or V to the unit matrix for the dimension that is non-zero.
          if (m > 0) {
            Eigen::Map<MartixXd> martix_u(output_u + i * size_mm, m, m);
            martix_u = MartixXd::Identity(m, m);
          } else if (n > 0) {
            Eigen::Map<MartixXd> martix_v(output_v + i * size_nn, n, n);
            martix_v = MartixXd::Identity(n, n);
          }
        }
      }
    } else {
      uint32_t min_core_num = 1;
      uint64_t max_core_num =
          std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
      if (max_core_num > martix_num) {
        max_core_num = martix_num;
      }
      auto shard_svd = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
          Eigen::Map<MartixXd> martix_x(input_x + i * size_mn, m, n);
          if (!empty) {
            Eigen::BDCSVD<MartixXd> svd(martix_x, options);
            Eigen::Map<MartixXd> martix_s(output_s + i * p, p, 1);
            martix_s = svd.singularValues().template cast<T>();
            if (compute_uv) {
              if (full_matrices) {
                Eigen::Map<MartixXd> martix_u(output_u + i * size_mm, m, m);
                Eigen::Map<MartixXd> martix_v(output_v + i * size_nn, n, n);
                martix_u = svd.matrixU();
                martix_v = svd.matrixV();
              } else {
                Eigen::Map<MartixXd> martix_u(output_u + i * size_mp, m, p);
                Eigen::Map<MartixXd> martix_v(output_v + i * size_np, n, p);
                martix_u = svd.matrixU();
                martix_v = svd.matrixV();
              }
            }
          } else if (compute_uv && full_matrices) {
            // For an empty matrix where only one dimension is zero, we still
            // set U or V to the unit matrix for the dimension that is non-zero.
            if (m > 0) {
              Eigen::Map<MartixXd> martix_u(output_u + i * size_mm, m, m);
              martix_u = MartixXd::Identity(m, m);
            } else if (n > 0) {
              Eigen::Map<MartixXd> martix_v(output_v + i * size_nn, n, n);
              martix_v = MartixXd::Identity(n, n);
            }
          }
        }
      };
      KERNEL_HANDLE_ERROR(
          CpuKernelUtils::ParallelFor(ctx, martix_num,
                                      martix_num / max_core_num, shard_svd),
          "Svd Compute failed.")
    }
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kSvd, SvdCpuKernel);
}  // namespace aicpu