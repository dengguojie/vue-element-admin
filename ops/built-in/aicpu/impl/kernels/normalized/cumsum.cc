/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
#include "cumsum.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kCumsumInputNum = 2;
const uint32_t kCumsumOutputNum = 1;
const int64_t paralled_data_size = 512 * 1024;
const char *const kCumsum = "Cumsum";
constexpr int64_t kFirstInputIndex = 0;
#define CUMSUM_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                        \
    uint32_t result = CumsumCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                    \
      KERNEL_LOG_ERROR("Cumsum kernel compute failed."); \
      return result;                                     \
    }                                                    \
    break;                                               \
  }
#define CUMSUM_COMPUTE_CASE_COMPLEX(DTYPE, TYPE, IN_TYPE, CTX) \
  case (DTYPE): {                                              \
    uint32_t result = CumsumCompute2<TYPE, IN_TYPE>(CTX);      \
    if (result != KERNEL_STATUS_OK) {                          \
      KERNEL_LOG_ERROR("Cumsum kernel compute failed.");       \
      return result;                                           \
    }                                                          \
    break;                                                     \
  }
}  // namespace

namespace aicpu {
uint32_t CumsumCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kCumsumInputNum, kCumsumOutputNum),
                      "[%s] check input and output failed.", kCumsum);
  // parse params
  KERNEL_HANDLE_ERROR(CumsumCheck(ctx), "[%s] check params failed.", kCumsum);
  auto input_data_type = ctx.Input(kFirstInputIndex)->GetDataType();
  switch (input_data_type) {
    CUMSUM_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    CUMSUM_COMPUTE_CASE(DT_FLOAT, float, ctx)
    CUMSUM_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    CUMSUM_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    CUMSUM_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    CUMSUM_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    CUMSUM_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    CUMSUM_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    CUMSUM_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    CUMSUM_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    CUMSUM_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    CUMSUM_COMPUTE_CASE_COMPLEX(DT_COMPLEX64, std::complex<float>, float, ctx)
    CUMSUM_COMPUTE_CASE_COMPLEX(DT_COMPLEX128, std::complex<double>, double, ctx)
    default:
      KERNEL_LOG_ERROR("Cumsum kernel data type [%s] not support.",
                       DTypeStr(input_data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
uint32_t CumsumCpuKernel::CumsumCheck(const CpuKernelContext &ctx) {
  KERNEL_CHECK_NULLPTR(ctx.Input(kFirstInputIndex)->GetData(),
                       KERNEL_STATUS_PARAM_INVALID, "get input failed.");
  KERNEL_CHECK_NULLPTR(ctx.Input(kFirstInputIndex)->GetTensorShape(),
                       KERNEL_STATUS_PARAM_INVALID,
                       "Get input tensor shape failed.")
  KERNEL_CHECK_NULLPTR(ctx.Output(kFirstInputIndex)->GetData(),
                       KERNEL_STATUS_PARAM_INVALID, "get output failed.");

  if (ctx.Input(1)->GetData() != nullptr) {
    KERNEL_CHECK_FALSE(
        (ctx.Input(1)->GetDataType() == DT_INT32 ||
         ctx.Input(1)->GetDataType() == DT_INT64),
        KERNEL_STATUS_PARAM_INVALID,
        "Data type of axis is not support, axis data type is [%u].",
        ctx.Input(1)->GetDataType());
    KERNEL_CHECK_FALSE(ctx.Input(1)->NumElements() == 1,
                       KERNEL_STATUS_PARAM_INVALID, "axis is out of shape")
    auto axis_data = reinterpret_cast<int32_t *>(ctx.Input(1)->GetData());
    int64_t axis = *axis_data;
    KERNEL_CHECK_FALSE((axis < ctx.Input(0)->GetTensorShape()->GetDims()),
                       KERNEL_STATUS_PARAM_INVALID,
                       "axis is larger than input dims - 1");
    KERNEL_CHECK_FALSE((axis >= -ctx.Input(0)->GetTensorShape()->GetDims()),
                       KERNEL_STATUS_PARAM_INVALID,
                       "axis is lower than -input dims");
  }
  std::vector<int64_t> shape_input =
      ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_output =
      ctx.Output(0)->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((shape_input.size() != 0), KERNEL_STATUS_PARAM_INVALID,
                     "Input must be at least rank 1, got [%zu].",
                     shape_input.size())
  KERNEL_CHECK_FALSE(
      (shape_input.size() == shape_output.size()), KERNEL_STATUS_PARAM_INVALID,
      "The output shape size should be same as the output shape size")
  return KERNEL_STATUS_OK;
}

void CumsumCpuKernel::CumsumGetAttr(const CpuKernelContext &ctx, bool &exclusive, bool &reverse) const {
  exclusive = false;
  AttrValue *exclusive_attr = ctx.GetAttr("exclusive");
  if (exclusive_attr != nullptr) {
    exclusive = exclusive_attr->GetBool();
  }

  reverse = false;
  AttrValue *reverse_attr = ctx.GetAttr("reverse");
  if (reverse_attr != nullptr) {
    reverse = reverse_attr->GetBool();
  }
}

template <typename T>
uint32_t CumsumCpuKernel::CumsumCompute(CpuKernelContext &ctx) {
  auto input_data = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto axis_data = reinterpret_cast<int32_t *>(ctx.Input(1)->GetData());
  bool exclusive;
  bool reverse;
  int32_t axis = 0;
  if (axis_data != nullptr) {
    axis = *axis_data;
  }

  CumsumGetAttr(ctx, exclusive, reverse);

  auto output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto shape = ctx.Input(kFirstInputIndex)->GetTensorShape();
  const int64_t rank = shape->GetDims();
  if (axis < 0) {
    axis += shape->GetDims();
  }
  int64_t inner = 1;
  int64_t outer = 1;
  int64_t depth = 1;
  for (int32_t i = 0; i < rank; ++i) {
    if (i < axis) {
      inner *= shape->GetDimSize(i);
    } else if (i > axis) {
      outer *= shape->GetDimSize(i);
    } else {
      depth = shape->GetDimSize(i);
    }
  }
  int64_t data_num = ctx.Input(kFirstInputIndex)->NumElements();
  int64_t data_size = data_num * static_cast<int64_t>(sizeof(T));
  if (data_size <= paralled_data_size) {
    for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
      int64_t outer_index_adj;
      if (reverse) {
        outer_index_adj = (outer - 1) - outer_index;
      } else {
        outer_index_adj = outer_index;
      }
      for (int64_t inner_index = 0; inner_index < inner; inner_index++) {
        auto accumulator = static_cast<T>(0);
        int64_t inner_index_adj;
        if (reverse) {
          inner_index_adj = (inner - 1) - inner_index;
        } else {
          inner_index_adj = inner_index;
        }
        for (int64_t depth_index = 0; depth_index < depth; depth_index++) {
          int64_t depth_index_adj;
          if (reverse) {
            depth_index_adj = (depth - 1) - depth_index;
          } else {
            depth_index_adj = depth_index;
          }
          int64_t index = outer_index_adj;
          index += inner_index_adj * depth * outer;
          index += depth_index_adj * outer;
          if (exclusive) {
            output_data[index] = accumulator;
            accumulator += input_data[index];
          } else {
            accumulator += input_data[index];
            output_data[index] = accumulator;
          }
        }
      }
    }
  } else {
    auto shard_cumsum = [&](int64_t start, int64_t end) {
      for (int64_t outer_index = start; outer_index < end; ++outer_index) {
        int64_t outer_index_adj;
        if (reverse) {
          outer_index_adj = (outer - 1) - outer_index;
        } else {
          outer_index_adj = outer_index;
        }
        for (int64_t inner_index = 0; inner_index < inner; inner_index++) {
          auto accumulator = static_cast<T>(0);
          int64_t inner_index_adj;
          if (reverse) {
            inner_index_adj = (inner - 1) - inner_index;
          } else {
            inner_index_adj = inner_index;
          }
          for (int64_t depth_index = 0; depth_index < depth; depth_index++) {
            int64_t depth_index_adj;
            if (reverse) {
              depth_index_adj = (depth - 1) - depth_index;
            } else {
              depth_index_adj = depth_index;
            }
            int64_t index = outer_index_adj;
            index += inner_index_adj * depth * outer;
            index += depth_index_adj * outer;
            if (exclusive) {
              output_data[index] = accumulator;
              accumulator += input_data[index];
            } else {
              accumulator += input_data[index];
              output_data[index] = accumulator;
            }
          }
        }
      }
    };
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(
        min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num > outer) {
      max_core_num = outer;
    }
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, outer, outer / max_core_num, shard_cumsum),
                        "CumSum Compute failed.")
  }
  return KERNEL_STATUS_OK;
}
template <typename T, typename T2>
uint32_t CumsumCpuKernel::CumsumCompute2(CpuKernelContext &ctx) {
  auto input_data = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto axis_data = reinterpret_cast<int32_t *>(ctx.Input(1)->GetData());
  bool exclusive;
  bool reverse;
  int32_t axis = 0;
  if (axis_data != nullptr) {
    axis = *axis_data;
  }

  CumsumGetAttr(ctx, exclusive, reverse);

  auto output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto shape = ctx.Input(0)->GetTensorShape();
  const int64_t rank = shape->GetDims();
  if (axis < 0) {
    axis += shape->GetDims();
  }
  int64_t inner = 1;
  int64_t outer = 1;
  int64_t depth = 1;
  for (int32_t i = 0; i < rank; ++i) {
    if (i < axis) {
      inner *= shape->GetDimSize(i);
    } else if (i > axis) {
      outer *= shape->GetDimSize(i);
    } else {
      depth = shape->GetDimSize(i);
    }
  }
  int64_t data_num = ctx.Input(0)->NumElements();
  std::vector<T2> input_data_real(data_num);
  std::vector<T2> input_data_imag(data_num);
  for (int64_t i = 0; i < data_num; ++i) {
    input_data_real[i] = input_data[i].real();
    input_data_imag[i] = input_data[i].imag();
  }
  int64_t data_size = data_num * static_cast<int64_t>(sizeof(T));
  if (data_size <= paralled_data_size) {
    for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
      int64_t outer_index_adj;
      if (reverse) {
        outer_index_adj = (outer - 1) - outer_index;
      } else {
        outer_index_adj = outer_index;
      }
      for (int64_t inner_index = 0; inner_index < inner; inner_index++) {
        auto accumulator_real = static_cast<T2>(0);
        auto accumulator_imag = static_cast<T2>(0);
        int64_t inner_index_adj;
        if (reverse) {
          inner_index_adj = (inner - 1) - inner_index;
        } else {
          inner_index_adj = inner_index;
        }
        for (int64_t depth_index = 0; depth_index < depth; depth_index++) {
          int64_t depth_index_adj;
          if (reverse) {
            depth_index_adj = (depth - 1) - depth_index;
          } else {
            depth_index_adj = depth_index;
          }
          int64_t index = outer_index_adj;
          index += inner_index_adj * depth * outer;
          index += depth_index_adj * outer;
          if (exclusive) {
            output_data[index] =
                std::complex<T2>(accumulator_real, accumulator_imag);
            accumulator_real += input_data_real[index];
            accumulator_imag += input_data_imag[index];
          } else {
            accumulator_real += input_data_real[index];
            accumulator_imag += input_data_imag[index];
            output_data[index] =
                std::complex<T2>(accumulator_real, accumulator_imag);
          }
        }
      }
    }
  } else {
    auto shard_cumsum = [&](int64_t start, int64_t end) {
      for (int64_t outer_index = start; outer_index < end; ++outer_index) {
        int64_t outer_index_adj;
        if (reverse) {
          outer_index_adj = (outer - 1) - outer_index;
        } else {
          outer_index_adj = outer_index;
        }
        for (int64_t inner_index = 0; inner_index < inner; inner_index++) {
          auto accumulator_real = static_cast<T2>(0);
          auto accumulator_imag = static_cast<T2>(0);
          int64_t inner_index_adj;
          if (reverse) {
            inner_index_adj = (inner - 1) - inner_index;
          } else {
            inner_index_adj = inner_index;
          }
          for (int64_t depth_index = 0; depth_index < depth; depth_index++) {
            int64_t depth_index_adj;
            if (reverse) {
              depth_index_adj = (depth - 1) - depth_index;
            } else {
              depth_index_adj = depth_index;
            }
            int64_t index = outer_index_adj;
            index += inner_index_adj * depth * outer;
            index += depth_index_adj * outer;
            if (exclusive) {
              output_data[index] =
                  std::complex<T2>(accumulator_real, accumulator_imag);
              accumulator_real += input_data_real[index];
              accumulator_imag += input_data_imag[index];
            } else {
              accumulator_real += input_data_real[index];
              accumulator_imag += input_data_imag[index];
              output_data[index] =
                  std::complex<T2>(accumulator_real, accumulator_imag);
            }
          }
        }
      }
    };
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(
        min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num > outer) {
      max_core_num = outer;
    }
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(
        ctx, outer, outer / max_core_num, shard_cumsum),
        "CumSum Compute failed.")
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kCumsum, CumsumCpuKernel);
}  // namespace aicpu