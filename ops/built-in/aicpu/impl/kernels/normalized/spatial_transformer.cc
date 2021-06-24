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
#include "spatial_transformer.h"

#include <cmath>
#include <algorithm>
#include <chrono>

#include "cpu_kernel_utils.h"
#include "cpu_kernel.h"

namespace {
const char *kSpatialTransformer = "SpatialTransformer";
const uint32_t kTotalThetaNumber = 6;

#define STN_COMPUTE_CASE(DTYPE, TYPE, CTX)                             \
  case (DTYPE): {                                                      \
    if (DoCompute<TYPE>(CTX) != KERNEL_STATUS_OK) {                    \
      KERNEL_LOG_ERROR("SpatialTransformer kernel compute failed.");   \
      return KERNEL_STATUS_INNER_ERROR;                                \
    }                                                                  \
    break;                                                             \
  }

#define STN_INNER_COMPUTE_CASE(DTYPE, TYPE, CTX)                       \
  case (DTYPE): {                                                      \
    uint32_t ret = KERNEL_STATUS_OK;                                   \
    if (date_format_ == FORMAT_NCHW) {                                 \
      ret = DoCompute4D<T, TYPE>(ctx);                                 \
    } else if (date_format_ == FORMAT_NC1HWC0) {                       \
      if (stn_ori_channel_ == 1) {                                     \
        ret = DoCompute5D_C1<T, TYPE>(ctx);                            \
      } else {                                                         \
        ret = DoCompute5D<T, TYPE>(ctx);                               \
      }                                                                \
    }                                                                  \
    if (ret != KERNEL_STATUS_OK) {                                     \
      KERNEL_LOG_ERROR("SpatialTransformer kernel compute failed.");   \
      return ret;                                                      \
    }                                                                  \
    break;                                                             \ 
  }                                                               

}

namespace aicpu {
uint32_t SpatialTransformerCpuKernel::GetInputAndCheckValid(CpuKernelContext &ctx) {
  input_tensor_ = ctx.Input(0);
  input_theta_ = ctx.Input(1);
  output_tensor_ = ctx.Output(0);
  if (input_tensor_ == nullptr || input_theta_ == nullptr || output_tensor_ == nullptr) {
    KERNEL_LOG_ERROR("Input or output invalid.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // only support NCHW and NHWC
  date_format_ = input_tensor_->GetTensorShape()->GetFormat();
  if (date_format_ == FORMAT_NCHW) {
    input_n_ = input_tensor_->GetTensorShape()->GetDimSize(0);
    input_c_ = input_tensor_->GetTensorShape()->GetDimSize(1);
    input_h_ = input_tensor_->GetTensorShape()->GetDimSize(2);
    input_w_ = input_tensor_->GetTensorShape()->GetDimSize(3);
    output_h_ = output_tensor_->GetTensorShape()->GetDimSize(2);
    output_w_ = output_tensor_->GetTensorShape()->GetDimSize(3);
  } else if (date_format_ == FORMAT_NC1HWC0) {
    input_n_ = input_tensor_->GetTensorShape()->GetDimSize(0);
    input_c1_ = input_tensor_->GetTensorShape()->GetDimSize(1);
    input_h_ = input_tensor_->GetTensorShape()->GetDimSize(2);
    input_w_ = input_tensor_->GetTensorShape()->GetDimSize(3);
    input_c0_ = input_tensor_->GetTensorShape()->GetDimSize(4);
    input_c_ = input_c1_ * input_c0_;
    output_h_ = output_tensor_->GetTensorShape()->GetDimSize(2);
    output_w_ = output_tensor_->GetTensorShape()->GetDimSize(3);
  }
  else {
    KERNEL_LOG_ERROR("Can't support data format[%d].", date_format_);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  
  bool dims_error_flag = (input_n_ == 0 || input_c_ == 0 || input_h_ == 0 || 
                          input_w_ == 0 || output_h_ == 0 || output_w_ == 0);
  if (dims_error_flag) {
    KERNEL_LOG_ERROR("Dims error.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // get and check data type
  input_data_type_ = static_cast<DataType>(input_tensor_->GetDataType());
  input_theta_type_ = static_cast<DataType>(input_theta_->GetDataType());
  output_data_type_ = static_cast<DataType>(output_tensor_->GetDataType());
  if (input_data_type_ != output_data_type_) {
    KERNEL_LOG_ERROR("Input data type[%s] and output data type[%s] are not same.",
      DTypeStr(input_data_type_).c_str(), DTypeStr(output_data_type_).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // get theta list
  AttrValue *theta_ptr = ctx.GetAttr("default_theta");
  KERNEL_CHECK_NULLPTR(theta_ptr, KERNEL_STATUS_PARAM_INVALID,
    "[%s] get attr default_theta fail.", kSpatialTransformer);
  theta_ = theta_ptr->GetListFloat();

  // get theta valid list
  AttrValue *theta_valid_ptr = ctx.GetAttr("use_default_theta");
  KERNEL_CHECK_NULLPTR(theta_valid_ptr, KERNEL_STATUS_PARAM_INVALID,
    "[%s] get attr use_default_theta fail.", kSpatialTransformer);
  theta_valid_ = theta_valid_ptr->GetListInt();

  // get stn original channel
  AttrValue *ori_channel_ptr = ctx.GetAttr("stn_ori_channel");
  KERNEL_CHECK_NULLPTR(ori_channel_ptr, KERNEL_STATUS_PARAM_INVALID,
    "[%s] get attr stn_ori_channel fail.", kSpatialTransformer);
  stn_ori_channel_ = ori_channel_ptr->GetInt();

  return KERNEL_STATUS_OK;
}

template <typename T, typename T1>
uint32_t SpatialTransformerCpuKernel::DoCompute4D(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("Enter SpatialTransformerCpuKernel::DoCompute4D.");
  const T* input_data_ptr = reinterpret_cast<T *>(input_tensor_->GetData());
  const T1* input_theta = reinterpret_cast<T1 *>(input_theta_->GetData());
  T* output_data_ptr = reinterpret_cast<T *>(output_tensor_->GetData());

  // init ouput_grid and input_grid, [M, 3] * [3, 2] = [M, 2]
  float* input_grid = (float *)malloc(sizeof(float) * output_h_ * output_w_ * 2);
  KERNEL_CHECK_NULLPTR(input_grid, KERNEL_STATUS_INNER_ERROR, "Can't malloc input_grid.");

  // init var
  std::vector<float> theta(6);
  uint32_t input_theta_idx = 0;
  uint32_t output_idx = 0;
  uint32_t input_idx = 0;
  float res, x_floor, y_floor, x_ref_1, y_ref_1, x_ref_0, y_ref_0, x, y = 0;
  int32_t m, n;
  for (uint32_t i = 0; i < input_n_; i++) {
    // init theta
    uint32_t predf_theta_idx = 0;
    for (uint32_t j = 0; j < kTotalThetaNumber; j++) {
      if (theta_valid_[j] == 1) {
        theta[j] = theta_[predf_theta_idx];
        predf_theta_idx++;
      }
      else {
        theta[j] = (float)input_theta[input_theta_idx];
        input_theta_idx++;
      }
    }

    // compute grid
    for (uint32_t s = 0; s < output_h_; ++s) {
      for (uint32_t t = 0; t < output_w_; ++t) {
        uint32_t input_grid_idx = (s * output_w_ + t) * 2;
        float ref_output_grid_1 = (float)s / output_h_ * 2 - 1;
        float ref_output_grid_2 = (float)t / output_w_ * 2 - 1;
        input_grid[input_grid_idx] = (ref_output_grid_1 * theta[0] +
          ref_output_grid_2 * theta[1] + theta[2] + 1) / 2 * input_h_;
        input_grid[input_grid_idx + 1] = (ref_output_grid_1 * theta[3] +
          ref_output_grid_2 * theta[4] + theta[5] + 1) / 2 * input_w_;
      }
    }

    // calc output data
    for (uint32_t j = 0; j < input_c_; j++) {
      uint32_t input_grid_idx = 0;

      for (uint32_t k = 0; k < output_h_ * output_w_; k++) {
        x = input_grid[input_grid_idx];
        y = input_grid[input_grid_idx + 1];
        x_floor = floor(x);
        y_floor = floor(y);
        x_ref_1 = x - x_floor;
        y_ref_1 = y - y_floor;
        x_ref_0 = 1.0f - x_ref_1;
        y_ref_0 = 1.0f - y_ref_1;
        res = 0.0f;

        m = x_floor;
        n = y_floor;
        if (m >= 0 && m < input_h_ && n >= 0 && n < input_w_) {
          res += x_ref_0 * y_ref_0 * (float)input_data_ptr[input_idx + m * input_w_ + n];
        }

        m = x_floor;
        n = y_floor + 1;
        if (m >= 0 && m < input_h_ && n >= 0 && n < input_w_) {
          res += x_ref_0 * y_ref_1 * (float)input_data_ptr[input_idx + m * input_w_ + n];
        }

        m = x_floor + 1;
        n = y_floor;
        if (m >= 0 && m < input_h_ && n >= 0 && n < input_w_) {
          res += x_ref_1 * y_ref_0 * (float)input_data_ptr[input_idx + m * input_w_ + n];
        }

        m = x_floor + 1;
        n = y_floor + 1;
        if (m >= 0 && m < input_h_ && n >= 0 && n < input_w_) {
          res += x_ref_1 * y_ref_1 * (float)input_data_ptr[input_idx + m * input_w_ + n];
        }

        output_data_ptr[output_idx] = (T)res;
        input_grid_idx += 2;
        output_idx++;
      }

      input_idx += input_h_ * input_w_;
    }
  }

  free(input_grid);
  input_grid = nullptr;

  return KERNEL_STATUS_OK;
}

template <typename T, typename T1>
uint32_t SpatialTransformerCpuKernel::DoCompute5D(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("Enter SpatialTransformerCpuKernel::DoCompute5D.");
  const T* input_data_ptr = reinterpret_cast<T *>(input_tensor_->GetData());
  const T1* input_theta = reinterpret_cast<T1 *>(input_theta_->GetData());
  T* output_data_ptr = reinterpret_cast<T *>(output_tensor_->GetData());

  // init ouput_grid and input_grid, [M, 3] * [3, 2] = [M, 2]
  float* input_grid = (float *)malloc(sizeof(float) * output_h_ * output_w_ * 2);
  KERNEL_CHECK_NULLPTR(input_grid, KERNEL_STATUS_INNER_ERROR, "Can't malloc input_grid.");

  // init var
  std::vector<float> theta(6);
  uint32_t input_theta_idx = 0;
  uint32_t output_idx = 0;
  uint32_t input_idx = 0;
  float x_floor, y_floor, x_ref_1, y_ref_1, x_ref_0, y_ref_0, x, y;
  int32_t m, n;
  float *res = (float*)malloc(sizeof(float) * input_c0_);
  KERNEL_CHECK_NULLPTR(input_grid, KERNEL_STATUS_INNER_ERROR, "Can't malloc input_grid.");
  for (uint32_t i = 0; i < input_n_; i++) {
    // init theta
    uint32_t predf_theta_idx = 0;
    for (uint32_t j = 0; j < kTotalThetaNumber; j++) {
      if (theta_valid_[j] == 1) {
        theta[j] = theta_[predf_theta_idx];
        predf_theta_idx++;
      }
      else {
        theta[j] = (float)input_theta[input_theta_idx];
        input_theta_idx++;
      }
    }

    // compute grid
    for (uint32_t s = 0; s < output_h_; ++s) {
      for (uint32_t t = 0; t < output_w_; ++t) {
        uint32_t input_grid_idx = (s * output_w_ + t) * 2;
        float ref_output_grid_1 = (float)s / output_h_ * 2 - 1;
        float ref_output_grid_2 = (float)t / output_w_ * 2 - 1;
        input_grid[input_grid_idx] = (ref_output_grid_1 * theta[0] +
          ref_output_grid_2 * theta[1] + theta[2] + 1) / 2 * input_h_;
        input_grid[input_grid_idx + 1] = (ref_output_grid_1 * theta[3] +
          ref_output_grid_2 * theta[4] + theta[5] + 1) / 2 * input_w_;
      }
    }

    // calc output data
    for (uint32_t j = 0; j < input_c1_; j++) {
      uint32_t input_grid_idx = 0;

      for (uint32_t k = 0; k < output_h_ * output_w_; k++) {
        x = input_grid[input_grid_idx];
        y = input_grid[input_grid_idx + 1];

        x_floor = floor(x);
        y_floor = floor(y);
        x_ref_1 = x - x_floor;
        y_ref_1 = y - y_floor;
        x_ref_0 = 1.0f - x_ref_1;
        y_ref_0 = 1.0f - y_ref_1;

        memset(res, 0.0f, sizeof(float) * input_c0_);

        m = x_floor;
        n = y_floor;
        if (m >= 0 && m < input_h_ && n >= 0 && n < input_w_) {
          uint32_t input_data_spos = input_idx + m * input_w_ * input_c0_ + n * input_c0_;
          for (uint32_t c0_i = 0; c0_i < input_c0_; c0_i++) {
            res[c0_i] += x_ref_0 * y_ref_0 * (float)input_data_ptr[input_data_spos + c0_i];
          }
        }

        m = x_floor;
        n = y_floor + 1;
        if (m >= 0 && m < input_h_ && n >= 0 && n < input_w_) {
          uint32_t input_data_spos = input_idx + m * input_w_ * input_c0_ + n * input_c0_;
          for (uint32_t c0_i = 0; c0_i < input_c0_; c0_i++) {
            res[c0_i] += x_ref_0 * y_ref_1 * (float)input_data_ptr[input_data_spos + c0_i];
          }
        }

        m = x_floor + 1;
        n = y_floor;
        if (m >= 0 && m < input_h_ && n >= 0 && n < input_w_) {
          uint32_t input_data_spos = input_idx + m * input_w_ * input_c0_ + n * input_c0_;
          for (uint32_t c0_i = 0; c0_i < input_c0_; c0_i++) {
            res[c0_i] += x_ref_1 * y_ref_0 * (float)input_data_ptr[input_data_spos + c0_i];
          }
        }

        m = x_floor + 1;
        n = y_floor + 1;
        if (m >= 0 && m < input_h_ && n >= 0 && n < input_w_) {
          uint32_t input_data_spos = input_idx + m * input_w_ * input_c0_ + n * input_c0_;
          for (uint32_t c0_i = 0; c0_i < input_c0_; c0_i++) {
            res[c0_i] += x_ref_1 * y_ref_1 * (float)input_data_ptr[input_data_spos + c0_i];
          }
        }

        for (uint32_t c0_i = 0; c0_i < input_c0_; c0_i++) {
          output_data_ptr[output_idx + c0_i] = (T)res[c0_i];
        }
        input_grid_idx += 2;
        output_idx += input_c0_;
      }

      input_idx += input_h_ * input_w_ * input_c0_;
    }
  }

  free(res);
  res = nullptr;
  free(input_grid);
  input_grid = nullptr;

  return KERNEL_STATUS_OK;
}

template <typename T, typename T1>
uint32_t SpatialTransformerCpuKernel::DoCompute5D_C1(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("Enter SpatialTransformerCpuKernel::DoCompute5D_C1.");
  const T* input_data_ptr = reinterpret_cast<T *>(input_tensor_->GetData());
  const T1* input_theta = reinterpret_cast<T1 *>(input_theta_->GetData());
  T* output_data_ptr = reinterpret_cast<T *>(output_tensor_->GetData());

  // init ouput_grid and input_grid, [M, 3] * [3, 2] = [M, 2]
  float* input_grid = (float *)malloc(sizeof(float) * output_h_ * output_w_ * 2);
  KERNEL_CHECK_NULLPTR(input_grid, KERNEL_STATUS_INNER_ERROR, "Can't malloc input_grid.");

  // init var
  std::vector<float> theta(6);
  uint32_t input_theta_idx = 0;
  uint32_t output_idx = 0;
  uint32_t input_idx = 0;
  float res, x_floor, y_floor, x_ref_1, y_ref_1, x_ref_0, y_ref_0, x, y = 0;
  int32_t m, n;
  for (uint32_t i = 0; i < input_n_; i++) {
    // init theta
    uint32_t predf_theta_idx = 0;
    for (uint32_t j = 0; j < kTotalThetaNumber; j++) {
      if (theta_valid_[j] == 1) {
        theta[j] = theta_[predf_theta_idx];
        predf_theta_idx++;
      }
      else {
        theta[j] = (float)input_theta[input_theta_idx];
        input_theta_idx++;
      }
    }

    // compute grid
    for (uint32_t s = 0; s < output_h_; ++s) {
      for (uint32_t t = 0; t < output_w_; ++t) {
        uint32_t input_grid_idx = (s * output_w_ + t) * 2;
        float ref_output_grid_1 = (float)s / output_h_ * 2 - 1;
        float ref_output_grid_2 = (float)t / output_w_ * 2 - 1;
        input_grid[input_grid_idx] = (ref_output_grid_1 * theta[0] +
          ref_output_grid_2 * theta[1] + theta[2] + 1) / 2 * input_h_;
        input_grid[input_grid_idx + 1] = (ref_output_grid_1 * theta[3] +
          ref_output_grid_2 * theta[4] + theta[5] + 1) / 2 * input_w_;
      }
    }

    // calc output data
    for (uint32_t j = 0; j < input_c1_; j++) {
      uint32_t input_grid_idx = 0;

      for (uint32_t k = 0; k < output_h_ * output_w_; k++) {
        x = input_grid[input_grid_idx];
        y = input_grid[input_grid_idx + 1];

        x_floor = floor(x);
        y_floor = floor(y);
        x_ref_1 = x - x_floor;
        y_ref_1 = y - y_floor;
        x_ref_0 = 1.0f - x_ref_1;
        y_ref_0 = 1.0f - y_ref_1;

        res = 0;

        m = x_floor;
        n = y_floor;
        if (m >= 0 && m < input_h_ && n >= 0 && n < input_w_) {
          res += x_ref_0 * y_ref_0 * (float)input_data_ptr[input_idx + m * input_w_ * input_c0_ + n * input_c0_];
        }

        m = x_floor;
        n = y_floor + 1;
        if (m >= 0 && m < input_h_ && n >= 0 && n < input_w_) {
          res += x_ref_0 * y_ref_1 * (float)input_data_ptr[input_idx + m * input_w_ * input_c0_ + n * input_c0_];
        }

        m = x_floor + 1;
        n = y_floor;
        if (m >= 0 && m < input_h_ && n >= 0 && n < input_w_) {
          res += x_ref_1 * y_ref_0 * (float)input_data_ptr[input_idx + m * input_w_ * input_c0_ + n * input_c0_];
        }

        m = x_floor + 1;
        n = y_floor + 1;
        if (m >= 0 && m < input_h_ && n >= 0 && n < input_w_) {
          res += x_ref_1 * y_ref_1 * (float)input_data_ptr[input_idx + m * input_w_ * input_c0_ + n * input_c0_];
        }

        output_data_ptr[output_idx] = (T)res;
        input_grid_idx += 2;
        output_idx += input_c0_;
      }

      input_idx += input_h_ * input_w_ * input_c0_;
    }
  }

  free(input_grid);
  input_grid = nullptr;

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SpatialTransformerCpuKernel::DoCompute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("Enter SpatialTransformerCpuKernel::DoCompute.");
  
  switch (input_theta_type_) {
    STN_INNER_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    STN_INNER_COMPUTE_CASE(DT_FLOAT, float, ctx)
    STN_INNER_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    STN_INNER_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    STN_INNER_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    STN_INNER_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    STN_INNER_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    STN_INNER_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    STN_INNER_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    STN_INNER_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    STN_INNER_COMPUTE_CASE(DT_DOUBLE, double, ctx)
  default:
    KERNEL_LOG_ERROR("SpatialTransformer kernel data type [%s] not support.", DTypeStr(input_data_type_).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t SpatialTransformerCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t res = GetInputAndCheckValid(ctx);
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
    "GetInputAndCheckValid process failed.");

  switch (input_data_type_) {
    STN_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    STN_COMPUTE_CASE(DT_FLOAT, float, ctx)
    STN_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    STN_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    STN_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    STN_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    STN_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    STN_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    STN_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    STN_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    STN_COMPUTE_CASE(DT_DOUBLE, double, ctx)
  default:
    KERNEL_LOG_ERROR("SpatialTransformer kernel data type [%s] not support.", DTypeStr(input_data_type_).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kSpatialTransformer, SpatialTransformerCpuKernel);
}  // namespace aicpu