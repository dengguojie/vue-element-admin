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
#include "avg_pool1d_avg_matrix.h"

#include <algorithm>
#include <vector>

#include "Eigen/Core"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/kernel_util.h"

namespace {
const char *kAvgPool1DAvgMatrix = "AvgPool1DAvgMatrix";
constexpr int64_t kInputN = 1;
constexpr int64_t kInputC1 = 1;
constexpr int64_t kInputH = 1;
constexpr int64_t kInputC0 = 16;
constexpr int64_t kDimSize = 4;
constexpr int64_t kPadSize = 2;

#define AVGPOOL1DAVGMATRIX_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                                    \
    uint32_t result = DoCompute<TYPE>(CTX);                          \
    if (result != KERNEL_STATUS_OK) {                                \
      KERNEL_LOG_ERROR("AvgPool1DAvgMatrix kernel doCompute failed."); \
      return result;                                                 \
    }                                                                \
    break;                                                           \
  }
static int64_t CalculateWoutput(const int64_t w_in_input, const int64_t pad_l,
                                const int64_t pad_r, const int64_t k_size,
                                const int64_t strides, const bool ceil_mode) {
  int64_t res = 0;
  if (ceil_mode) {
    res = (w_in_input + pad_l + pad_r - k_size + strides - 1) / strides + 1;
  } else {
    res = ((w_in_input + pad_l + pad_r) - k_size) / strides + 1;
  }
  if (pad_l > 0) {
    if (((res - 1) * strides) >= (w_in_input + pad_l)) {
      res--;
    }
  }
  return res;
}
}  // namespace

namespace aicpu {
uint32_t AvgPool1DAvgMatrixCpuKernel::CheckParam(CpuKernelContext &ctx){
  auto output_data_temp = ctx.Output(0)->GetData();
  Tensor *input_tensor = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(output_data_temp, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get output data failed.", kAvgPool1DAvgMatrix);
  auto input_shape = input_tensor->GetTensorShape();
  std::vector<int64_t> dims = input_shape->GetDimSizes();
  KERNEL_CHECK_FALSE((dims.size() >= kDimSize), KERNEL_STATUS_PARAM_INVALID,
                     "%s dims size [%zu] must >= 4.", kAvgPool1DAvgMatrix,
                     dims.size());
  AttrValue *k_size_ptr = ctx.GetAttr("ksize");
  KERNEL_CHECK_NULLPTR(k_size_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get attr ksize fail.", kAvgPool1DAvgMatrix);
  AttrValue *strides_ptr = ctx.GetAttr("strides");
  KERNEL_CHECK_NULLPTR(strides_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get attr strides fail.", kAvgPool1DAvgMatrix);
  AttrValue *ceil_mode_ptr = ctx.GetAttr("ceil_mode");
  KERNEL_CHECK_NULLPTR(ceil_mode_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get attr ceil_mode fail.", kAvgPool1DAvgMatrix);
  AttrValue *pads_ptr = ctx.GetAttr("pads");
  KERNEL_CHECK_NULLPTR(ceil_mode_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get attr pads fail.", kAvgPool1DAvgMatrix);
  AttrValue *count_include_pad_ptr = ctx.GetAttr("count_include_pad");
  KERNEL_CHECK_NULLPTR(count_include_pad_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get attr count_include_pad fail.",
                       kAvgPool1DAvgMatrix);
  std::vector<int64_t> pads = pads_ptr->GetListInt();
  KERNEL_CHECK_FALSE((pads.size() >= kPadSize), KERNEL_STATUS_PARAM_INVALID,
                     "%s pads [%d] must have at least two elements.",
                     kAvgPool1DAvgMatrix, pads.size());
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t AvgPool1DAvgMatrixCpuKernel::DoCompute(CpuKernelContext &ctx) {
  auto output_data_temp = ctx.Output(0)->GetData();
  Tensor *input_tensor = ctx.Input(0);
  auto output_data = reinterpret_cast<T *>(output_data_temp);
  auto input_shape = input_tensor->GetTensorShape();
  auto input_format = input_shape->GetFormat();
  std::vector<int64_t> dims = input_shape->GetDimSizes();
  AttrValue *k_size_ptr = ctx.GetAttr("ksize");
  AttrValue *strides_ptr = ctx.GetAttr("strides");
  AttrValue *ceil_mode_ptr = ctx.GetAttr("ceil_mode");
  AttrValue *pads_ptr = ctx.GetAttr("pads");
  AttrValue *count_include_pad_ptr = ctx.GetAttr("count_include_pad");
  int64_t k_size = k_size_ptr->GetInt();
  int64_t strides = strides_ptr->GetInt();
  KERNEL_CHECK_FALSE((strides != 0), KERNEL_STATUS_PARAM_INVALID,
                     "%s strides [%d] must not be equal to zero.",
                     kAvgPool1DAvgMatrix, strides);
  bool ceil_mode = ceil_mode_ptr->GetBool();
  bool count_include_pad = count_include_pad_ptr->GetBool();
  std::vector<int64_t> pads = pads_ptr->GetListInt();
  int64_t w_in_input = 1;
  if ((input_format == FORMAT_NCHW) || (input_format == FORMAT_NC1HWC0)) {
    w_in_input = dims[3];
  } else if (input_format == FORMAT_NHWC) {
    w_in_input = dims[2];
  } else {
    KERNEL_LOG_ERROR(
        "Format is not in [FORMAT_NHWC or FORMAT_NCHW or FORMAT_NC1HWC0],"
        "current input format is [%d].",
        input_format);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int64_t pad_l = pads[0];
  int64_t pad_r = pads[1];
  int64_t w_output =
      CalculateWoutput(w_in_input, pad_l, pad_r, k_size, strides, ceil_mode);
  int64_t data_num = 0;
  int64_t start = 0;
  int64_t end = 0;
  int64_t out_offset_point = 0;
  int64_t out_put_size = ctx.Output(0)->NumElements();
  // AvgPool1DAvgMatrix Support intput_format(FORMAT_NCHW, FORMAT_NHWC,
  // FORMAT_NC1HWC0) , output_format has only one type FORMAT_NC1HWC0. we only
  // need dim_w from input_format to calculate a new value as w_output.In output
  // process, data_num's value affected by different factors(w_output,
  // strides,k_size, pads, count_include_pad) .Finally,  adding  1/data_num to
  // output_data.
  for (int64_t n = 0; n < kInputN; n++) {
    for (int64_t c1 = 0; c1 < kInputC1; c1++) {
      for (int64_t h = 0; h < kInputH; h++) {
        for (int64_t w = 0; w < w_output; w++) {
          start = strides * w;
          end = strides * w + k_size;
          if (!count_include_pad) {
            start = std::max(start, pad_l);
            end = std::min(end, w_in_input + pad_l);
          } else {
            end = std::min(end, w_in_input + pad_l + pad_r);
          }
          data_num = end - start;
          KERNEL_CHECK_FALSE((data_num != 0), KERNEL_STATUS_PARAM_INVALID,
                             "%s data_num [%d] must not be equal to zero.",
                             kAvgPool1DAvgMatrix, data_num);
          T tmp = static_cast<T>(1.0 / data_num);
          for (int64_t c0 = 0; c0 < kInputC0; c0++) {
            out_offset_point = n * kInputC1 * kInputH * w_output * kInputC0 +
                               c1 * kInputH * w_output * kInputC0 +
                               h * w_output * kInputC0 + w * kInputC0 + c0;
            KERNEL_CHECK_FALSE(
                (out_offset_point < out_put_size), KERNEL_STATUS_PARAM_INVALID,
                "%s out_offset_point [%lld] must < "
                "out_put_size [%lld].",
                kAvgPool1DAvgMatrix, out_offset_point, out_put_size);
            output_data[out_offset_point] = tmp;
          }
        }
      }
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t AvgPool1DAvgMatrixCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *input_tensor = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(input_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get input_tensor fail.", kAvgPool1DAvgMatrix);
  Tensor *output_tensor = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get output_tensor fail.", kAvgPool1DAvgMatrix);
  KERNEL_CHECK_FALSE((CheckParam(ctx) == KERNEL_STATUS_OK),
                     KERNEL_STATUS_PARAM_INVALID, "CheckParam failed."); 
  DataType dt = static_cast<DataType>(input_tensor->GetDataType());
  switch (dt) {
    AVGPOOL1DAVGMATRIX_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    AVGPOOL1DAVGMATRIX_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    AVGPOOL1DAVGMATRIX_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    AVGPOOL1DAVGMATRIX_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    AVGPOOL1DAVGMATRIX_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    AVGPOOL1DAVGMATRIX_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    AVGPOOL1DAVGMATRIX_COMPUTE_CASE(DT_FLOAT, float, ctx)
    AVGPOOL1DAVGMATRIX_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_WARN(
          "AvgPool1DAvgMatrix kernels does not support this data type [%d].", dt);
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kAvgPool1DAvgMatrix, AvgPool1DAvgMatrixCpuKernel);
}  // namespace aicpu