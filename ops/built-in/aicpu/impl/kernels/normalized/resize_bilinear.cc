/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All right reserved.
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
#include <vector>
#include <securec.h>
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/sparse_tensor.h"
#include "utils/kernel_util.h"
#include "resize_bilinear.h"

namespace {
constexpr uint32_t kInputNum = 1;
constexpr uint32_t kOutputNum = 1;
const char *kResizeBilinear = "ResizeBilinear";
}  // namespace

namespace aicpu {
float Scaling(size_t in_size, size_t out_size, bool align_corners) {
  return (align_corners && out_size > 1)
             ? (in_size - 1) / static_cast<float>(out_size - 1)
             : in_size / static_cast<float>(out_size);
}

template <typename T>
inline T ComputeLerp(T top_left, T top_right, T bottom_left, T bottom_right,
                     T x_lerp, T y_lerp) {
  T top = top_left + (top_right - top_left) * x_lerp;
  T bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  return top + (bottom - top) * y_lerp;
}

float ScaleGrid(const int x, const float scale) {
  return static_cast<float>(x) * scale;
}

void ComputeInterpolationWeights(const size_t out_size, const size_t in_size,
                                 const float scale,
                                 CachedInterpolation *interpolation) {
  interpolation[out_size].lower = 0;
  interpolation[out_size].upper = 0;
  for (size_t i = 0; i <= out_size - 1; ++i) {
    const float in = ScaleGrid(i, scale);
    const float in_f = std::floor(in);
    interpolation[i].lower =
        std::max(static_cast<size_t>(in_f), static_cast<size_t>(0));
    interpolation[i].upper =
        std::min(static_cast<size_t>(std::ceil(in)), in_size - 1);
    interpolation[i].lerp = in - in_f;
  }
}

uint32_t ResizeBilinearCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  Tensor *input_tensor = ctx.Input(0);
  Tensor *output_tensor = ctx.Output(0);
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "ResizeBilinear check params failed.");

  shape_ = input_tensor->GetTensorShape()->GetDimSizes();
  size_ = output_tensor->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((shape_.size() == 4), KERNEL_STATUS_PARAM_INVALID,
                     "Dim of input[0] must be 4, but got[%zu].", shape_.size());
  KERNEL_CHECK_FALSE((size_.size() == 4), KERNEL_STATUS_PARAM_INVALID,
                     "Dim of output[0] must be 4, but got[%zu].", size_.size());
  AttrValue *pattr_align_corners = ctx.GetAttr("align_corners");
  KERNEL_CHECK_FALSE((pattr_align_corners != nullptr),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Get attr [align_corners] failed.");
  align_corners_ = pattr_align_corners->GetBool();
  dtype_ = input_tensor->GetDataType();

  size_t in_height = shape_[2];
  size_t in_width = shape_[3];
  size_t out_height = size_[2];
  size_t out_width = size_[3];
  height_scale_ = Scaling(in_height, out_height, align_corners_);
  width_scale_ = Scaling(in_width, out_width, align_corners_);
  return KERNEL_STATUS_OK;
}

template <typename T1, typename T2>
uint32_t ResizeBilinearCpuKernel::DoCompute(CpuKernelContext &ctx) {
  auto input_addr = reinterpret_cast<T1 *>(ctx.Input(0)->GetData());
  auto output_addr = reinterpret_cast<T2 *>(ctx.Output(0)->GetData());

  size_t batch_size = shape_[0];
  size_t channel = shape_[1];
  size_t in_height = shape_[2];
  size_t in_width = shape_[3];
  size_t out_height = size_[2];
  size_t out_width = size_[3];
  size_t out_hw_size = out_height * out_width;
  size_t in_hw_size = in_height * in_width;
  size_t bhwc_size = in_hw_size * channel * batch_size;
  int64_t output_num = ctx.Output(0)->NumElements();

  if (out_height == in_height && out_width == in_width) {
    for (size_t i = 0; i < bhwc_size; ++i) {
      output_addr[i] = static_cast<float>(input_addr[i]);
    }
  }

  std::vector<CachedInterpolation> ys(out_height + 1);
  std::vector<CachedInterpolation> xs(out_width + 1);

  ComputeInterpolationWeights(out_height, in_height, height_scale_, ys.data());
  ComputeInterpolationWeights(out_width, in_width, width_scale_, xs.data());

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < channel; ++c) {
      for (size_t h = 0; h < out_height; ++h) {
        const T1 *ys_input_lower_ptr = input_addr + ys[h].lower * in_width;
        const T1 *ys_input_upper_ptr = input_addr + ys[h].upper * in_width;
        const T2 ys_lerp = T2(ys[h].lerp);
        for (size_t w = 0; w < out_width; ++w) {
          const size_t xs_lower = xs[w].lower;
          const size_t xs_upper = xs[w].upper;
          const T2 xs_lerp = T2(xs[w].lerp);
          const T2 top_left(ys_input_lower_ptr[xs_lower]);
          const T2 top_right(ys_input_lower_ptr[xs_upper]);
          const T2 bottom_left(ys_input_upper_ptr[xs_lower]);
          const T2 bottom_right(ys_input_upper_ptr[xs_upper]);
          int64_t output_index = h * out_width + w;
          KERNEL_CHECK_FALSE((output_index < output_num),
              KERNEL_STATUS_INNER_ERROR,
              "The index of output[0]:[%lld] out of range:[%lld].",
              output_index, output_num);
          output_addr[output_index] = ComputeLerp(
              top_left, top_right, bottom_left, bottom_right, xs_lerp, ys_lerp);
        }
      }
      output_addr += out_hw_size;
      input_addr += in_hw_size;
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t ResizeBilinearCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t res = GetInputAndCheck(ctx);
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res,
                     "GetInputAndCheck failed.");

  if (dtype_ == DT_FLOAT16) {
    res = DoCompute<Eigen::half, float>(ctx);
  } else if (dtype_ == DT_FLOAT) {
    res = DoCompute<float, float>(ctx);
  } else {
    KERNEL_LOG_ERROR(
        "ResizeBilinear op doesn't support input tensor types: [%s]",
        DTypeStr(dtype_).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res,
                     "ResizeBilinear Compute failed.");
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kResizeBilinear, ResizeBilinearCpuKernel);
}  // namespace aicpu
