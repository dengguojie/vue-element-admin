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
#include "resize_bilinear_grad.h"
#include "resize_bilinear.h"
#include <vector>
#include <securec.h>
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/sparse_tensor.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kInputNum = 2;
constexpr uint32_t kOutputNum = 1;
const char *kResizeBilinearGrad = "ResizeBilinearGrad";
}  // namespace

namespace aicpu {
uint32_t ResizeBilinearGradCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  Tensor *input_tensor = ctx.Input(0);
  Tensor *size_tensor = ctx.Input(1);
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "ResizeBilinearGrad check params failed.");

  shape_ = input_tensor->GetTensorShape()->GetDimSizes();
  size_ = size_tensor->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((shape_.size() == 4),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Dim of input[0] must be 4, but got[%zu].", shape_.size());
  KERNEL_CHECK_FALSE((size_.size() == 4),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Dim of input[1] must be 4, but got[%zu].", size_.size());
  AttrValue *pattr_align_corners = ctx.GetAttr("align_corners");
  KERNEL_CHECK_FALSE((pattr_align_corners != nullptr),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Get attr [align_corners] failed.");
  align_corners_ = pattr_align_corners->GetBool();
  dtype_ = input_tensor->GetDataType();
  DataType size_dtype = size_tensor->GetDataType();
  KERNEL_CHECK_FALSE((size_dtype == dtype_), KERNEL_STATUS_PARAM_INVALID,
                     "The type of input[0] and input[1] must be the same");

  size_t in_height = shape_[2];
  size_t in_width = shape_[3];
  size_t out_height = size_[2];
  size_t out_width = size_[3];
  height_scale_ = Scaling(out_height, in_height, align_corners_);
  width_scale_ = Scaling(out_width, in_width, align_corners_);
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ResizeBilinearGradCpuKernel::DoCompute(CpuKernelContext &ctx) {
  auto dloss_addr = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_addr = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  auto ret = memset_s(output_addr, ctx.Output(0)->GetDataSize(), 0,
                      ctx.Output(0)->GetDataSize());
  KERNEL_CHECK_FALSE((ret == EOK), ret,
                     "Output buffer memset failed, ret: [%d].", ret);

  size_t batch_size = shape_[0];
  size_t channel = shape_[1];
  size_t in_height = shape_[2];
  size_t in_width = shape_[3];
  size_t out_height = size_[2];
  size_t out_width = size_[3];
  size_t out_hw_size = out_height * out_width;
  size_t in_hw_size = in_height * in_width;

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < channel; ++c) {
      for (size_t h = 0; h < in_height; ++h) {
        const float in_y = static_cast<float>(h) * height_scale_;
        const size_t top_y_index =
            std::max(static_cast<size_t>(floorf(in_y)), static_cast<size_t>(0));
        const size_t bottom_y_index =
            std::min(static_cast<size_t>(ceilf(in_y)), out_height - 1);
        const float y_lerp = in_y - floorf(in_y);
        const float inverse_y_lerp = 1.0 - y_lerp;
        for (size_t w = 0; w < in_width; ++w) {
          const float in_x = static_cast<float>(w) * width_scale_;
          const size_t left_x_index = std::max(
              static_cast<size_t>(floorf(in_x)), static_cast<size_t>(0));
          const size_t right_x_index =
              std::min(static_cast<size_t>(ceilf(in_x)), out_width - 1);
          const float x_lerp = in_x - floorf(in_x);
          const float inverse_x_lerp = 1.0 - x_lerp;
          output_addr[top_y_index * out_width + left_x_index] +=
              dloss_addr[h * in_width + w] * T(inverse_y_lerp * inverse_x_lerp);
          output_addr[top_y_index * out_width + right_x_index] +=
              dloss_addr[h * in_width + w] * T(inverse_y_lerp * x_lerp);
          output_addr[bottom_y_index * out_width + left_x_index] +=
              dloss_addr[h * in_width + w] * T(y_lerp * inverse_x_lerp);
          output_addr[bottom_y_index * out_width + right_x_index] +=
              dloss_addr[h * in_width + w] * T(y_lerp * x_lerp);
        }
      }
      output_addr += out_hw_size;
      dloss_addr += in_hw_size;
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t ResizeBilinearGradCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t res = GetInputAndCheck(ctx);
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res,
                     "GetInputAndCheck failed.");

  if (dtype_ == DT_FLOAT16) {
    res = DoCompute<Eigen::half>(ctx);
  } else if (dtype_ == DT_FLOAT) {
    res = DoCompute<float>(ctx);
  } else {
    KERNEL_LOG_ERROR(
        "ResizeBilinearGrad op doesn't support input tensor types: [%s]",
        DTypeStr(dtype_).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res,
                     "ResizeBilinearGrad Compute failed.");
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kResizeBilinearGrad, ResizeBilinearGradCpuKernel);
}  // namespace aicpu
