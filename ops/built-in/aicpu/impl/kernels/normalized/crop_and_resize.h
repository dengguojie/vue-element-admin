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
#ifndef AICPU_KERNELS_NORMALIZED_CROP_AND_RESIZE_H_
#define AICPU_KERNELS_NORMALIZED_CROP_AND_RESIZE_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/eigen_tensor.h"

namespace aicpu {
class CropAndResizeMsCpuKernel : public CpuKernel {
 public:
  ~CropAndResizeMsCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t GetMethodAndAttr(const CpuKernelContext &ctx);
  uint32_t GetInputIndexX(const CpuKernelContext &ctx);
  uint32_t GetInputBox(const CpuKernelContext &ctx);
  uint32_t GetInputCropSize(const CpuKernelContext &ctx);
  uint32_t GetInputAndCheck(CpuKernelContext &ctx);
  std::vector<Tensor *> inputs_;
  std::vector<Tensor *> outputs_;

  std::string method_;
  float extrapolation_value_ = 0;

  std::vector<int64_t> x_shape_;
  std::vector<int64_t> crop_size_shape_;
  std::vector<int64_t> boxes_shape_;
  std::vector<int64_t> box_index_shape_;
  DataType x_dtype_ = DT_INT32;

  template <typename T>
  struct CropAndResize {
    // We assume that the tensor sizes are correct.
    bool operator()(typename TTypes<T, 4>::Tensor image,
                    typename TTypes<float, 2>::Tensor boxes,
                    typename TTypes<int32_t, 1>::Tensor box_index,
                    const std::string &method_name, float extrapolation_value,
                    typename TTypes<float, 4>::Tensor crops,
                    const CpuKernelContext &ctx) {
      const int64_t imageHeight = image.dimension(1);
      const int64_t imageWidth = image.dimension(2);

      const int64_t numBoxes = crops.dimension(0);
      const int64_t cropHeight = crops.dimension(1);
      const int64_t cropWidth = crops.dimension(2);
      const int64_t depth = crops.dimension(3);

      auto CropAndResizePerBox = [&](int64_t start_box, int64_t limit_box) {
        for (int64_t b = start_box; b < limit_box; ++b) {
          if (method_name == "bilinear_v2") {
            int64_t y1 = boxes(b, 0) * imageHeight;
            int64_t x1 = boxes(b, 1) * imageWidth;
            int64_t y2 =boxes(b, 2) * imageHeight;
            int64_t x2 = boxes(b, 3) * imageWidth;
            int64_t w = 1;
            int64_t h = 1;
            if ((x2 - x1 + 1) > 1) {
              w = x2 - x1 + 1;
            };
            if ((y2 - y1 + 1) > 1) {
              h = y2 - y1 + 1;
            };
            const int32_t b_in = box_index(b);
            for (int64_t y = 0; y < cropHeight; ++y) {
              float y_point =
                  (y + 0.5) * (h / static_cast<float>(cropHeight)) - 0.5;
              int64_t y_base = std::floor(y_point);
              y_base = std::max(static_cast<int64_t>(0), y_base);
              y_base = std::min(y_base, h - 1);
              int64_t y_top = std::ceil(y_point);
              y_top = std::max(static_cast<int64_t>(0), y_top);
              y_top = std::min(y_top, h - 1);
              float y_shift = y_point - y_base;
              for (int64_t x = static_cast<int64_t>(0); x < cropWidth; ++x) {
                float x_point =
                    (x + 0.5) * (w / static_cast<float>(cropWidth)) - 0.5;
                int64_t x_base = std::floor(x_point);
                x_base = std::max(static_cast<int64_t>(0), x_base);
                x_base = std::min(x_base, w - 1);
                int64_t x_top = std::ceil(x_point);
                x_top = std::max(static_cast<int64_t>(0), x_top);
                x_top = std::min(x_top, w - 1);
                float x_shift = x_point - x_base;
                for (int64_t d = 0; d < depth; ++d) {
                  const float top_left(static_cast<float>(
                      image(b_in, y1 + y_base, x1 + x_base, d)));
                  const float top_right(static_cast<float>(
                      image(b_in, y1 + y_base, x1 + x_top, d)));
                  const float bottom_left(static_cast<float>(
                      image(b_in, y1 + y_top, x1 + x_base, d)));
                  const float bottom_right(
                      static_cast<float>(image(b_in, y1 + y_top, x1 + x_top, d)));
                  float ret = top_left * (1 - y_shift) * (1 - x_shift) +
                              bottom_right * y_shift * x_shift +
                              top_right * (1 - y_shift) * x_shift +
                              bottom_left * y_shift * (1 - x_shift);
                  crops(b, y, x, d) = ret;
                }
              }
            }
          } else {
            const float y1 = boxes(b, 0);
            const float x1 = boxes(b, 1);
            const float y2 = boxes(b, 2);
            const float x2 = boxes(b, 3);
            const int32_t b_in = box_index(b);
            const float height_scale =
                (cropHeight > 1)
                    ? (y2 - y1) * (imageHeight - 1) / (cropHeight - 1)
                    : 0;
            const float width_scale =
                (cropWidth > 1)
                    ? (x2 - x1) * (imageWidth - 1) / (cropWidth - 1)
                    : 0;
            for (int64_t y = 0; y < cropHeight; ++y) {
              const float in_y = (cropHeight > 1)
                                    ? y1 * (imageHeight - 1) + y * height_scale
                                    : 0.5 * (y1 + y2) * (imageHeight - 1);
              if (in_y < 0 || in_y > imageHeight - 1) {
                for (int64_t x = 0; x < cropWidth; ++x) {
                  for (int64_t d = 0; d < depth; ++d) {
                    crops(b, y, x, d) = extrapolation_value;
                  }
                }
                continue;
              }
              if (method_name == "bilinear") {
                const int64_t top_y_index = floorf(in_y);
                const int64_t bottom_y_index = ceilf(in_y);
                const float y_lerp = in_y - top_y_index;
                for (int64_t x = 0; x < cropWidth; ++x) {
                  const float in_x =
                      (cropWidth > 1) ? x1 * (imageWidth - 1) + x * width_scale
                                      : 0.5 * (x1 + x2) * (imageWidth - 1);
                  if (in_x < 0 || in_x > imageWidth - 1) {
                    for (int64_t d = 0; d < depth; ++d) {
                      crops(b, y, x, d) = extrapolation_value;
                    }
                    continue;
                  }
                  const int64_t left_x_index = floorf(in_x);
                  const int64_t right_x_index = ceilf(in_x);
                  const float x_lerp = in_x - left_x_index; 
                  for (int64_t d = 0; d < depth; ++d) {
                    const float top_left(static_cast<float>(
                        image(b_in, top_y_index, left_x_index, d)));
                    const float top_right(static_cast<float>(
                        image(b_in, top_y_index, right_x_index, d)));
                    const float bottom_left(static_cast<float>(
                        image(b_in, bottom_y_index, left_x_index, d)));
                    const float bottom_right(static_cast<float>(
                        image(b_in, bottom_y_index, right_x_index, d)));
                    const float top = top_left + (top_right - top_left) * x_lerp;
                    const float bottom =
                        bottom_left + (bottom_right - bottom_left) * x_lerp;
                    crops(b, y, x, d) = top + (bottom - top) * y_lerp;
                  }
                }
              } else {
                for (int64_t x = 0; x < cropWidth; ++x) {
                  const float in_x =
                      (cropWidth > 1) ? x1 * (imageWidth - 1) + x * width_scale
                                      : 0.5 * (x1 + x2) * (imageWidth - 1);
                  if (in_x < 0 || in_x > imageWidth - 1) {
                    for (int64_t d = 0; d < depth; ++d) {
                      crops(b, y, x, d) = extrapolation_value;
                    }
                    continue;
                  }
                  const int64_t closest_x_index = roundf(in_x);
                  const int64_t closest_y_index = roundf(in_y);
                  for (int64_t d = 0; d < depth; ++d) {
                    crops(b, y, x, d) = static_cast<float>(
                        image(b_in, closest_y_index, closest_x_index, d));
                  }
                }
              }
            }
          }
        }
      };
      // Sharding across boxes.
      (void)CpuKernelUtils::ParallelFor(
          ctx, numBoxes, 1, CropAndResizePerBox);
      return true;
    }
  };

  template <typename T>
  static uint32_t CalCropAndResize(std::vector<Tensor *> &inputs,
                                   std::vector<Tensor *> &outputs,
                                   const std::vector<int64_t> &x_shape,
                                   const std::vector<int64_t> &boxes_shape,
                                   const std::string &method,
                                   float extrapolation_value,
                                   const CpuKernelContext &ctx) {
    // input
    EigenTensor image(inputs[0], inputs[0]->GetData());
    EigenTensor boxes(inputs[1], inputs[1]->GetData());
    EigenTensor box_index(inputs[2], inputs[2]->GetData());
    EigenTensor crop_size(inputs[3], inputs[3]->GetData());

    auto batch_size = x_shape[0];
    // need in old kernel: auto depth = x_shape[3];

    auto numBoxes = boxes_shape[0];
    auto crop_size_vec = crop_size.vec<int32_t>();
    int64_t cropHeight = crop_size_vec(0);
    int64_t cropWidth = crop_size_vec(1);
    if (!(cropHeight > 0 && cropWidth > 0)) {
      KERNEL_LOG_ERROR(
          "The value of cropHeight: [%lld] and cropWidth: [%lld] should > 0",
          cropHeight, cropWidth);
      return KERNEL_STATUS_PARAM_INVALID;
    }

    // output
    const int kOutputY = 0;
    EigenTensor output(outputs[kOutputY], outputs[kOutputY]->GetData());

    typename TTypes<int32_t, 1>::Tensor box_index_t =
        box_index.tensor<int32_t, 1>();
    for (int64_t b = 0; b < numBoxes; ++b) {
      if (!(box_index_t(b) >= 0 &&
            box_index_t(b) < static_cast<int>(batch_size))) {
        KERNEL_LOG_ERROR("Invalid box_index[%lld] value: [%d],"
                         "should be in [0, %lld)!",
                         b, box_index_t(b), batch_size);
        return KERNEL_STATUS_PARAM_INVALID;
      }
    }

    if (CropAndResize<T>()(image.tensor<T, 4>(), boxes.tensor<float, 2>(),
                           box_index.tensor<int32_t, 1>(), method,
                           extrapolation_value, output.tensor<float, 4>(),
                           ctx)) {
      return KERNEL_STATUS_OK;
    }

    return KERNEL_STATUS_PARAM_INVALID;
  }
};
}  // namespace aicpu

#endif  // AICPU_CROP_AND_RESIZE_H
