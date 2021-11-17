/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020.
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
#include "crop_and_resize.h"

namespace {
const char *kCropAndResize = "CropAndResize";
const std::string kMethodBiliner = "bilinear";
const std::string kMethodBilinerV2 = "bilinear_v2";
const std::string kMethodNearest = "nearest";
}  // namespace

namespace aicpu {
uint32_t CropAndResizeMsCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  AttrValue *method = ctx.GetAttr("method");
  KERNEL_CHECK_NULLPTR(method, KERNEL_STATUS_PARAM_INVALID,
                       "Get attr:[method] failed.");
  method_ = method->GetString();

  KERNEL_LOG_INFO("CropAndResize method: [%s]", method_.c_str());
  if (method_ != kMethodBiliner && method_ != kMethodBilinerV2 &&
      method_ != kMethodNearest) {
    KERNEL_LOG_ERROR("Invalid attr[method]: [%s], must be in [%s, %s, %s]",
                     method_.c_str(), kMethodBiliner.c_str(),
                     kMethodBilinerV2.c_str(), kMethodNearest.c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  AttrValue *extrapolation_value = ctx.GetAttr("extrapolation_value");
  KERNEL_CHECK_NULLPTR(extrapolation_value, KERNEL_STATUS_PARAM_INVALID,
                       "Get attr:[extrapolation_value] failed.");
  extrapolation_value_ = extrapolation_value->GetFloat();

  // input_0: x
  Tensor *x_tensor = ctx.Input(kInputIndexX);
  KERNEL_CHECK_NULLPTR(x_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "Get input:[0] failed");
  x_dtype_ = static_cast<DataType>(x_tensor->GetDataType());
  std::shared_ptr<TensorShape> x_shape = x_tensor->GetTensorShape();
  x_shape_ = x_shape->GetDimSizes();
  if (x_shape_.size() != 4) {
    KERNEL_LOG_ERROR("The shape size of input[0]:[%zu], should be [4]",
                     x_shape_.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto image_height = x_shape_[1];
  auto image_width = x_shape_[2];

  if (!(image_height > 0 && image_width > 0)) {
    KERNEL_LOG_ERROR(
        "The value of image_height(shape[1] of input[0]): [%lld] and "
        "image_width(shape[2] of input[0]): [%lld] should > 0",
        image_height, image_width);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // input_1: boxes
  Tensor *boxes_tensor = ctx.Input(kInputIndexBoxes);
  KERNEL_CHECK_NULLPTR(boxes_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "Get input:[1] failed");
  std::shared_ptr<TensorShape> boxes_shape = boxes_tensor->GetTensorShape();
  boxes_shape_ = boxes_shape->GetDimSizes();
  KERNEL_CHECK_FALSE(boxes_shape_.size() == 2, KERNEL_STATUS_PARAM_INVALID,
                     "Invalid boxes shape size: [%zu], should be [2]",
                     boxes_shape_.size());
  KERNEL_CHECK_FALSE(boxes_shape_[1] == 4, KERNEL_STATUS_PARAM_INVALID,
                     "The boxes_shape dim[1]: [%lld] not equal to [4]",
                     boxes_shape_[1]);

  // input_2: box_index
  Tensor *box_index_tensor = ctx.Input(kInputIndexBoxIndex);
  KERNEL_CHECK_NULLPTR(box_index_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "Get input:[2] failed");
  std::shared_ptr<TensorShape> box_index_shape =
      box_index_tensor->GetTensorShape();
  box_index_shape_ = box_index_shape->GetDimSizes();

  KERNEL_CHECK_FALSE(boxes_shape_[0] == box_index_shape_[0],
                     KERNEL_STATUS_PARAM_INVALID,
                     "Inconsistent num_boxes, boxes_shape_[0] (shape[0] of input[1]): "
                     "[%lld], box_index_shape_[0] (shape[0] of input[2]): [%lld]",
                     boxes_shape_[0], box_index_shape_[0]);

  // input_3: crop_size
  Tensor *crop_size_tensor = ctx.Input(kInputIndexCropSize);
  KERNEL_CHECK_NULLPTR(crop_size_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "Get input:[3] failed");
  std::shared_ptr<TensorShape> crop_size_shape =
      crop_size_tensor->GetTensorShape();
  crop_size_shape_ = crop_size_shape->GetDimSizes();

  KERNEL_CHECK_FALSE(crop_size_shape_.size() == 1, KERNEL_STATUS_PARAM_INVALID,
                     "Invalid crop_size_shape size (dim of input[3]): [%zu]",
                     crop_size_shape_.size());
  KERNEL_CHECK_FALSE(crop_size_shape_[0] == 2, KERNEL_STATUS_PARAM_INVALID,
                     "Invalid crop_size_shape[0] (shape[0] of input[3]): [%lld]",
                     crop_size_shape_[0]);

  inputs_.push_back(x_tensor);
  inputs_.push_back(boxes_tensor);
  inputs_.push_back(box_index_tensor);
  inputs_.push_back(crop_size_tensor);
  // get output Tensors
  const int kNumOutput = 1;
  for (int i = 0; i < kNumOutput; ++i) {
    Tensor *tensor = ctx.Output(i);
    KERNEL_CHECK_NULLPTR(tensor, KERNEL_STATUS_PARAM_INVALID,
                         "Get output tensor[%d] failed", i)

    if (tensor->NumElements() < 1) {
      KERNEL_LOG_ERROR(
          "Illegal output tensor[%d], "
          "the number of elements in it should be >= [1]",
          i);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    outputs_.push_back(tensor);
  }
  return KERNEL_STATUS_OK;
}

uint32_t CropAndResizeMsCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t res = GetInputAndCheck(ctx);
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res,
                     "GetInputAndCheck failed.");

  std::map<int,
           std::function<uint32_t(
               std::vector<Tensor *> &, std::vector<Tensor *> &,
               const std::vector<int64_t> &x_shape,
               const std::vector<int64_t> &boxes_shape,
               const std::vector<int64_t> &box_index_shape,
               const std::vector<int64_t> &crop_size_shape, std::string &method,
               float extrapolation_value, CpuKernelContext &ctx)>>
      calls;

  calls[DT_INT8] = CalCropAndResize<int8_t>;
  calls[DT_INT16] = CalCropAndResize<int16_t>;
  calls[DT_INT32] = CalCropAndResize<int32_t>;
  calls[DT_INT64] = CalCropAndResize<int64_t>;
  calls[DT_FLOAT16] = CalCropAndResize<Eigen::half>;
  calls[DT_FLOAT] = CalCropAndResize<float>;
  calls[DT_DOUBLE] = CalCropAndResize<double>;
  calls[DT_UINT8] = CalCropAndResize<uint8_t>;
  calls[DT_UINT16] = CalCropAndResize<uint16_t>;

  return calls[x_dtype_](inputs_, outputs_, x_shape_, boxes_shape_,
                         box_index_shape_, crop_size_shape_, method_,
                         extrapolation_value_, ctx);
}

REGISTER_CPU_KERNEL(kCropAndResize, CropAndResizeMsCpuKernel);
}  //  namespace aicpu
