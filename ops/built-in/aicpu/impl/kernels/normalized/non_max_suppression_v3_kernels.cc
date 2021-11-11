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
#include "non_max_suppression_v3_kernels.h"

#include <queue>

#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "cpu_attr_value.h"
#include "cpu_tensor.h"
#include "cpu_tensor_shape.h"
#include "log.h"
#include "status.h"
#include "utils/allocator_utils.h"

namespace {
const char *kNonMaxSuppressionV3 = "NonMaxSuppressionV3";
}

namespace aicpu {
uint32_t NonMaxSuppressionV3CpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("GetInputAndCheck start!! ");
  // get input tensors
  // get boxes with size [num_boxes, 4]
  boxes_ = ctx.Input(0);
  KERNEL_CHECK_FALSE((boxes_ != nullptr), KERNEL_STATUS_PARAM_INVALID,
                     "GetInputAndCheck: get "
                     "input:0 boxes failed.");
  std::shared_ptr<TensorShape> boxes_shape = boxes_->GetTensorShape();
  KERNEL_CHECK_FALSE((boxes_shape != nullptr), KERNEL_STATUS_PARAM_INVALID,
                     "The boxes_shape couldn't be null.");
  int32_t boxes_rank = boxes_shape->GetDims();
  KERNEL_LOG_DEBUG("Input dim size of boxes is %d(boxes_rank) ", boxes_rank);

  if (boxes_rank != 2 || boxes_shape->GetDimSize(1) != 4) {
    KERNEL_LOG_ERROR(
        "The input dim size of boxes must be 2-D and must have 4 columns, "
        "while %d, %lld",
        boxes_rank, boxes_shape->GetDimSize(1));
    return KERNEL_STATUS_PARAM_INVALID;
  }
  num_boxes_ = boxes_shape->GetDimSize(0);

  // get scores with size [num_boxes]
  scores_ = ctx.Input(1);
  KERNEL_CHECK_FALSE((scores_ != nullptr), KERNEL_STATUS_PARAM_INVALID,
                     "GetInputAndCheck: get "
                     "input:1 scores failed.");
  std::shared_ptr<TensorShape> scores_shape = scores_->GetTensorShape();
  KERNEL_CHECK_FALSE((scores_shape != nullptr), KERNEL_STATUS_PARAM_INVALID,
                     "The scores_shape couldn't be null.");
  int32_t scores_rank = scores_shape->GetDims();
  KERNEL_LOG_DEBUG("Input dim size of scores is %d(scores_rank) ", scores_rank);
  KERNEL_CHECK_FALSE((scores_rank == 1), KERNEL_STATUS_PARAM_INVALID,
                     "The input dim size of scores must be 1-D, while %d.",
                     scores_rank);
  KERNEL_CHECK_FALSE((scores_shape->GetDimSize(0) == num_boxes_),
                     KERNEL_STATUS_PARAM_INVALID,
                     "The len of scores must be equal to the number of boxes, "
                     "while dims[%lld], num_boxes_[%d].",
                     scores_shape->GetDimSize(0), num_boxes_);

  // get max_output_size : scalar
  Tensor *max_output_size_tensor = ctx.Input(2);
  KERNEL_CHECK_FALSE((max_output_size_tensor != nullptr),
                     KERNEL_STATUS_PARAM_INVALID,
                     "GetInputAndCheck: get "
                     "input:2 max_output_size failed.");
  max_output_size_ = *static_cast<int32_t *>(max_output_size_tensor->GetData());
  KERNEL_LOG_DEBUG(
      "The max_output_size_data_ptr address %p",
      reinterpret_cast<float *>(max_output_size_tensor->GetData()));

  // get iou_threshold : scalar
  iou_threshold_tensor_ = ctx.Input(3);
  KERNEL_CHECK_FALSE((iou_threshold_tensor_ != nullptr),
                     KERNEL_STATUS_PARAM_INVALID,
                     "GetInputAndCheck: get "
                     "input:3 iou_threshold failed.");

  // get score_threshold: scalar
  score_threshold_tensor_ = ctx.Input(4);
  KERNEL_CHECK_FALSE((score_threshold_tensor_ != nullptr),
                     KERNEL_STATUS_PARAM_INVALID,
                     "GetInputAndCheck: get "
                     "input:4 score_threshold failed.");

  // get output tensors
  output_indices_ = ctx.Output(0);
  KERNEL_CHECK_FALSE((output_indices_ != nullptr), KERNEL_STATUS_PARAM_INVALID,
                     "GetInputAndCheck: get "
                     "output:0 output_indices failed.");

  boxes_scores_dtype_ = static_cast<DataType>(boxes_->GetDataType());
  if (boxes_scores_dtype_ != DT_FLOAT16 && boxes_scores_dtype_ != DT_FLOAT) {
    KERNEL_LOG_ERROR(
        "The dtype of input[0]boxes and scores must be float16 or float32.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  threshold_dtype_ =
      static_cast<DataType>(iou_threshold_tensor_->GetDataType());
  if (threshold_dtype_ != DT_FLOAT16 && threshold_dtype_ != DT_FLOAT) {
    KERNEL_LOG_ERROR("The dtype of input[3]iou_threshold must be float16 or float32.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  KERNEL_LOG_DEBUG("The boxes address %p: ",
                   reinterpret_cast<float *>(boxes_->GetData()));
  KERNEL_LOG_DEBUG("The scores address %p: ",
                   reinterpret_cast<float *>(scores_->GetData()));
  KERNEL_LOG_DEBUG(
      "The max_output_size_tensor address %p: ",
      reinterpret_cast<float *>(max_output_size_tensor->GetData()));
  KERNEL_LOG_DEBUG("The iou_threshold_tensor_ address %p: ",
                   reinterpret_cast<float *>(iou_threshold_tensor_->GetData()));
  KERNEL_LOG_DEBUG(
      "The score_threshold_tensor_ address %p: ",
      reinterpret_cast<float *>(score_threshold_tensor_->GetData()));
  KERNEL_LOG_DEBUG("The output_indices_ address %p: ",
                   reinterpret_cast<float *>(output_indices_->GetData()));

  KERNEL_LOG_INFO("GetInputAndCheck end!!");
  return KERNEL_STATUS_OK;
}

template <typename T>
T inline NonMaxSuppressionV3CpuKernel::IOUSimilarity(const T *box_1,
                                                     const T *box_2) {
  const T ymin_i = std::min<T>(box_1[0], box_1[2]);
  const T xmin_i = std::min<T>(box_1[1], box_1[3]);
  const T ymax_i = std::max<T>(box_1[0], box_1[2]);
  const T xmax_i = std::max<T>(box_1[1], box_1[3]);
  const T ymin_j = std::min<T>(box_2[0], box_2[2]);
  const T xmin_j = std::min<T>(box_2[1], box_2[3]);
  const T ymax_j = std::max<T>(box_2[0], box_2[2]);
  const T xmax_j = std::max<T>(box_2[1], box_2[3]);
  const T area_i = (ymax_i - ymin_i) * (xmax_i - xmin_i);
  const T area_j = (ymax_j - ymin_j) * (xmax_j - xmin_j);
  if (area_i <= static_cast<T>(0) || area_j <= static_cast<T>(0)) {
    return static_cast<T>(0.0);
  }
  const T intersection_ymin = std::max<T>(ymin_i, ymin_j);
  const T intersection_xmin = std::max<T>(xmin_i, xmin_j);
  const T intersection_ymax = std::min<T>(ymax_i, ymax_j);
  const T intersection_xmax = std::min<T>(xmax_i, xmax_j);
  const T intersection_area =
      std::max<T>(intersection_ymax - intersection_ymin, static_cast<T>(0.0)) *
      std::max<T>(intersection_xmax - intersection_xmin, static_cast<T>(0.0));
  if ((area_i + area_j - intersection_area) == static_cast<T>(0)) {
      return static_cast<T>(0.0);
  }
  return intersection_area / (area_i + area_j - intersection_area);
}

template <typename T, typename T_threshold>
uint32_t NonMaxSuppressionV3CpuKernel::DoCompute() {
  KERNEL_LOG_INFO("DoCompute start!!");

  Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> boxes_map(
      reinterpret_cast<T *>(boxes_->GetData()), num_boxes_, 4);
  std::vector<T> scores_data(num_boxes_);
  std::copy_n(reinterpret_cast<T *>(scores_->GetData()), num_boxes_,
              scores_data.begin());

  auto iou_threshold = static_cast<T>(
      *(static_cast<T_threshold *>(iou_threshold_tensor_->GetData())));
  auto score_threshold = static_cast<T>(
      *(static_cast<T_threshold *>(score_threshold_tensor_->GetData())));
  std::unique_ptr<int32_t[]> indices_data(new int32_t[max_output_size_]);
  if (indices_data == nullptr) {
    KERNEL_LOG_ERROR(
        "DoCompute: new indices_data failed");
    return KERNEL_STATUS_INNER_ERROR;
  }

  if (iou_threshold < static_cast<T>(0.0) ||
      iou_threshold > static_cast<T>(1.0)) {
    KERNEL_LOG_ERROR(
        "DoCompute: input[3]iou_threshold must be in the "
        "range [0, 1].");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  struct Candidate {
    int box_index;
    T score;
  };

  auto cmp = [](const Candidate bs_i, const Candidate bs_j) {
    return bs_i.score < bs_j.score;
  };

  std::priority_queue<Candidate, std::deque<Candidate>, decltype(cmp)>
      candidate_priority_queue(cmp);
  for (uint32_t i = 0; i < scores_data.size(); ++i) {
    if (scores_data[i] > score_threshold) {
      candidate_priority_queue.emplace(Candidate({(int)i, scores_data[i]}));
    }
  }

  T similarity = static_cast<T>(0.0);
  Candidate next_candidate = {.box_index = 0, .score = static_cast<T>(0.0)};
  int32_t cnt = 0;

  while (cnt < max_output_size_ && !candidate_priority_queue.empty()) {
    next_candidate = candidate_priority_queue.top();
    candidate_priority_queue.pop();
    // iterate through the previously selected boxes backwards to see if
    // `next_candidate` should be suppressed.
    bool should_suppress = false;
    for (int j = cnt - 1; j >= 0; --j) {
      similarity = IOUSimilarity(&boxes_map(next_candidate.box_index, 0),
                                 &boxes_map(indices_data[j], 0));
      // decide whether to perform suppression
      if (similarity >= iou_threshold) {
        should_suppress = true;
        break;
      }
    }
    if (!should_suppress) {
      indices_data[cnt] = next_candidate.box_index;
      cnt += 1;
      candidate_priority_queue.push(next_candidate);
    }
  }

  std::vector<int64_t> output_shape = {cnt};
  KERNEL_LOG_INFO("The num of selected indices is %d", cnt);
  auto ret = CpuKernelAllocatorUtils::UpdateOutputDataTensor(
      output_shape, DT_INT32, indices_data.get(),
      max_output_size_ * sizeof(int32_t), output_indices_);
  KERNEL_CHECK_FALSE(
      (ret == KERNEL_STATUS_OK), KERNEL_STATUS_INNER_ERROR,
      "UpdateOutputDataTensor failed.")

  KERNEL_LOG_INFO("DoCompute end!!");
  return KERNEL_STATUS_OK;
}

uint32_t NonMaxSuppressionV3CpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("NonMaxSuppressionV3 kernel in.");
  uint32_t res = GetInputAndCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    return res;
  }

  float fp32_data_type = 0.0;
  Eigen::half fp16_data_type(0.0);
  if (boxes_scores_dtype_ == DT_FLOAT16 && threshold_dtype_ == DT_FLOAT16) {
    res = DoCompute<Eigen::half, Eigen::half>();
  } else if (boxes_scores_dtype_ == DT_FLOAT &&
             threshold_dtype_ == DT_FLOAT16) {
    res = DoCompute<float, Eigen::half>();
  } else if (boxes_scores_dtype_ == DT_FLOAT16 &&
             threshold_dtype_ == DT_FLOAT) {
    res = DoCompute<Eigen::half, float>();
  } else if (boxes_scores_dtype_ == DT_FLOAT && threshold_dtype_ == DT_FLOAT) {
    res = DoCompute<float, float>();
  }

  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res,
                     "Compute failed.");

  KERNEL_LOG_INFO("Compute end!!");
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kNonMaxSuppressionV3, NonMaxSuppressionV3CpuKernel);
}  // namespace aicpu
