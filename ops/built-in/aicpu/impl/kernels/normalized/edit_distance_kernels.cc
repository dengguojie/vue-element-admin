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

#include "edit_distance_kernels.h"
#include <map>
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/sparse_tensor.h"

namespace {
const char *EDIT_DISTANCE = "EditDistance";
}

namespace aicpu {
template <typename T>
static void DistanceTask(
    GroupIterable &hypothesis_grouper, GroupIterable &truth_grouper,
    GroupIterable::IteratorStep &hypothesis_iter,
    GroupIterable::IteratorStep &truth_iter,
    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> &output_t,
    std::vector<int64_t> &output_strides, bool &normalize_) {
  auto cmp = std::equal_to<T>();
  while (hypothesis_iter != hypothesis_grouper.end() &&
         truth_iter != truth_grouper.end()) {
    Group truth_i = *truth_iter;
    Group hypothesis_j = *hypothesis_iter;
    std::vector<int64_t> g_truth = truth_i.group();
    std::vector<int64_t> g_hypothesis = hypothesis_j.group();
    auto truth_seq = truth_i.values<T>();
    auto hypothesis_seq = hypothesis_j.values<T>();

    if (g_truth == g_hypothesis) {
      auto loc = std::inner_product(g_truth.begin(), g_truth.end(),
                                    output_strides.begin(), int64_t{0});
      output_t(loc) = LevenshteinDistance<T>(
          std::vector<T>(truth_seq.data(), truth_seq.data() + truth_seq.size()),
          std::vector<T>(hypothesis_seq.data(),
                         hypothesis_seq.data() + hypothesis_seq.size()),
          cmp);
      if (normalize_) output_t(loc) /= truth_seq.size();

      ++hypothesis_iter;
      ++truth_iter;
    } else if (g_truth > g_hypothesis) {  // zero-length truth
      auto loc = std::inner_product(g_hypothesis.begin(), g_hypothesis.end(),
                                    output_strides.begin(), int64_t{0});
      output_t(loc) = hypothesis_seq.size();
      if (normalize_ && output_t(loc) != 0.0f) {
        output_t(loc) = std::numeric_limits<float>::infinity();
      }
      ++hypothesis_iter;
    } else {  // zero-length hypothesis
      auto loc = std::inner_product(g_truth.begin(), g_truth.end(),
                                    output_strides.begin(), int64_t{0});
      output_t(loc) = (normalize_) ? 1.0 : truth_seq.size();
      ++truth_iter;
    }
  }
}
template <typename T>
static void UpdateResult(
    GroupIterable &hypothesis_grouper, GroupIterable &truth_grouper,
    GroupIterable::IteratorStep &hypothesis_iter,
    GroupIterable::IteratorStep &truth_iter,
    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> &output_t,
    std::vector<int64_t> &output_strides, bool &normalize_) {
  while (hypothesis_iter != hypothesis_grouper.end()) {  // zero-length truths
    Group hypothesis_j = *hypothesis_iter;
    std::vector<int64_t> g_hypothesis = hypothesis_j.group();
    auto hypothesis_seq = hypothesis_j.values<T>();
    auto loc = std::inner_product(g_hypothesis.begin(), g_hypothesis.end(),
                                  output_strides.begin(), int64_t{0});
    output_t(loc) = hypothesis_seq.size();
    if (normalize_ && output_t(loc) != 0.0f) {
      output_t(loc) = std::numeric_limits<float>::infinity();
    }
    ++hypothesis_iter;
  }
  while (truth_iter != truth_grouper.end()) {  // missing hypotheses
    Group truth_i = *truth_iter;
    std::vector<int64_t> g_truth = truth_i.group();
    auto truth_seq = truth_i.values<T>();
    auto loc = std::inner_product(g_truth.begin(), g_truth.end(),
                                  output_strides.begin(), int64_t{0});
    output_t(loc) = (normalize_) ? 1.0 : truth_seq.size();
    ++truth_iter;
  }
}

template <typename T>
uint32_t EditDistanceTask(std::vector<Tensor *> &inputs_,
                          std::vector<Tensor *> &outputs_, bool &normalize_) {
  if (inputs_.size() == 0 || outputs_.size() == 0) {
    KERNEL_LOG_ERROR(
        "EditDistanceKernel::EditDistanceTask: input or output is empty.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *hypothesis_indices = inputs_[0];
  Tensor *hypothesis_values = inputs_[1];
  Tensor *truth_indices = inputs_[3];
  Tensor *truth_values = inputs_[4];
  std::vector<int64_t> hypothesis_st_shape(
      (int64_t *)inputs_[2]->GetData(),
      (int64_t *)inputs_[2]->GetData() +
          inputs_[2]->GetTensorShape()->GetDimSize(0));
  std::vector<int64_t> truth_st_shape(
      (int64_t *)inputs_[5]->GetData(),
      (int64_t *)inputs_[5]->GetData() +
          inputs_[5]->GetTensorShape()->GetDimSize(0));
  // Assume indices are sorted in row-major order.
  std::vector<int64_t> sorted_order(truth_st_shape.size());
  std::iota(sorted_order.begin(), sorted_order.end(), 0);

  SparseTensor hypothesis;
  SparseTensor truth;
  if (hypothesis.CreateSparseTensor(hypothesis_indices, hypothesis_values,
                                    hypothesis_st_shape,
                                    sorted_order) != KERNEL_STATUS_OK ||
      truth.CreateSparseTensor(truth_indices, truth_values, truth_st_shape,
                               sorted_order) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("create sparse tensor failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // Group dims 0, 1, ..., RANK - 1.  The very last dim is assumed
  // to store the variable length sequences.
  std::vector<int64_t> group_dims(truth_st_shape.size() - 1);
  std::iota(group_dims.begin(), group_dims.end(), 0);

  float *outptr = (float *)outputs_[0]->GetData();
  if (outptr == NULL) {
    KERNEL_LOG_ERROR(
        "EditDistanceKernel::EditDistanceTask: output[0]->GetData is null.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  std::vector<int64_t> output_shape =
      outputs_[0]->GetTensorShape()->GetDimSizes();
  Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> output_t(
      outptr, outputs_[0]->NumElements());
  output_t.setZero();
  std::vector<int64_t> output_strides(output_shape.size());
  output_strides[output_shape.size() - 1] = 1;
  for (int d = output_shape.size() - 2; d >= 0; --d) {
    output_strides[d] = output_strides[d + 1] * output_shape[d + 1];
  }
  auto hypothesis_grouper = hypothesis.group(group_dims);
  auto truth_grouper = truth.group(group_dims);

  auto hypothesis_iter = hypothesis_grouper.begin();
  auto truth_iter = truth_grouper.begin();
  DistanceTask<T>(hypothesis_grouper, truth_grouper, hypothesis_iter,
                  truth_iter, output_t, output_strides, normalize_);
  UpdateResult<T>(hypothesis_grouper, truth_grouper, hypothesis_iter,
                  truth_iter, output_t, output_strides, normalize_);
  return KERNEL_STATUS_OK;
}

uint32_t EditDistanceKernel::DoCompute() {
  std::map<int, std::function<uint32_t(std::vector<Tensor *> &,
                                       std::vector<Tensor *> &, bool &)>>
      calls;
  calls[DT_INT8] = EditDistanceTask<int8_t>;
  calls[DT_INT16] = EditDistanceTask<int16_t>;
  calls[DT_INT32] = EditDistanceTask<int32_t>;
  calls[DT_INT64] = EditDistanceTask<int64_t>;
  calls[DT_FLOAT16] = EditDistanceTask<Eigen::half>;
  calls[DT_FLOAT] = EditDistanceTask<float>;
  calls[DT_DOUBLE] = EditDistanceTask<double>;
  calls[DT_UINT8] = EditDistanceTask<uint8_t>;
  calls[DT_UINT16] = EditDistanceTask<uint16_t>;
  calls[DT_UINT32] = EditDistanceTask<uint32_t>;
  calls[DT_UINT64] = EditDistanceTask<uint64_t>;
  if (calls.find(param_type_) == calls.end()) {
    KERNEL_LOG_ERROR(
        "EditDistanceKernel op don't support input tensor types: %s",
        typeid(param_type_).name());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return calls[param_type_](inputs_, outputs_, normalize_);
}

uint32_t EditDistanceKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("EditDistanceKernel::GetInputAndCheck start!");
  // get attr: normalize
  AttrValue *normalize = ctx.GetAttr("normalize");
  KERNEL_CHECK_NULLPTR(normalize, KERNEL_STATUS_PARAM_INVALID,
                       "get attr:normalize failed.");
  normalize_ = normalize->GetBool();
  // get input Tensors
  const int num_input = 6;
  for (int i = 0; i < num_input; ++i) {
    Tensor *tensor = ctx.Input(i);
    if (tensor == nullptr) {
      KERNEL_LOG_ERROR(
          "EditDistanceKernel::GetInputAndCheck: get input tensor[%d] failed",
          i);
      return KERNEL_STATUS_PARAM_INVALID;
    }

    if (tensor->NumElements() < 1) {
      KERNEL_LOG_ERROR("EditDistanceKernel::GetInputAndCheck: illegal input tensor[%d]", i);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    inputs_.push_back(tensor);
  }
  // get output Tensors
  const int num_output = 1;
  for (int i = 0; i < num_output; ++i) {
    Tensor *tensor = ctx.Output(i);
    if (tensor == nullptr) {
      KERNEL_LOG_ERROR(
          "EditDistanceKernel::GetInputAndCheck: get output tensor[%d] failed",
          i);
      return KERNEL_STATUS_PARAM_INVALID;
    }

    if (tensor->NumElements() < 1) {
      KERNEL_LOG_ERROR("EditDistanceKernel::GetInputAndCheck: illegal output tensor[%d]", i);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    outputs_.push_back(tensor);
  }
  // get param type
  param_type_ = static_cast<DataType>(inputs_[1]->GetDataType());
  KERNEL_LOG_INFO("EditDistanceKernel::GetInputAndCheck success!");
  return KERNEL_STATUS_OK;
}

uint32_t EditDistanceKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("EditDistanceKernel::Compute start!!");

  uint32_t res = GetInputAndCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    return res;
  }

  res = DoCompute();
  if (res != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("EditDistanceKernel::Compute failed");
    return res;
  }

  KERNEL_LOG_INFO("EditDistanceKernel::Compute success!!");
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(EDIT_DISTANCE, EditDistanceKernel);
}  // namespace aicpu
