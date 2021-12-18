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
#include "edit_distance.h"
#include <map>
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/sparse_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kEditDistance = "EditDistance";
}

namespace aicpu {
template <typename T>
static void DistanceTask(
    GroupIterable &hypothesis_grouper, GroupIterable &truth_grouper,
    GroupIterable::IteratorStep &hypothesis_iter,
    GroupIterable::IteratorStep &truth_iter,
    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> &output_t,
    std::vector<int64_t> &output_strides, bool &normalize) {
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
      if (normalize)
        output_t(loc) /= truth_seq.size();

      ++hypothesis_iter;
      ++truth_iter;
    } else if (g_truth > g_hypothesis) {  // zero-length truth
      auto loc = std::inner_product(g_hypothesis.begin(), g_hypothesis.end(),
                                    output_strides.begin(), int64_t{0});
      output_t(loc) = hypothesis_seq.size();
      if (normalize && output_t(loc) != 0.0f) {
        output_t(loc) = std::numeric_limits<float>::infinity();
      }
      ++hypothesis_iter;
    } else {  // zero-length hypothesis
      auto loc = std::inner_product(g_truth.begin(), g_truth.end(),
                                    output_strides.begin(), int64_t{0});
      output_t(loc) = (normalize) ? 1.0 : truth_seq.size();
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
    std::vector<int64_t> &output_strides, bool &normalize) {
  while (hypothesis_iter != hypothesis_grouper.end()) {  // zero-length truths
    Group hypothesis_j = *hypothesis_iter;
    std::vector<int64_t> g_hypothesis = hypothesis_j.group();
    auto hypothesis_seq = hypothesis_j.values<T>();
    auto loc = std::inner_product(g_hypothesis.begin(), g_hypothesis.end(),
                                  output_strides.begin(), int64_t{0});
    output_t(loc) = hypothesis_seq.size();
    if (normalize && output_t(loc) != 0.0f) {
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
    output_t(loc) = (normalize) ? 1.0 : truth_seq.size();
    ++truth_iter;
  }
}

template <typename T>
uint32_t EditDistanceTask(std::vector<Tensor *> &inputs,
                          std::vector<Tensor *> &outputs, bool &normalize) {
  if (inputs.size() == 0 || outputs.size() == 0) {
    KERNEL_LOG_ERROR("EditDistanceTask input or output is empty.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *hypothesis_indices = inputs[0];
  Tensor *hypothesis_values = inputs[1];
  Tensor *truth_indices = inputs[3];
  Tensor *truth_values = inputs[4];
  std::vector<int64_t> hypothesis_st_shape(
      (int64_t *)inputs[2]->GetData(),
      (int64_t *)inputs[2]->GetData() + inputs[2]->GetTensorShape()->GetDimSize(0));
  std::vector<int64_t> truth_st_shape(
      (int64_t *)inputs[5]->GetData(),
      (int64_t *)inputs[5]->GetData() + inputs[5]->GetTensorShape()->GetDimSize(0));
  // Assume indices are sorted in row-major order.
  std::vector<int64_t> sorted_order(truth_st_shape.size());
  std::iota(sorted_order.begin(), sorted_order.end(), 0);

  SparseTensor hypothesis;
  SparseTensor truth;
  if (hypothesis.CreateSparseTensor(hypothesis_indices, hypothesis_values,
                                    hypothesis_st_shape, sorted_order) != KERNEL_STATUS_OK ||
      truth.CreateSparseTensor(truth_indices, truth_values, truth_st_shape,
                               sorted_order) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Create sparse tensor failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // Group dims 0, 1, ..., RANK - 1.  The very last dim is assumed
  // to store the variable length sequences.
  std::vector<int64_t> group_dims(truth_st_shape.size() - 1);
  std::iota(group_dims.begin(), group_dims.end(), 0);

  float *outptr = (float *)outputs[0]->GetData();
  if (outptr == NULL) {
    KERNEL_LOG_ERROR("EditDistanceTask output[0]->GetData is null.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  std::vector<int64_t> output_shape =
      outputs[0]->GetTensorShape()->GetDimSizes();
  Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>> output_t(outptr, outputs[0]->NumElements());
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
                  truth_iter, output_t, output_strides, normalize);
  UpdateResult<T>(hypothesis_grouper, truth_grouper, hypothesis_iter,
                  truth_iter, output_t, output_strides, normalize);
  return KERNEL_STATUS_OK;
}

uint32_t EditDistanceMsCpuKernel::DoCompute() {
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
        "EditDistanceMsCpuKernel op doesn't support input tensor types: [%s]",
        DTypeStr(param_type_).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return calls[param_type_](inputs_, outputs_, normalize_);
}

uint32_t EditDistanceMsCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("GetInputAndCheck start!");
  // get attr: normalize
  AttrValue *normalize = ctx.GetAttr("normalize");
  KERNEL_CHECK_NULLPTR(normalize, KERNEL_STATUS_PARAM_INVALID,
                       "Get attr:[normalize] failed.");
  normalize_ = normalize->GetBool();
  // get input Tensors
  const int kNumInput = 6;
  for (int i = 0; i < kNumInput; ++i) {
    Tensor *tensor = ctx.Input(i);
    KERNEL_CHECK_NULLPTR(tensor, KERNEL_STATUS_PARAM_INVALID,
                         "EditDistance Get input "
                         "tensor[%d] failed",
                         i)

    if (tensor->NumElements() < 1) {
      KERNEL_LOG_ERROR("Illegal input tensor[%d]", i);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    inputs_.push_back(tensor);
  }
  // get output Tensors
  const int kNumOutput = 1;
  for (int i = 0; i < kNumOutput; ++i) {
    Tensor *tensor = ctx.Output(i);
    KERNEL_CHECK_NULLPTR(tensor, KERNEL_STATUS_PARAM_INVALID,
                         "GetInputAndCheck: Get "
                         "output tensor[%d] failed",
                         i)
    if (tensor->NumElements() < 1) {
      KERNEL_LOG_ERROR(
          "GetInputAndCheck: The number of elements:[%lld] in output "
          "tensor[%d] should be > 0!", tensor->NumElements(), i);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    outputs_.push_back(tensor);
  }
  // get param type
  param_type_ = static_cast<DataType>(inputs_[1]->GetDataType());
  KERNEL_LOG_INFO("GetInputAndCheck success!");
  return KERNEL_STATUS_OK;
}

uint32_t EditDistanceMsCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t res = GetInputAndCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    return res;
  }

  res = DoCompute();
  if (res != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Compute failed");
    return res;
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kEditDistance, EditDistanceMsCpuKernel);
}  // namespace aicpu
