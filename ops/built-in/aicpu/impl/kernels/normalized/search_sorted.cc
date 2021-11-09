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
#include "search_sorted.h"
#include <numeric>
#include <atomic>
#include <utility>
#include "Eigen/Core"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const char *kSearchSorted = "SearchSorted";
constexpr size_t kInputSize = 2;
constexpr size_t kOutputSize = 1;
}  // namespace

namespace aicpu {
template <typename S>
std::pair<bool, const S *>CustomizedLowerBound(const S *seq_start,
                                               const S *seq_end,
                                               const S key,
                                               uint64_t sequence_len) {
  while (seq_start < seq_end) {
    auto offset = (seq_end - seq_start) >> 1;
    if (static_cast<uint64_t>(offset) > sequence_len) {
      return std::make_pair(false, seq_start);
    }
    const S *mid = seq_start + offset;
    if (!(key <= *mid)) {
      seq_start = mid + 1;
    } else {
      seq_end = mid;
    }
  }
  return std::make_pair(true, seq_start);
}

uint32_t SearchSortedKernel::CheckShape() {
  std::vector<int64_t> sequence_dims = sequence_shape_;
  std::vector<int64_t> values_dims = values_shape_;
  size_t dim_num = sequence_dims.size();
  sequence_dims.pop_back();
  values_dims.pop_back();
  if (sequence_dims != values_dims && dim_num != 1) {
    KERNEL_LOG_ERROR(
        "Sorted sequence should be [1]-dimensional or has all but "
        "the last dimension matching the dimensions of values,"
        "dim of input[0]:[%zu], dim of input[1]:[%zu].",
        dim_num, values_shape_.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename S>
uint32_t CheckParam(CpuKernelContext &ctx, Tensor *sequence_t, Tensor *values_t,
                    Tensor *output_t, std::vector<int64_t> sequence_shape,
                    std::vector<int64_t> values_shape) {
  size_t search_len = sequence_shape.size();
  if (output_t->NumElements() != values_t->NumElements()) {
    KERNEL_LOG_ERROR(
        "The output dimensions [%lld] must match the dimensions of input "
        "values [%lld]",
        output_t->NumElements(), values_t->NumElements());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto sequence = reinterpret_cast<S*>(sequence_t->GetData());
  size_t list_count =
      std::accumulate(sequence_shape.begin(), sequence_shape.end() - 1, 1,
                      std::multiplies<int>());
  std::atomic<bool> task_flag(true);
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      for (size_t j = 0; j < search_len - 1; j++) {
        if (sequence[i * search_len + j] > sequence[i * search_len + j + 1]) {
          task_flag.store(false);
          KERNEL_LOG_ERROR("The input sequence must be sorted!");
        }
      }
    }
  };
  uint32_t ret = CpuKernelUtils::ParallelFor(ctx, list_count, 1, task);
  if (ret != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("CpuKernelUtils::ParallelFor failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (!task_flag.load()) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t SearchSortedKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  AttrValue *right = ctx.GetAttr("right");
  KERNEL_CHECK_NULLPTR(right, KERNEL_STATUS_PARAM_INVALID,
                       "Get attr:[right] failed.");
  right_ = right->GetBool();

  sequence_t_ = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(sequence_t_, KERNEL_STATUS_PARAM_INVALID,
                       "Get input:[0] failed");
  sequence_dtype_ = static_cast<DataType>(sequence_t_->GetDataType());
  auto sequence_shape = sequence_t_->GetTensorShape();
  sequence_shape_ = sequence_shape->GetDimSizes();

  values_t_ = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(values_t_, KERNEL_STATUS_PARAM_INVALID,
                       "Get input:[1] failed");
  values_dtype_ = static_cast<DataType>(values_t_->GetDataType());
  auto values_shape = values_t_->GetTensorShape();
  values_shape_ = values_shape->GetDimSizes();

  output_t_ = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output_t_, KERNEL_STATUS_PARAM_INVALID,
                       "Get output:[1] failed");
  output_dtype_ = static_cast<DataType>(output_t_->GetDataType());
  auto output_shape = output_t_->GetTensorShape();

  // inputs: sequence, values
  if (ctx.GetInputsSize() != kInputSize) {
    KERNEL_LOG_ERROR(
        "Input number is: [%d], but SearchSorted needs [%zu] inputs.",
        ctx.GetInputsSize(), kInputSize);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // outputs: positions
  if (ctx.GetOutputsSize() != kOutputSize) {
    KERNEL_LOG_ERROR(
        "Output number is: [%d], but SearchSorted needs [%zu] outputs.",
        ctx.GetOutputsSize(), kOutputSize);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return CheckShape();
}

template <typename S, typename T>
uint32_t CalSearchSorted(bool right, Tensor *sequence_t, Tensor *values_t,
                         Tensor *output_t, std::vector<int64_t> sequence_shape,
                         std::vector<int64_t> values_shape,
                         CpuKernelContext &ctx) {
  auto res = CheckParam<S>(ctx, sequence_t, values_t, output_t, sequence_shape,
                           values_shape);
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res,
                     "CheckParam failed, result = [%d].", res);
  auto sequence = reinterpret_cast<S*>(sequence_t->GetData());
  auto values = reinterpret_cast<S*>(values_t->GetData());
  auto output = reinterpret_cast<T*>(output_t->GetData());
  size_t elem_num = values_t->NumElements();
  size_t seq_dim = sequence_shape.size();
  size_t search_repeat = values_shape.back();
  size_t search_len = sequence_shape.back();
  KERNEL_CHECK_FALSE((search_repeat != 0), KERNEL_STATUS_INNER_ERROR,
                     "The value in the shape of input[1] cannot be 0.");
  KERNEL_CHECK_FALSE((search_len > 0), KERNEL_STATUS_INNER_ERROR,
                     "The last dim in the shape of input[0] must be > [0].");
  uint64_t sequence_len = sequence_t->NumElements();

  std::atomic<bool> task_flag(true);
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      auto seq_start = (seq_dim == 1)
                         ? sequence
                         : sequence + (i / search_repeat) * search_len;
      auto bound = CustomizedLowerBound(seq_start,
                                        seq_start + search_len,
                                        values[i], sequence_len);
      if (!bound.first) {
        task_flag.store(false);
        KERNEL_LOG_ERROR("Indices of input[0] is out of range: [%u].",
                         sequence_len);
      }
      output[i] = right ? std::upper_bound(seq_start, seq_start + search_len,
                            values[i]) - seq_start
                        : bound.second - seq_start;
    }
  };

  uint32_t ret = CpuKernelUtils::ParallelFor(ctx, elem_num, 1, task);
  if (ret != KERNEL_STATUS_OK || !task_flag.load()) {
    KERNEL_LOG_ERROR("CpuKernelUtils::ParallelFor failed.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

uint32_t SearchSortedKernel::Compute(CpuKernelContext &ctx) {
  uint32_t res = GetInputAndCheck(ctx);
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res,
                     "GetInputAndCheck failed, result = [%u].", res);

  std::map<int,std::map<int, std::function<uint32_t(
                        bool, Tensor *, Tensor *, Tensor *,
                        std::vector<int64_t>,
                        std::vector<int64_t>, CpuKernelContext &)>>> calls;
  calls[DT_FLOAT][DT_INT32] = CalSearchSorted<float, int>;
  calls[DT_DOUBLE][DT_INT32] = CalSearchSorted<double, int>;
  calls[DT_INT8][DT_INT32] = CalSearchSorted<int8_t, int>;
  calls[DT_INT16][DT_INT32] = CalSearchSorted<int16_t, int>;
  calls[DT_INT32][DT_INT32] = CalSearchSorted<int32_t, int>;
  calls[DT_INT64][DT_INT32] = CalSearchSorted<int64_t, int>;
  calls[DT_FLOAT][DT_INT64] = CalSearchSorted<float, int64_t>;
  calls[DT_DOUBLE][DT_INT64] = CalSearchSorted<double, int64_t>;
  calls[DT_INT8][DT_INT64] = CalSearchSorted<int8_t, int64_t>;
  calls[DT_INT16][DT_INT64] = CalSearchSorted<int16_t, int64_t>;
  calls[DT_INT32][DT_INT64] = CalSearchSorted<int32_t, int64_t>;
  calls[DT_INT64][DT_INT64] = CalSearchSorted<int64_t, int64_t>;

  auto iter = calls.find(sequence_dtype_);
  if (iter == calls.end()) {
    KERNEL_LOG_ERROR(
        "SearchSorted op doesn't support input[0] and input[1] tensor types: "
        "[%s]",
        DTypeStr(sequence_dtype_).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  } else {
    if (iter->second.find(output_dtype_) == iter->second.end()) {
      KERNEL_LOG_ERROR(
          "SearchSorted op doesn't support output[0] tensor types: [%s]",
          DTypeStr(output_dtype_).c_str());
    }
  }
  return iter->second[output_dtype_](right_, sequence_t_, values_t_, output_t_,
                                     sequence_shape_, values_shape_, ctx);
}

REGISTER_CPU_KERNEL(kSearchSorted, SearchSortedKernel);
}  // namespace aicpu
