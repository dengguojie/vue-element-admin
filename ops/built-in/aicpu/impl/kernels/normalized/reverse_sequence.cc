/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
#include "reverse_sequence.h"
#include "Eigen/Core"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const char *kReverseSequence = "ReverseSequence";
const int kOutputIndex = 2;
const int kEven = 2;
}

namespace aicpu {
template <typename Tlen>
static void CheckSequence(int seq_dim, const Tlen *seq,
                          std::vector<int64_t> &shape,
                          std::vector<int64_t> &seq_lengths_shape) {
  for (int64_t d = 0; d < static_cast<int64_t>(seq_lengths_shape[0]); d++) {
    if (seq[d] < 0) {
      KERNEL_LOG_ERROR("Invalid seq_lengths value[%d]: [%lu]", d, seq[d]);
      return;
    }
    if (seq[d] > shape[seq_dim]) {
      KERNEL_LOG_ERROR("CheckSequence, seq[%d]: [%lu], shape[%d]: [%lu]", d,
                       seq[d], seq_dim, shape[seq_dim]);
      return;
    }
  }
}
template <typename T, typename Tlen>
uint32_t CalReverseSequence(int seq_dim, int batch_dim,
                            const std::vector<void *> &ioAddrs,
                            std::vector<int64_t> &shape,
                            std::vector<int64_t> &seq_lengths_shape,
                            CpuKernelContext &ctx) {
  // inputs
  T *input = reinterpret_cast<T *>(ioAddrs[0]);
  Tlen *seq = reinterpret_cast<Tlen *>(ioAddrs[1]);
  // outputs
  T *output = reinterpret_cast<T *>(ioAddrs[kOutputIndex]);

  CheckSequence(seq_dim, seq, shape, seq_lengths_shape);

  int seq_step = 1;
  for (int i = seq_dim + 1; i < static_cast<int>(shape.size()); i++) {
    seq_step *= shape[i];
  }

  int skip_size = 1;
  for (int i = seq_dim; i < static_cast<int>(shape.size()); i++) {
    skip_size *= shape[i];
  }
  int run_len = seq_step;

  int batch_size = 1;
  for (int i = batch_dim + 1; i < static_cast<int>(shape.size()); ++i) {
    batch_size *= shape[i];
  }

  int total_size = 1;
  for (int i = 0; i < static_cast<int>(shape.size()); ++i) {
    total_size *= shape[i];
  }
  KERNEL_CHECK_FALSE((shape[seq_dim] != 0), KERNEL_STATUS_PARAM_INVALID,
                     "The shape[%d] of input[0] cannot be 0.", seq_dim);
  KERNEL_CHECK_FALSE((batch_size != 0), KERNEL_STATUS_PARAM_INVALID,
                     "The value of batch_size cannot be 0.");
  KERNEL_CHECK_FALSE((shape[batch_dim] != 0), KERNEL_STATUS_PARAM_INVALID,
                     "The shape[%d] of input[0] cannot be 0.", batch_dim);
  int n = total_size / (run_len * shape[seq_dim]);
  bool parallel_in = run_len > n;
  const int64_t kMaxCoreNum = std::max(
      static_cast<uint32_t>(1), aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

  auto shard = [&](const int64_t start, const int64_t end) {
    for (int j = start; j < end; ++j) {  // 0~n
      int begin = run_len * shape[seq_dim] * j;
      auto shard_in = [&](int64_t start_in, int64_t end_in) {
        for (int r = start_in; r < end_in; ++r) {
          int offset = r + begin;
          int reverse_num = seq[offset / batch_size % shape[batch_dim]];
          for (int i = 0; i < shape[seq_dim]; ++i) {
            if (i < reverse_num / kEven) {
              output[i * seq_step + offset] =
                  input[(reverse_num - i - 1) * seq_step + offset];
              output[(reverse_num - i - 1) * seq_step + offset] =
                  input[i * seq_step + offset];
            }
            if (i >= reverse_num ||
                (i == reverse_num / kEven && reverse_num % kEven)) {
              output[i * seq_step + offset] = input[i * seq_step + offset];
            }
          }
        }
      };
      if (parallel_in) {
        (void)CpuKernelUtils::ParallelFor(ctx, run_len, run_len / kMaxCoreNum, shard_in);
      } else {
        shard_in(0, run_len);
      }
    }
  };
  if (!parallel_in) {
    uint32_t ret = CpuKernelUtils::ParallelFor(ctx, n, n / kMaxCoreNum, shard);
    if (ret != KERNEL_STATUS_OK) {
      KERNEL_LOG_ERROR("CpuKernelUtils::ParallelFor failed");
      return KERNEL_STATUS_INNER_ERROR;
    }
  } else {
    shard(0, n);
  }

  return KERNEL_STATUS_OK;
}
uint32_t ReverseSequenceMsCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  AttrValue *seq_dim = ctx.GetAttr("seq_dim");
  KERNEL_CHECK_NULLPTR(seq_dim, KERNEL_STATUS_PARAM_INVALID,
                       "Get attr:[seq_dim] failed.");
  seq_dim_ = seq_dim->GetInt();

  AttrValue *batch_dim = ctx.GetAttr("batch_dim");
  KERNEL_CHECK_NULLPTR(batch_dim, KERNEL_STATUS_PARAM_INVALID,
                       "Get attr:[batch_dim] failed.");
  batch_dim_ = batch_dim->GetInt();

  // input_0: x
  Tensor *x_tensor = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(x_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "Get input:[0] failed")
  x_dtype_ = static_cast<DataType>(x_tensor->GetDataType());
  std::shared_ptr<TensorShape> x_shape = x_tensor->GetTensorShape();

  for (auto i = 0; i < x_shape->GetDims(); i++) {
    x_shape_.emplace_back(x_shape->GetDimSize(i));
  }

  // input_1: seq_lengths
  Tensor *seq_lengths_tensor = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(seq_lengths_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "Get input:[1] failed")
  seq_lengths_dtype_ = static_cast<DataType>(seq_lengths_tensor->GetDataType());
  std::shared_ptr<TensorShape> seq_lengths_shape =
      seq_lengths_tensor->GetTensorShape();
  for (auto i = 0; i < seq_lengths_shape->GetDims(); i++) {
    seq_lengths_shape_.emplace_back(seq_lengths_shape->GetDimSize(i));
  }

  if (seq_lengths_dtype_ != DT_INT32 && seq_lengths_dtype_ != DT_INT64) {
    KERNEL_LOG_ERROR("Invalid type of seq_lengths: [%s]",
                     DTypeStr(seq_lengths_dtype_).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (seq_lengths_shape_.size() != 1) {
    KERNEL_LOG_ERROR("Invalid seq_lengths shape size: [%d]",
                     seq_lengths_shape_.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (batch_dim_ == seq_dim_ ||
      static_cast<uint64_t>(seq_dim_) >= x_shape_.size() ||
      static_cast<uint64_t>(batch_dim_) >= x_shape_.size()) {
    KERNEL_LOG_ERROR("Invalid batch_dim_: [%d], seq_dim_: [%d], x dims:[ %d]",
                     batch_dim_, seq_dim_, x_shape_.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (seq_lengths_shape_[0] !=
      static_cast<int64_t>(x_shape->GetDimSize(batch_dim_))) {
    KERNEL_LOG_ERROR("seq_lengths_shape_[0] != x_shape.dim(%d) size: [%d]",
                     batch_dim_, x_shape->GetDimSize(batch_dim_));
    return KERNEL_STATUS_PARAM_INVALID;
  }

  Tensor *output_tensor = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "Get output:[0] failed")
  ioAddrs_.push_back(reinterpret_cast<void *>(x_tensor->GetData()));
  ioAddrs_.push_back(reinterpret_cast<void *>(seq_lengths_tensor->GetData()));
  ioAddrs_.push_back(reinterpret_cast<void *>(output_tensor->GetData()));

  KERNEL_LOG_INFO("Parse done, seq_dim: [%d], batch_dim: %d, x_dtype: [%d]",
                  seq_dim_, batch_dim_, x_dtype_);

  return KERNEL_STATUS_OK;
}

uint32_t ReverseSequenceMsCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t res = GetInputAndCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    return res;
  }

  std::map<
      int,
      std::map<int, std::function<uint32_t(
                        int, int, std::vector<void *> &, std::vector<int64_t> &,
                        std::vector<int64_t> &, CpuKernelContext &)>>>
      calls;
  calls[DT_FLOAT16][DT_INT32] = CalReverseSequence<Eigen::half, int32_t>;
  calls[DT_FLOAT][DT_INT32] = CalReverseSequence<float, int32_t>;
  calls[DT_DOUBLE][DT_INT32] = CalReverseSequence<double, int32_t>;
  calls[DT_INT8][DT_INT32] = CalReverseSequence<int8_t, int32_t>;
  calls[DT_INT16][DT_INT32] = CalReverseSequence<int16_t, int32_t>;
  calls[DT_INT32][DT_INT32] = CalReverseSequence<int32_t, int32_t>;
  calls[DT_INT64][DT_INT32] = CalReverseSequence<int64_t, int32_t>;
  calls[DT_UINT8][DT_INT32] = CalReverseSequence<uint8_t, int32_t>;
  calls[DT_UINT16][DT_INT32] = CalReverseSequence<uint16_t, int32_t>;
  calls[DT_UINT32][DT_INT32] = CalReverseSequence<uint32_t, int32_t>;
  calls[DT_UINT64][DT_INT32] = CalReverseSequence<uint64_t, int32_t>;
  calls[DT_BOOL][DT_INT32] = CalReverseSequence<bool, int32_t>;

  calls[DT_FLOAT16][DT_INT64] = CalReverseSequence<Eigen::half, int64_t>;
  calls[DT_FLOAT][DT_INT64] = CalReverseSequence<float, int64_t>;
  calls[DT_DOUBLE][DT_INT64] = CalReverseSequence<double, int64_t>;
  calls[DT_INT8][DT_INT64] = CalReverseSequence<int8_t, int64_t>;
  calls[DT_INT16][DT_INT64] = CalReverseSequence<int16_t, int64_t>;
  calls[DT_INT32][DT_INT64] = CalReverseSequence<int32_t, int64_t>;
  calls[DT_INT64][DT_INT64] = CalReverseSequence<int64_t, int64_t>;
  calls[DT_UINT8][DT_INT64] = CalReverseSequence<uint8_t, int64_t>;
  calls[DT_UINT16][DT_INT64] = CalReverseSequence<uint16_t, int64_t>;
  calls[DT_UINT32][DT_INT64] = CalReverseSequence<uint32_t, int64_t>;
  calls[DT_UINT64][DT_INT64] = CalReverseSequence<uint64_t, int64_t>;
  calls[DT_BOOL][DT_INT64] = CalReverseSequence<bool, int64_t>;

  return calls[x_dtype_][seq_lengths_dtype_](seq_dim_, batch_dim_, ioAddrs_,
                                             x_shape_, seq_lengths_shape_, ctx);
}

REGISTER_CPU_KERNEL(kReverseSequence, ReverseSequenceMsCpuKernel);
}  // namespace aicpu
