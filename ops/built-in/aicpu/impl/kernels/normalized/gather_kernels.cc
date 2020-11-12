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

#include "gather_kernels.h"
#include <map>
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/sparse_tensor.h"

namespace {
const char *GATHER = "GatherD";
}

namespace aicpu {
template <typename I, typename T>
void CopyTask(size_t cur, std::vector<int64_t> &pos, T *input, I *index,
              int &dim, T *output, std::vector<int64_t> &output_shape,
              std::vector<int64_t> &out_cargo_size,
              std::vector<int64_t> &input_cargo_size) {
  for (int64_t i = 0; i < output_shape[cur]; ++i) {
    pos[cur] = i;
    if (cur == output_shape.size() - 1) {
      int64_t input_offset = 0;
      int64_t out_offset = 0;
      // out offset
      for (size_t j = 0; j < output_shape.size(); ++j) {
        out_offset += pos[j] * out_cargo_size[j];
      }
      // input offset
      int64_t cur_index = pos[dim];
      pos[dim] = index[out_offset];
      for (size_t j = 0; j < output_shape.size(); ++j) {
        input_offset += pos[j] * input_cargo_size[j];
      }
      // do copy
      output[out_offset] = input[input_offset];
      pos[dim] = cur_index;
    } else {
      // CopyTask
      CopyTask(cur + 1, pos, input, index, dim, output, output_shape,
               out_cargo_size, input_cargo_size);
    }
  }
}

template <typename I, typename T>
uint32_t GatherTask(std::vector<Tensor *> &inputs_,
                    std::vector<Tensor *> &outputs_) {
  if (inputs_.size() == 0 || outputs_.size() == 0) {
    KERNEL_LOG_ERROR("GatherKernel::GatherTask: input or output is empty.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // init
  T *input = (T *)inputs_[0]->GetData();
  int *dim = (int *)inputs_[1]->GetData();
  I *index = (I *)inputs_[2]->GetData();
  T *out = (T *)outputs_[0]->GetData();
  KERNEL_CHECK_NULLPTR(input, KERNEL_STATUS_PARAM_INVALID,
                       "GatherTask input null.")
  KERNEL_CHECK_NULLPTR(dim, KERNEL_STATUS_PARAM_INVALID, "GatherTask dim null.")
  KERNEL_CHECK_NULLPTR(index, KERNEL_STATUS_PARAM_INVALID,
                       "GatherTask index null.")
  KERNEL_CHECK_NULLPTR(out, KERNEL_STATUS_PARAM_INVALID, "GatherTask out null.")
  // out_cargo_size
  std::vector<int64_t> output_shape =
      outputs_[0]->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> out_cargo_size =
      std::vector<int64_t>(output_shape.size(), 1);
  for (int i = out_cargo_size.size() - 2; i >= 0; --i) {
    out_cargo_size[i] = output_shape[i + 1] * out_cargo_size[i + 1];
  }
  // input_cargo_size
  std::vector<int64_t> input_shape =
      inputs_[0]->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> input_cargo_size =
      std::vector<int64_t>(input_shape.size(), 1);
  for (int i = input_cargo_size.size() - 2; i >= 0; --i) {
    input_cargo_size[i] = input_shape[i + 1] * input_cargo_size[i + 1];
  }
  // copy task
  std::vector<int64_t> pos(output_shape.size(), 0);
  int copy_dim = *dim;
  CopyTask<I, T>(0, pos, input, index, copy_dim, out, output_shape,
                 out_cargo_size, input_cargo_size);
  return KERNEL_STATUS_OK;
}

uint32_t GatherKernel::DoCompute() {
  std::map<int, std::map<int, std::function<uint32_t(std::vector<Tensor *> &,
                                                     std::vector<Tensor *> &)>>>
      calls;
  calls[DT_INT32][DT_INT8] = GatherTask<int32_t, int8_t>;
  calls[DT_INT32][DT_INT16] = GatherTask<int32_t, int16_t>;
  calls[DT_INT32][DT_INT32] = GatherTask<int32_t, int32_t>;
  calls[DT_INT32][DT_INT64] = GatherTask<int32_t, int64_t>;
  calls[DT_INT32][DT_FLOAT16] = GatherTask<int32_t, Eigen::half>;
  calls[DT_INT32][DT_FLOAT] = GatherTask<int32_t, float>;
  calls[DT_INT32][DT_DOUBLE] = GatherTask<int32_t, double>;
  calls[DT_INT32][DT_UINT8] = GatherTask<int32_t, uint8_t>;
  calls[DT_INT32][DT_UINT16] = GatherTask<int32_t, uint16_t>;
  calls[DT_INT32][DT_UINT32] = GatherTask<int32_t, uint32_t>;
  calls[DT_INT32][DT_UINT64] = GatherTask<int32_t, uint64_t>;

  calls[DT_INT64][DT_INT8] = GatherTask<int64_t, int8_t>;
  calls[DT_INT64][DT_INT16] = GatherTask<int64_t, int16_t>;
  calls[DT_INT64][DT_INT32] = GatherTask<int64_t, int32_t>;
  calls[DT_INT64][DT_INT64] = GatherTask<int64_t, int64_t>;
  calls[DT_INT64][DT_FLOAT16] = GatherTask<int64_t, Eigen::half>;
  calls[DT_INT64][DT_FLOAT] = GatherTask<int64_t, float>;
  calls[DT_INT64][DT_DOUBLE] = GatherTask<int64_t, double>;
  calls[DT_INT64][DT_UINT8] = GatherTask<int64_t, uint8_t>;
  calls[DT_INT64][DT_UINT16] = GatherTask<int64_t, uint16_t>;
  calls[DT_INT64][DT_UINT32] = GatherTask<int64_t, uint32_t>;
  calls[DT_INT64][DT_UINT64] = GatherTask<int64_t, uint64_t>;

  if (calls.find(index_type_) == calls.end()) {
    KERNEL_LOG_ERROR("GatherKernel op don't support index tensor types: %s",
                     typeid(index_type_).name());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return calls[index_type_][param_type_](inputs_, outputs_);
}

uint32_t GatherKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("GatherKernel::GetInputAndCheck start!");
  // get input Tensors
  const int num_input = 3;
  for (int i = 0; i < num_input; ++i) {
    Tensor *tensor = ctx.Input(i);
    if (tensor == nullptr) {
      KERNEL_LOG_ERROR(
          "GatherKernel::GetInputAndCheck: get input tensor[%d] failed", i);
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
          "GatherKernel::GetInputAndCheck: get output tensor[%d] failed", i);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    outputs_.push_back(tensor);
  }
  // get param type
  param_type_ = static_cast<DataType>(inputs_[0]->GetDataType());
  index_type_ = static_cast<DataType>(inputs_[2]->GetDataType());
  KERNEL_LOG_INFO("GatherKernel::GetInputAndCheck success!");
  return KERNEL_STATUS_OK;
}

uint32_t GatherKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("GatherKernel::Compute start!!");

  uint32_t res = GetInputAndCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    return res;
  }

  res = DoCompute();
  if (res != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("GatherKernel::Compute failed");
    return res;
  }

  KERNEL_LOG_INFO("GatherKernel::Compute success!!");
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(GATHER, GatherKernel);
}  // namespace aicpu
