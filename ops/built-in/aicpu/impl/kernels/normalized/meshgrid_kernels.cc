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

#include "meshgrid_kernels.h"

#include <securec.h>
#include <string>

#include "Eigen/Core"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"

using std::string;

namespace {
const char *Meshgrid = "Meshgrid";
}

namespace aicpu {

template <typename T>
uint32_t MeshgridTask(std::vector<void *> &ioAddrs, std::string &indexing,
                      size_t &ndim, std::vector<int> &bcast) {
  KERNEL_LOG_INFO("MeshgridCpuKernel::Compute start!! ");
  for (int i = 0; i < static_cast<int>(ndim); ++i) {  // 0~ndim
    auto new_i = i;
    auto s = bcast;
    if (indexing == "xy" && i < 2) {
      new_i = 1 - i;
      auto tmp = s[0];
      s[0] = s[1];
      s[1] = tmp;
    }
    size_t row = 1;
    size_t col = 1;
    for (int j = 0; j <= new_i; j++) {
      row *= s[j];
    }
    for (int j = new_i + 1; j < static_cast<int>(s.size()); j++) {
      col *= s[j];
    }

    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> input_map(
        (T *)ioAddrs[i], bcast[i], 1);
    const auto &input = Eigen::Tensor<T, 2, Eigen::RowMajor>(input_map);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> output(
        (T *)ioAddrs[ndim + i], row, col);

    Eigen::Tensor<T, 2, Eigen::RowMajor> origin(bcast[i], row * col / bcast[i]);
    for (int c = 0; c < bcast[i]; ++c) {
      for (int r = 0; r < static_cast<int>(row * col / bcast[i]); ++r) {
        origin(c, r) = input(c, 0);
      }
    }

    for (size_t j = 0; j < row * col / bcast[i] / col; ++j) {
      Eigen::array<size_t, 2> offsets_in = {0, col * j};
      Eigen::array<size_t, 2> offsets_out = {bcast[i] * j, 0};
      Eigen::array<size_t, 2> extents = {static_cast<size_t>(bcast[i]), col};
      output.slice(offsets_out, extents) = origin.slice(offsets_in, extents);
    }
  }

  KERNEL_LOG_INFO("MeshgridCpuKernel::Compute end!! ");
  return KERNEL_STATUS_OK;
}

uint32_t MeshgridCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t res = GetInputAndCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    return res;
  }

  std::map<int, std::function<uint32_t(std::vector<void *> &, std::string &,
                                       size_t &, std::vector<int> &)>>
      calls;
  calls[DT_INT8] = MeshgridTask<int8_t>;
  calls[DT_INT16] = MeshgridTask<int16_t>;
  calls[DT_INT32] = MeshgridTask<int32_t>;
  calls[DT_INT64] = MeshgridTask<int64_t>;
  calls[DT_FLOAT16] = MeshgridTask<Eigen::half>;
  calls[DT_FLOAT] = MeshgridTask<float>;
  calls[DT_DOUBLE] = MeshgridTask<double>;
  calls[DT_UINT8] = MeshgridTask<uint8_t>;
  calls[DT_UINT16] = MeshgridTask<uint16_t>;
  calls[DT_UINT32] = MeshgridTask<uint32_t>;
  calls[DT_UINT64] = MeshgridTask<uint64_t>;
  calls[DT_BOOL] = MeshgridTask<bool>;
  if (calls.find(input_type_) == calls.end()) {
    KERNEL_LOG_ERROR(
        "MeshgridCpuKernel op don't support input tensor types: %s",
        typeid(input_type_).name());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return calls[input_type_](ioAddrs_, indexing_, ndim_, bcast_);
}

uint32_t MeshgridCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("MeshgridCpuKernel::GetInputAndCheck start!! ");

  AttrValue *indexing = ctx.GetAttr("indexing");
  KERNEL_CHECK_NULLPTR(indexing, KERNEL_STATUS_PARAM_INVALID,
                       "get attr:indexing failed.");
  indexing_ = indexing->GetString();

  // get input_tensor
  Tensor *input_tensor_ = ctx.Input(0);
  if (input_tensor_ == nullptr) {
    KERNEL_LOG_ERROR("UniqueWithPadKernel::get input:0 failed");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  input_type_ = static_cast<DataType>(input_tensor_->GetDataType());

  ndim_ = ctx.GetInputsSize();
  bcast_.resize(ndim_);
  ioAddrs_.resize(2 * ndim_);
  if (ctx.GetInputsSize() != ctx.GetOutputsSize()) {
    KERNEL_LOG_ERROR("The number of input and output should be the same.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (int n = 0; n < static_cast<int>(ctx.GetInputsSize()); ++n) {
    Tensor *input_tensor = ctx.Input(n);
    Tensor *output_tensor = ctx.Output(n);
    std::shared_ptr<TensorShape> input_shape = input_tensor->GetTensorShape();
    if (input_shape->GetDims() != 1) {
      KERNEL_LOG_ERROR("input tensor should be 1-D.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    bcast_[n] = input_shape->GetDimSize(0);

    ioAddrs_[n] = reinterpret_cast<void *>(input_tensor->GetData());
    ioAddrs_[n + ndim_] = reinterpret_cast<void *>(output_tensor->GetData());
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(Meshgrid, MeshgridCpuKernel);
}  // namespace aicpu
