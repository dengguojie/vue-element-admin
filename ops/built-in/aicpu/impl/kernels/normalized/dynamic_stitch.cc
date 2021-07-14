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
#include "dynamic_stitch.h"
#include <map>
#include <memory.h>
#include <utility>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "securec.h"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"
#include "utils/eigen_tensor.h"
#include "utils/sparse_tensor.h"

namespace {
const char *kDynamicStitch = "DynamicStitch";
const size_t kExpandMax = 10;
}  // namespace

namespace aicpu {
uint32_t DynamicStitchKernel::GetInputAndCheck(CpuKernelContext &ctx,
                                               int *first_dim_size,
                                               int *data_elements_size) {
  n_ = ctx.GetInputsSize() / 2;

  int32_t max_index = -1;
  if (data_elements_size) {
    *data_elements_size = 0;
  }
  for (int i = 0; i < n_; ++i) {
    Tensor *indices = ctx.Input(i);
    if (indices->NumElements() > 0) {
      EigenTensor indicesET(indices, indices->GetData());
      Eigen::Tensor<int32_t, 0, Eigen::RowMajor> m =
          indicesET.flat<int32_t>().maximum();
      max_index = std::max(m(), max_index);
    }
    if (data_elements_size) {
      *data_elements_size += indices->NumElements();
    }
  }

  *first_dim_size = max_index + 1;

  // Validate that data[i].shape = indices[i].shape + constant
  Tensor *data0 = ctx.Input(n_);
  Tensor *indices0 = ctx.Input(0);
  input_dtype_ = static_cast<DataType>(data0->GetDataType());
  for (int input_num = 0; input_num < n_; input_num++) {
    Tensor *indices = ctx.Input(input_num);
    Tensor *data = ctx.Input(n_ + input_num);
    KERNEL_CHECK_FALSE(
        StartsWith(data->GetTensorShape(), indices->GetTensorShape()),
        KERNEL_STATUS_PARAM_INVALID,
        "The input data[%d].shape does not start with indices[%d].shape",
        input_num, input_num);
    KERNEL_CHECK_FALSE(
        input_num == 0 || SameExtraShape(data0, indices0, data, indices),
        KERNEL_STATUS_PARAM_INVALID,
        "Need data[0].shape[%d:] = data[%d].shape[%d:]",
        indices0->GetTensorShape()->GetDims(), input_num,
        indices->GetTensorShape()->GetDims());
  }

  // infer output shape: [*first_dim_size] + data.shape[indices.dims:]
  std::vector<int64_t> result_shape{*first_dim_size};
  for (int d = indices0->GetTensorShape()->GetDims();
       d < data0->GetTensorShape()->GetDims(); d++) {
    result_shape.push_back(data0->GetTensorShape()->GetDimSize(d));
  }
  // update output shape
  Tensor *y = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(y, KERNEL_STATUS_PARAM_INVALID,
                       "Get output:[0] failed.");
  auto y_shape = y->GetTensorShape();
  KERNEL_CHECK_NULLPTR(y_shape, KERNEL_STATUS_PARAM_INVALID,
                       "Get output:[0] shape failed.");
  y_shape->SetDimSizes(result_shape);
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t CalDynamicStitch(CpuKernelContext &ctx, int first_dim_size,
                          int data_elements_size, int n) {
  Tensor *merged = ctx.Output(0);
  if (first_dim_size > 0) {
    // compute flat_outer_dims shape
    std::vector<int64_t> orig = merged->GetTensorShape()->GetDimSizes();
    int64_t num_out_dims = 2;
    std::vector<int64_t> out_dims(num_out_dims, 0);
    for (int64_t out_dim = 0; out_dim <= num_out_dims - 1; ++out_dim) {
      out_dims[out_dim] =
          out_dim >= static_cast<int64_t>(orig.size()) ? 1 : orig[out_dim];
    }
    for (int64_t in_dim = num_out_dims;
         in_dim < static_cast<int64_t>(orig.size()); ++in_dim) {
      out_dims[num_out_dims - 1] *= orig[in_dim];
    }
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> merged_flat(
        (T*)merged->GetData(), out_dims[0], out_dims[1]);
    const auto slice_size = merged_flat.dimension(1);
    const size_t slice_bytes = slice_size * sizeof(T);
    for (int input_num = 0; input_num < n; input_num++) {
      Tensor *indices_ms = ctx.Input(input_num);
      EigenTensor indices(indices_ms, indices_ms->GetData());

      auto indices_vec = indices.flat<int32_t>();
      Tensor *data_ms = ctx.Input(input_num + n);
      Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> data_flat(
          (T*)data_ms->GetData(), indices_vec.dimension(0), slice_size);
      T *merged_base = merged_flat.data();
      const T *data_base = data_flat.data();
      uint64_t merged_size = merged->GetDataSize();
      for (size_t i = 0; i < static_cast<size_t>(indices_vec.size()); i++) {
        int index = SubtleMustCopy(indices_vec(i));
        KERNEL_CHECK_FALSE(index <= static_cast<int>(data_elements_size * kExpandMax),
                           KERNEL_STATUS_PARAM_INVALID,
                           "The value of indices[%zu]: [%d] is too big that"
                           "greater than [%zu] * total size of input[0]:[%d]",
                           i, index, kExpandMax, data_elements_size);
        KERNEL_CHECK_FALSE(index < first_dim_size, KERNEL_STATUS_PARAM_INVALID,
                           "The value of indices[%zu]:[%d] is out of range:[%d]",
                           i, index, first_dim_size);
        auto ret = memcpy_s(merged_base + index * slice_size, merged_size,
                            data_base + i * slice_size, slice_bytes);
        merged_size -= slice_bytes;
        KERNEL_CHECK_FALSE((ret == EOK), KERNEL_STATUS_INNER_ERROR,
                           "Memcpy to output[0] failed, return = [%d]", ret);
      }
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t DynamicStitchKernel::Compute(CpuKernelContext &ctx) {
  int first_dim_size = 0;
  int data_elements_size = 0;
  uint32_t res = GetInputAndCheck(ctx, &first_dim_size, &data_elements_size);
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res,
                     "GetInputAndCheck failed.");

  std::map<int, std::function<uint32_t(CpuKernelContext &, int, int, int)>>
      calls;
  calls[DT_FLOAT16] = CalDynamicStitch<Eigen::half>;
  calls[DT_FLOAT] = CalDynamicStitch<float>;
  calls[DT_DOUBLE] = CalDynamicStitch<double>;
  calls[DT_INT8] = CalDynamicStitch<int8_t>;
  calls[DT_INT16] = CalDynamicStitch<int16_t>;
  calls[DT_INT32] = CalDynamicStitch<int32_t>;
  calls[DT_INT64] = CalDynamicStitch<int64_t>;
  calls[DT_UINT8] = CalDynamicStitch<uint8_t>;
  calls[DT_UINT16] = CalDynamicStitch<uint16_t>;
  calls[DT_UINT32] = CalDynamicStitch<uint32_t>;
  calls[DT_UINT64] = CalDynamicStitch<uint64_t>;
  calls[DT_BOOL] = CalDynamicStitch<bool>;

  auto iter = calls.find(input_dtype_);
  if (iter == calls.end()) {
    KERNEL_LOG_ERROR(
        "DynamicStitch op doesn't support index tensor types: [%s]",
        DTypeStr(input_dtype_).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return iter->second(ctx, first_dim_size, data_elements_size, n_);
}

REGISTER_CPU_KERNEL(kDynamicStitch, DynamicStitchKernel);
}  // namespace aicpu
