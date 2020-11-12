/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of unique cust
 */

#include "unique_cust_kernels.h"

#include "cpu_tensor.h"
#include "cpu_tensor_shape.h"
#include "cpu_types.h"

namespace {
const char *UNIQUE_CUST = "UniqueCust";

template <typename T>
uint32_t UniqueTask(aicpu::Tensor *x, aicpu::Tensor *y, aicpu::Tensor *idx,
                    int64_t N) {
  T *a = reinterpret_cast<T *>(x->GetData());
  if (a == nullptr) {
    return 1;
  }

  T *out = reinterpret_cast<T *>(y->GetData());
  if (out == nullptr) {
    return 1;
  }

  T *idx_vec = reinterpret_cast<T *>(idx->GetData());
  if (idx_vec == nullptr) {
    return 1;
  }

  std::unordered_map<T, int> uniq;
  uniq.reserve(2 * N);
  for (int i = 0, j = 0; i < N; ++i) {
    auto it = uniq.emplace(a[i], j);
    idx_vec[i] = it.first->second;
    if (it.second) {
      ++j;
    }
  }
  for (const auto &it : uniq) {
    out[it.second] = it.first;
  }

  // update outputshape
  auto y_shape = y->GetTensorShape();
  if (y_shape == nullptr) {
    return 1;
  }

  if (y_shape->GetUnknownRank()) {
    std::vector<int64_t> y_shape_values = y_shape->GetDimSizes();
    if (y_shape_values.size() == 0) {
      y_shape_values.push_back(uniq.size());
    } else {
      y_shape_values[0] = uniq.size();
    }

    y_shape->SetDimSizes(y_shape_values);
  }
  return 0;
}
}

namespace aicpu {
uint32_t UniqueCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *param_tensor = ctx.Input(0);
  if (param_tensor == nullptr) {
    return 1;
  }
  auto param_shape = param_tensor->GetTensorShape();
  if (param_shape == nullptr) {
    return 1;
  }

  DataType param_type = param_tensor->GetDataType();
  int64_t p_size = 1;
  for (int i = 0; i < param_shape->GetDims(); ++i) {
    p_size *= param_shape->GetDimSize(i);
  }

  std::map<int, std::function<uint32_t(aicpu::Tensor *, aicpu::Tensor *,
                                       aicpu::Tensor *, int64_t)> >
      calls;
  calls[DataType::DT_INT32] = UniqueTask<int32_t>;
  calls[DataType::DT_INT64] = UniqueTask<int64_t>;

  return calls[param_type](param_tensor, ctx.Output(0), ctx.Output(1), p_size);
}

REGISTER_CPU_KERNEL(UNIQUE_CUST, UniqueCpuKernel);
}  // namespace aicpu
