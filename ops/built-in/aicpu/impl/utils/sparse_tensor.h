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

#ifndef AICPU_SPARSETENSOR_H
#define AICPU_SPARSETENSOR_H

#include <algorithm>
#include <memory>

#include "cpu_tensor.h"
#include "eigen_tensor.h"
#include "log.h"
#include "sparse_group.h"
#include "status.h"

namespace aicpu {
template <typename T>
const T SubtleMustCopy(const T &x) {
  auto *to_x = reinterpret_cast<const volatile T *>(&x);
  return *to_x;
}
}

namespace aicpu {
class SparseTensor {
 public:
  SparseTensor() : dims_(0) {}
  ~SparseTensor() = default;

  /*
   * creat sparse tensor
   * @param ix: index tensor
   * @param tensorvals: tensorvals tensor
   * @param shape: shape vec
   * @param order: order vec
   * @return uint32_t: 0->success other->failed
   */
  uint32_t CreateSparseTensor(Tensor *ix, Tensor *tensorvals,
                              std::vector<int64_t> shape,
                              std::vector<int64_t> order);

  /*
   * sparse indices valid
   * @return uint32_t: 0->success other->failed
   */
  uint32_t IndicesValid() const;

  /*
   * group sparse tensor
   * @return GroupIterable
   */
  GroupIterable group(const std::vector<int64_t> &group_ix) const;

  /*
   * sparse eigen tensor indices valid
   * @return uint32_t: 0->success other->failed
   */
  template <typename T>
  uint32_t EigenTensorIndicesValid(int64_t n) const {
    KERNEL_LOG_INFO("Start to execute eigen IndicesValid.");
    bool valid = true;
    bool different = false;
    bool increasing = true;

    const auto ix_t = ix_->matrix<T>();
    if (n == 0) {
      for (int di = 0; di < dims_; ++di) {
        if (ix_t(n, di) < 0 || ix_t(n, di) >= shape_[di]) {
          valid = false;
        }
      }
      different = true;
    } else {
      for (int di = 0; di < dims_; ++di) {
        if (ix_t(n, di) < 0 || ix_t(n, di) >= shape_[di]) {
          valid = false;
        }
        int64_t diff = ix_t(n, order_[di]) - ix_t(n - 1, order_[di]);
        if (diff > 0) {
          different = true;
        }
        if (!different && diff < 0) {
          increasing = false;
        }
      }
    }

    if (!valid) {
      KERNEL_LOG_ERROR("Indices is out of bounds, index=%lld.", n);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    if (!increasing) {
      KERNEL_LOG_ERROR("indices is out of order, index=%lld.", n);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    if (!different) {
      KERNEL_LOG_ERROR("indices is repeated, index=%lld.", n);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    KERNEL_LOG_INFO("Execute eigen IndicesValid end.");
    return KERNEL_STATUS_OK;
  }

  /*
   * validate sparse to dense
   * @param output: output tensor
   * @return bool: true->success false->failed
   */
  bool ValidateToDense(const Tensor *out) const;

  /*
   * sparse tensor to dense tensor
   * @param output: output tensor
   * @return uint32_t: 0->success other->failed
   */
  template <typename IndiceT, typename ValueT>
  uint32_t ToDense(Tensor *output) {
    KERNEL_LOG_INFO("Start to execute ToDense.");
    if (output == nullptr || output->GetData() == nullptr) {
      KERNEL_LOG_ERROR("Output tensor is nullptr.");
      return KERNEL_STATUS_INNER_ERROR;
    }
    EigenTensor outputET(output, output->GetData());
    if (!ValidateToDense(output)) {
      KERNEL_LOG_ERROR("Validate to dense param failed.");
      return KERNEL_STATUS_INNER_ERROR;
    }
    auto output_t = outputET.flat<ValueT>();
    auto ix_t = ix_->matrix<IndiceT>();
    auto vals_t = vals_->vec<ValueT>();
    std::vector<int64_t> strides(dims_);
    const auto &out_shape = output->GetTensorShape();
    if (dims_ > 0) {
      strides[dims_ - 1] = 1;
    }
    for (int32_t d = dims_ - 2; d >= 0; --d) {
      strides[d] = strides[d + 1] * out_shape->GetDimSize(d + 1);
    }
    for (int n = 0; n < vals_t.dimension(0); ++n) {
      bool invalid_dims = false;
      int64_t ix = 0;
      for (int d = 0; d < dims_; ++d) {
        const int64_t ix_n_d = ix_t(n, d);
        if (ix_n_d > out_shape->GetDimSize(d)) {
          invalid_dims = true;
        }
        ix += strides[d] * ix_n_d;
      }
      if (invalid_dims) {
        KERNEL_LOG_ERROR("Sparse to dense got invalid dims.");
        return KERNEL_STATUS_INNER_ERROR;
      }
      output_t(ix) = vals_t(n);
    }
    KERNEL_LOG_INFO("Execute ToDense end.");
    return KERNEL_STATUS_OK;
  }

 private:
  std::shared_ptr<EigenTensor> ix_;
  std::shared_ptr<EigenTensor> vals_;
  std::vector<int64_t> shape_;
  std::vector<int64_t> order_;
  int32_t dims_;
};
}  // namespace aicpu

#endif  // AICPU_SPARSETENSOR_H
