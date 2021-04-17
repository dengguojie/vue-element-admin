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

#include "sparse_tensor.h"

#include "cpu_types.h"

namespace aicpu {

uint32_t SparseTensor::CreateSparseTensor(Tensor *ix, Tensor *vals,
                                          std::vector<int64_t> shape,
                                          std::vector<int64_t> order) {
  KERNEL_LOG_INFO("Start to execute CreateSparseTensor.");
  if (ix == nullptr || ix->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Ix is nullptr.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (vals == nullptr || vals->GetData() == nullptr) {
    KERNEL_LOG_ERROR("Vals is nullptr.");
    return KERNEL_STATUS_INNER_ERROR;
  }

  if (ix->GetTensorShape()->GetDims() > 2) {
    KERNEL_LOG_ERROR(
        "Index tensor dim size less than 2 or equal to 2, got size [%d].",
        ix->GetTensorShape()->GetDims());
    return KERNEL_STATUS_INNER_ERROR;
  }

  int64_t dims = (ix->GetTensorShape()->GetDims() == 0)
                     ? 1
                     : ix->GetTensorShape()->GetDimSize(0);
  int64_t vals_dim0 = (vals->GetTensorShape()->GetDims() == 0)
                     ? 1
                     : vals->GetTensorShape()->GetDimSize(0);
  if (dims != vals_dim0) {
    KERNEL_LOG_ERROR("Ix dim_size_0 [%lld] != vals dim_size_0 [%lld]", dims,
                     vals_dim0);
    return KERNEL_STATUS_INNER_ERROR;
  }
  dims = ix->GetTensorShape()->GetDims() == 2
             ? ix->GetTensorShape()->GetDimSize(1)
             : 1;
  int64_t orderSize = static_cast<int64_t>(order.size());
  int64_t shapeSize = static_cast<int64_t>(shape.size());
  if (orderSize != dims) {
    KERNEL_LOG_ERROR("orderSize [%lld] != dims [%lld]", orderSize, dims);
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (shapeSize != dims) {
    KERNEL_LOG_ERROR("shapeSize [%lld] != dims [%lld]", shapeSize, dims);
    return KERNEL_STATUS_INNER_ERROR;
  }
  ix_ = std::make_shared<EigenTensor>(ix, ix->GetData());
  vals_ = std::make_shared<EigenTensor>(vals, vals->GetData());
  if (ix_ == nullptr || vals_ == nullptr) {
    KERNEL_LOG_ERROR("Indices or values creat eigen tensor failed.");
    return KERNEL_STATUS_INNER_ERROR;
  }

  shape_.assign(shape.begin(), shape.end());
  order_.assign(order.begin(), order.end());
  dims_ = dims;
  KERNEL_LOG_INFO("execute CreateSparseTensor end.");
  return KERNEL_STATUS_OK;
}

uint32_t SparseTensor::IndicesValid() {
  KERNEL_LOG_INFO("Start to execute IndicesValid.");
  for (auto ord : order_) {
    if (ord < 0) {
      KERNEL_LOG_ERROR("Order was not provided.");
      return KERNEL_STATUS_INNER_ERROR;
    }
  }
  int64_t dim_size = (ix_->GetTensor()->GetTensorShape()->GetDims() == 0)
                         ? 1
                         : ix_->GetTensor()->GetTensorShape()->GetDimSize(0);
  for (int64_t n = 0; n < dim_size; ++n) {
    if (ix_->GetTensor()->GetDataType() == DT_INT32) {
      if (EigenTensorIndicesValid<int32_t>(n) != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("Indices valid failed.");
        return KERNEL_STATUS_PARAM_INVALID;
      }
    } else {
      if (EigenTensorIndicesValid<int64_t>(n) != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("Indices valid failed.");
        return KERNEL_STATUS_PARAM_INVALID;
      }
    }
  }
  KERNEL_LOG_INFO("Execute IndicesValid end.");
  return KERNEL_STATUS_OK;
}

bool SparseTensor::ValidateToDense(Tensor *out) {
  KERNEL_LOG_INFO("Start to execute ValidateToDense.");
  if (out->GetDataType() != vals_->GetTensor()->GetDataType()) {
    KERNEL_LOG_ERROR("Output data type must match vals, got out [%d], vals [%d].",
                     out->GetDataType(), vals_->GetTensor()->GetDataType());
    return false;
  }
  if (out->GetTensorShape()->GetDims() != dims_) {
    KERNEL_LOG_ERROR(
        "Output dims must match idx, got output dims [%d], idx dims [%d].",
        out->GetTensorShape()->GetDims(), dims_);
    return false;
  }
  const auto out_shape = out->GetTensorShape();
  int32_t shapeSize = static_cast<int32_t>(shape_.size());
  if (shapeSize != out_shape->GetDims()) {
    KERNEL_LOG_ERROR(
        "output dims must match shape dims, got output dims [%d], shape dims [%d].",
        out_shape->GetDims(), shapeSize);
    return false;
  }
  for (size_t d = 0; d < shape_.size(); ++d) {
    if (shape_[d] > out_shape->GetDimSize(d)) {
      KERNEL_LOG_ERROR(
          "Valid output shape dims value falied, index [%z], shape value [%d], "
          "greater than output shape value [%d].",
          d, shape_[d], out_shape->GetDimSize(d));
      return false;
    }
  }
  KERNEL_LOG_INFO("Execute ValidateToDense end.");
  return true;
}

GroupIterable SparseTensor::group(const std::vector<int64_t> &group_ix) const {
  if (group_ix.size() > static_cast<size_t>(dims_)) {
    KERNEL_LOG_WARN("Grop_ix.size:%d > dims_:%d", group_ix.size(), dims_);
  }
  return GroupIterable(const_cast<Tensor *>(ix_->GetTensor()),
                       const_cast<Tensor *>(vals_->GetTensor()), dims_,
                       group_ix);
}
}
