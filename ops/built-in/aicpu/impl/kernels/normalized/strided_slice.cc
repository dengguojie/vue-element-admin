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

#include "strided_slice.h"

#include <algorithm>
#include "securec.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "cpu_types.h"

namespace {
constexpr uint32_t kStridedSliceInputNum = 4;
constexpr uint32_t kStridedSliceOutputNum = 1;
constexpr const char *kStridedSlice = "StridedSlice";
}

namespace aicpu {
template <typename T>
static inline void DataLeftShift(T &data) {  data = data << 1;  }

static uint32_t ProcessEllipsisMask(
    const std::vector<int64_t> &begin,
    const std::vector<int64_t> &end,
    const std::vector<int64_t> &strides,
    const std::vector<int64_t> &x_shape,
    int64_t ellipsis_mask, int64_t new_axis_mask,
    size_t &i, size_t &j, int64_t &bit_mask, bool &has_ellipsis,
    int64_t &begin_j, int64_t &end_j, int64_t &strides_j,
    std::vector<int64_t> &begin_res,
    std::vector<int64_t> &end_res,
    std::vector<int64_t> &strides_res) {
  if (ellipsis_mask & bit_mask) {
    if (has_ellipsis) {
      KERNEL_LOG_ERROR("[%s] multiple ellipses in slice spec not allowed.",
                       kStridedSlice);
      return KERNEL_STATUS_INNER_ERROR;
    }

    j++;
    DataLeftShift(bit_mask);
    size_t ellipsis_bits = x_shape.size() - strides.size();
    int64_t bit_mask_tmp = 1;
    for (size_t k = 0; k < strides.size(); ++k) {
      if ((new_axis_mask & bit_mask_tmp) && !(ellipsis_mask & bit_mask_tmp)) {
        ellipsis_bits++;
      }
      DataLeftShift(bit_mask_tmp);
    }
    for (size_t k = 0; k <= ellipsis_bits; ++k) {
      begin_res.push_back(0);
      end_res.push_back(x_shape[i]);
      strides_res.push_back(1);
      i++;
    }
    begin_j = begin[j];
    end_j = end[j];
    strides_j = strides[j];
    has_ellipsis = true;
  }

  return KERNEL_STATUS_OK;
}

inline void ProcessBeginMask(const std::vector<int64_t> &strides,
                             const std::vector<int64_t> &x_shape,
                             int64_t begin_mask, int64_t shrink_axis_mask,
                             size_t i, size_t j, int64_t bit_mask,
                             int64_t &begin_j) {
  if ((begin_mask & bit_mask) && (!(shrink_axis_mask & bit_mask))) {
    begin_j = (strides[j] > 0) ? 0 : -1;
  }
}

inline void ProcessEndMask(const std::vector<int64_t> &strides,
                           const std::vector<int64_t> &x_shape,
                           int64_t end_mask, int64_t shrink_axis_mask,
                           size_t i, size_t j, int64_t bit_mask,
                           int64_t &end_j) {
  if ((end_mask & bit_mask) && !(shrink_axis_mask & bit_mask)) {
    end_j = (strides[j] > 0) ? x_shape[i] : -(x_shape[i] + 1);
  }
}

inline bool ProcessNewAxisMask(int64_t new_axis_mask,
                               size_t &i, int64_t &bit_mask) {
  if (new_axis_mask & bit_mask) {
    i--;
    return true;
  } else {
    return false;
  }
}

inline uint32_t ProcessShrinkAxisMask(const std::vector<int64_t> &x_shape,
                                      int64_t shrink_axis_mask,
                                      size_t i, int64_t bit_mask,
                                      int64_t begin_j, int64_t strides_j,
                                      int64_t &end_j) {
  if (shrink_axis_mask & bit_mask) {
    if ((begin_j < -x_shape[i]) || (begin_j >= x_shape[i]) || (strides_j < 0)) {
      KERNEL_LOG_ERROR("[%s] process shrink axis mask failed.", kStridedSlice);
      return KERNEL_STATUS_INNER_ERROR;
    }
    end_j = begin_j + 1;
  }
  return KERNEL_STATUS_OK;
}

uint32_t ProcessMasks(const std::vector<int64_t> &begin,
                      const std::vector<int64_t> &end,
                      const std::vector<int64_t> &strides,
                      const std::vector<int64_t> &x_shape,
                      int64_t begin_mask, int64_t end_mask,
                      int64_t ellipsis_mask, int64_t new_axis_mask,
                      int64_t shrink_axis_mask,
                      size_t &i, size_t &j,
                      int64_t &bit_mask, bool &has_ellipsis,
                      std::vector<int64_t> &begin_res,
                      std::vector<int64_t> &end_res,
                      std::vector<int64_t> &strides_res) {
  int64_t begin_j = begin[j];
  int64_t end_j = end[j];
  int64_t strides_j = strides[j];
  if (j < strides.size()) {
    if (ProcessEllipsisMask(begin, end, strides, x_shape, ellipsis_mask,
        new_axis_mask, i, j, bit_mask, has_ellipsis,
        begin_j, end_j, strides_j, begin_res, end_res, strides_res) ==
        KERNEL_STATUS_INNER_ERROR) {
      return KERNEL_STATUS_INNER_ERROR;
    }
    ProcessBeginMask(strides, x_shape, begin_mask, shrink_axis_mask,
                     i, j, bit_mask, begin_j);
    ProcessEndMask(strides, x_shape, end_mask, shrink_axis_mask,
                   i, j, bit_mask, end_j);
    if (ProcessNewAxisMask(new_axis_mask, i, bit_mask)) {
      return KERNEL_STATUS_OK;
    }
    if (ProcessShrinkAxisMask(x_shape, shrink_axis_mask, i, bit_mask,
        begin_j, strides_j, end_j) == KERNEL_STATUS_INNER_ERROR) {
      return KERNEL_STATUS_INNER_ERROR;
    }
  } else {
    begin_j = 0;
    end_j = x_shape[i];
    strides_j = 1;
  }

  begin_res.push_back(begin_j);
  end_res.push_back(end_j);
  strides_res.push_back(strides_j);
  return KERNEL_STATUS_OK;
}

uint32_t StridedSliceCpuKernel::InitParamsWithMasks(
    const std::vector<int64_t> &x_shape,
    int64_t begin_mask, int64_t end_mask,
    int64_t ellipsis_mask, int64_t new_axis_mask,
    int64_t shrink_axis_mask,
    std::vector<int64_t> &begin,
    std::vector<int64_t> &end,
    std::vector<int64_t> &strides) {
  size_t i = 0;
  size_t j = 0;
  int64_t bit_mask = 1;
  bool has_ellipsis = false;
  std::vector<int64_t> begin_res;
  std::vector<int64_t> end_res;
  std::vector<int64_t> strides_res;
  while (i < x_shape.size()) {
    KERNEL_HANDLE_ERROR(ProcessMasks(begin, end, strides, x_shape,
        begin_mask, end_mask, ellipsis_mask, new_axis_mask,
        shrink_axis_mask, i, j, bit_mask, has_ellipsis,
        begin_res, end_res, strides_res),
        "[%s] process masks failed.", kStridedSlice);
    i++;
    j++;
    DataLeftShift(bit_mask);
  }

  auto begin_iter = begin_res.begin();
  auto end_iter = end_res.begin();
  for (auto strides_iter = strides_res.begin();
      strides_iter != strides_res.end(); ) {
    if (*strides_iter == 0) {
      begin_iter = begin_res.erase(begin_iter);
      end_iter = end_res.erase(end_iter);
      strides_iter = strides_res.erase(strides_iter);
    } else {
      begin_iter++;
      end_iter++;
      strides_iter++;
    }
  }

  if (begin_res.empty() || end_res.empty() || strides_res.empty()) {
    KERNEL_LOG_ERROR("[%s] init params with masks failed.", kStridedSlice);
    return KERNEL_STATUS_INNER_ERROR;
  }

  begin = begin_res;
  end = end_res;
  strides = strides_res;
  KERNEL_LOG_INFO("[%s] begin with masks: [%s].", kStridedSlice,
                  VectorToString(begin).c_str());
  KERNEL_LOG_INFO("[%s] end with masks: [%s].", kStridedSlice,
                  VectorToString(end).c_str());
  KERNEL_LOG_INFO("[%s] strides with masks: [%s].", kStridedSlice,
                  VectorToString(strides).c_str());
  return KERNEL_STATUS_OK;
}

uint32_t StridedSliceCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kStridedSliceInputNum,
                                  kStridedSliceOutputNum),
                      "[%s] check params failed.", kStridedSlice);

  // parse params
  KERNEL_HANDLE_ERROR(ParseKernelParams(ctx),
                      "[%s] parse kernel params failed.", kStridedSlice);

  // init params with masks
  KERNEL_HANDLE_ERROR(InitParamsWithMasks(x_shape_, begin_mask_, end_mask_,
      ellipsis_mask_, new_axis_mask_, shrink_axis_mask_,
      begin_, end_, strides_),
      "[%s] init params with masks failed.", kStridedSlice);

  // cal strided slice
  Tensor *x_tensor = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(x_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get input[0] failed.", kStridedSlice);
  Tensor *y_tensor = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(y_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get output[0] failed.", kStridedSlice);
  DataType data_type = x_tensor->GetDataType();

#define STRIDED_SLICE_CASE(dtype, T)                                        \
  case dtype:                                                               \
    return CalStridedSlice<T>(ctx, begin_, end_, strides_,                  \
                              x_tensor, y_tensor);

  switch (data_type) {
    STRIDED_SLICE_CASE(DT_INT8, int8_t)
    STRIDED_SLICE_CASE(DT_INT16, int16_t)
    STRIDED_SLICE_CASE(DT_INT32, int32_t)
    STRIDED_SLICE_CASE(DT_INT64, int64_t)
    STRIDED_SLICE_CASE(DT_UINT8, uint8_t)
    STRIDED_SLICE_CASE(DT_UINT16, uint16_t)
    STRIDED_SLICE_CASE(DT_UINT32, uint32_t)
    STRIDED_SLICE_CASE(DT_UINT64, uint64_t)
    STRIDED_SLICE_CASE(DT_FLOAT16, Eigen::half)
    STRIDED_SLICE_CASE(DT_FLOAT, float)
    STRIDED_SLICE_CASE(DT_DOUBLE, double)
    STRIDED_SLICE_CASE(DT_BOOL, bool)
    default:
      KERNEL_LOG_ERROR("[%s] doesn't support input[0] data_type [%s].",
                       kStridedSlice, DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

#undef STRIDED_SLICE_CASE
}

uint32_t StridedSliceCpuKernel::ParseKernelParams(CpuKernelContext &ctx) {
  // get inputs
  Tensor *x_tensor = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(x_tensor, KERNEL_STATUS_PARAM_INVALID,
                       "[%s] get input[0] failed.", kStridedSlice);
  x_shape_ = x_tensor->GetTensorShape()->GetDimSizes();
  KERNEL_LOG_INFO("[%s] get input[0] shape: [%s].",
                  kStridedSlice, VectorToString(x_shape_).c_str());

  KERNEL_HANDLE_ERROR(ParseIndexInput(ctx, 1, begin_),
                      "[%s] parse index input failed.", kStridedSlice);
  KERNEL_HANDLE_ERROR(ParseIndexInput(ctx, 2, end_),
                      "[%s] parse index input failed.", kStridedSlice);
  KERNEL_HANDLE_ERROR(ParseIndexInput(ctx, 3, strides_),
                      "[%s] parse index input failed.", kStridedSlice);

  // get masks
  KERNEL_HANDLE_ERROR(GetMaskAttr(ctx, "begin_mask", begin_mask_),
                      "[%s] get mask attr failed.", kStridedSlice);
  KERNEL_HANDLE_ERROR(GetMaskAttr(ctx, "end_mask", end_mask_),
                      "[%s] get mask attr failed.", kStridedSlice);
  KERNEL_HANDLE_ERROR(GetMaskAttr(ctx, "ellipsis_mask", ellipsis_mask_),
                      "[%s] get mask attr failed.", kStridedSlice);
  KERNEL_HANDLE_ERROR(GetMaskAttr(ctx, "new_axis_mask", new_axis_mask_),
                      "[%s] get mask attr failed.", kStridedSlice);
  KERNEL_HANDLE_ERROR(GetMaskAttr(ctx, "shrink_axis_mask", shrink_axis_mask_),
                      "[%s] get mask attr failed.", kStridedSlice);

  return KERNEL_STATUS_OK;
}

uint32_t StridedSliceCpuKernel::ParseIndexInput(CpuKernelContext &ctx,
                                                uint32_t index,
                                                std::vector<int64_t> &vec) {
  Tensor *index_tensor = ctx.Input(index);
  KERNEL_CHECK_NULLPTR(index_tensor, KERNEL_STATUS_INNER_ERROR,
                       "[%s] get input[%u] failed.", kStridedSlice, index);
  int64_t tensor_size = index_tensor->NumElements();
  switch (index_tensor->GetDataType()) {
    case DT_INT32: {
      int32_t *tensor_data = static_cast<int32_t *>(index_tensor->GetData());
      vec.insert(vec.begin(), tensor_data, tensor_data + tensor_size);
      break;
    }
    case DT_INT64: {
      int64_t *tensor_data = static_cast<int64_t *>(index_tensor->GetData());
      vec.insert(vec.begin(), tensor_data, tensor_data + tensor_size);
      break;
    }
    default:
      KERNEL_LOG_ERROR("[%s] input[%u] data_tpye must be in {int32 int64}.",
                       kStridedSlice, index);
      return KERNEL_STATUS_PARAM_INVALID;
  }

  KERNEL_LOG_INFO("[%s] get input[%u]: [%s].", kStridedSlice, index,
                  VectorToString(vec).c_str());
  return KERNEL_STATUS_OK;
}

uint32_t StridedSliceCpuKernel::GetMaskAttr(CpuKernelContext &ctx,
                                            std::string attr,
                                            int64_t &mask) {
  AttrValue *mask_attr = ctx.GetAttr(attr);
  if (mask_attr != nullptr) {
    mask = mask_attr->GetInt();
  } else {
    KERNEL_LOG_WARN("[%s] can not get attr [%s].", kStridedSlice, attr.c_str());
    mask = 0;
  }
  KERNEL_LOG_INFO("[%s] get attr [%s]: [%d].",
                  kStridedSlice, attr.c_str(), mask);
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kStridedSlice, StridedSliceCpuKernel);
}  // namespace aicpu
