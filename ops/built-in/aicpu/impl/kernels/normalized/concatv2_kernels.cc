/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of concatv2
 */

#include "concatv2_kernels.h"

using namespace std;

namespace {
const char *ConcatV2 = "ConcatV2";
}

namespace aicpu {
uint32_t ConcatV2Kernel::CheckAndInitParams(CpuKernelContext &ctx) {
  AttrValue *n_ptr = ctx.GetAttr("N");
  KERNEL_CHECK_NULLPTR(n_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "Get attr[N] failed.");
  n_ = n_ptr->GetInt();
  // "x" is a list of at least 2 "tensor" objects of the same type
  KERNEL_CHECK_FALSE((n_ >= 2), KERNEL_STATUS_PARAM_INVALID,
                     "Attr[N] must >= 2, but got attr[N]=%lld", n_);

  uint32_t input_num = ctx.GetInputsSize();

  // input_num is n_(concat tensor num) + 1(concat_dim)
  KERNEL_CHECK_FALSE((static_cast<int64_t>(input_num) - 1 == n_),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Input num must equal attr[N]:%lld + 1,"
                     "but got input num:%u",
                     n_, input_num);

  Tensor *concat_dim_ptr = ctx.Input(n_);
  KERNEL_CHECK_NULLPTR(concat_dim_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "Get input[concat_dim] failed.");
  auto concat_dim_shape_ptr = concat_dim_ptr->GetTensorShape();
  KERNEL_CHECK_NULLPTR(concat_dim_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "Get input[concat_dim] shape failed.");
  int32_t concat_dim_dims = concat_dim_shape_ptr->GetDims();
  KERNEL_CHECK_FALSE(
      (concat_dim_dims == 0), KERNEL_STATUS_PARAM_INVALID,
      "Input[concat_dim] should be a scalar integer, but got rank=%d.",
      concat_dim_dims);
  int64_t concat_dim = 0;
  DataType concat_dim_data_type = concat_dim_ptr->GetDataType();
  KERNEL_CHECK_FALSE(
      (concat_dim_data_type == DT_INT32 || concat_dim_data_type == DT_INT64),
      KERNEL_STATUS_PARAM_INVALID,
      "Input[concat_dim] data type must DT_INT32 or DT_INT64,"
      "but got data type=%d.",
      concat_dim_data_type);
  auto concat_dim_data_ptr = concat_dim_ptr->GetData();
  KERNEL_CHECK_NULLPTR(concat_dim_data_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "Get input[concat_dim] data failed.");
  if (concat_dim_data_type == DT_INT32) {
    concat_dim =
        static_cast<int64_t>(*reinterpret_cast<int32_t *>(concat_dim_data_ptr));
  } else {
    concat_dim = *reinterpret_cast<int64_t *>(concat_dim_data_ptr);
  }

  Tensor *input0_ptr = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(input0_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "Get input[x0] failed.");
  auto input0_shape_ptr = input0_ptr->GetTensorShape();
  KERNEL_CHECK_NULLPTR(input0_shape_ptr, KERNEL_STATUS_PARAM_INVALID,
                       "Get input[x0] shape failed.");
  input_dims_ = input0_shape_ptr->GetDims();
  data_type_ = input0_ptr->GetDataType();
  KERNEL_LOG_INFO("data type:%d", data_type_);
  axis_ = concat_dim < 0 ? concat_dim + input_dims_ : concat_dim;
  KERNEL_CHECK_FALSE((0 <= axis_ && axis_ < input_dims_),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Input[concat_dim] need in the "
                     "range[%d, %d), but got %lld.",
                     -input_dims_, input_dims_, concat_dim);
  inputs_flat_dim0_ = 1;
  for (uint32_t d = 0; d < axis_; ++d) {
    inputs_flat_dim0_ *= input0_shape_ptr->GetDimSize(d);
  }
  return KERNEL_STATUS_OK;
}

uint32_t ConcatV2Kernel::Compute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("ConcatV2Kernel start.");
  KERNEL_CHECK_FALSE((CheckAndInitParams(ctx) == KERNEL_STATUS_OK),
                     KERNEL_STATUS_PARAM_INVALID, "CheckAndInitParams failed.");
  switch (data_type_) {
    case DT_FLOAT16:
      return DoCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return DoCompute<float>(ctx);
    case DT_INT8:
      return DoCompute<int8_t>(ctx);
    case DT_INT16:
      return DoCompute<int16_t>(ctx);
    case DT_INT32:
      return DoCompute<int32_t>(ctx);
    case DT_INT64:
      return DoCompute<int64_t>(ctx);
    case DT_UINT8:
      return DoCompute<uint8_t>(ctx);
    case DT_UINT16:
      return DoCompute<uint16_t>(ctx);
    case DT_UINT32:
      return DoCompute<uint32_t>(ctx);
    case DT_UINT64:
      return DoCompute<uint64_t>(ctx);
    default:
      KERNEL_LOG_ERROR("unsupport datatype:%d", data_type_);
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

REGISTER_CPU_KERNEL(ConcatV2, ConcatV2Kernel);
}  // namespace aicpu