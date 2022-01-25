/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description:This file provides the function of expandding.
 * Author: Huawei.
 * Create:2021-10-08.
 */
#include "expand.h"
using namespace std;

namespace {
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 1;
}  // namespace

namespace expand {
template <typename T, typename IndexT>
uint32_t CopyExpandIndex(std::vector<T> x_indexes,
                         std::vector<T>& out_indexes,
                         std::vector<IndexT> shape_in,
                         std::vector<IndexT>& shape_out) {
  out_indexes.clear();
  uint64_t expand_to_axis =
      1;  //  Default target dimension when beginning to expand tensor
  uint64_t copy_size = 1;       //  Default size when beginning to copy tensor
  uint64_t break_axis_num = 0;  //  Default axis when beginning to expand tensor
  for (int64_t i = static_cast<int64_t>(shape_in.size() - 1); i >= 0; i--) {
    if (shape_in[i] != shape_out[i]) {
      if (shape_in[i] != static_cast<IndexT>(1)) {
        KERNEL_LOG_ERROR(
            "Param error,shape_in[%lu]!=1 when shape_in[%lu] != "
            "shape_out[%lu]",
            i, i, i);
        return aicpu::KERNEL_STATUS_PARAM_INVALID;
      }
      expand_to_axis = static_cast<uint64_t>(shape_out[i]);
      break_axis_num = i;
      break;
    }
  }
  if (shape_in.size() >= 1) {
    if (break_axis_num == 0) {
      for (uint64_t i = shape_in.size() - 1; i > 0; i--) {
        copy_size = copy_size * static_cast<uint64_t>(shape_in[i]);
      }
    } else {
      for (uint64_t i = shape_in.size() - 1; i > break_axis_num - 1; i--) {
        copy_size = copy_size * static_cast<uint64_t>(shape_in[i]);
      }
    }
  }

  uint64_t shape_in_size = 1;
  for (uint64_t i = 0; i < shape_in.size(); i++) {
    shape_in_size = shape_in_size * static_cast<uint64_t>(shape_in[i]);
  }

  uint64_t copy_num = shape_in_size / copy_size;

  for (uint64_t i = 0; i < copy_num; i++) {
    for (uint64_t j = 0; j < expand_to_axis; j++) {
      out_indexes.insert(out_indexes.end(), x_indexes.begin() + (i * copy_size),
                         x_indexes.begin() + ((i + 1) * copy_size));
    }
  }
  return aicpu::KERNEL_STATUS_OK;
}

template <typename T, typename IndexT>
uint32_t GetExpandIndex(std::vector<T> x_indexes,
                        std::vector<T>& out_indexes,
                        std::vector<IndexT> shape_in,
                        std::vector<IndexT>& shape_out,
                        std::vector<IndexT>& shape_out_temp) {
  IndexT expand_to_axis = static_cast<IndexT>(1);
  uint64_t break_axis_num = 0;

  for (int64_t j = static_cast<int64_t>(shape_in.size() - 1); j >= 0; j--) {
    if (shape_in[j] != shape_out[j]) {
      if (shape_in[j] != static_cast<IndexT>(1)) {
        KERNEL_LOG_ERROR(
            "Param error,shape_in[%lu]!=1 when shape_in[%lu] != "
            "shape_out[%lu]",
            j, j, j);
        return aicpu::KERNEL_STATUS_PARAM_INVALID;
      }
      expand_to_axis = shape_out[j];
      break_axis_num = j;
      break;
    }
  }

  std::vector<IndexT> shape_in_temp = shape_in;
  shape_in_temp[break_axis_num] = expand_to_axis;

  if (CopyExpandIndex<T, IndexT>(x_indexes, out_indexes, shape_in,
                                 shape_in_temp) != aicpu::KERNEL_STATUS_OK) {
    return aicpu::KERNEL_STATUS_PARAM_INVALID;
  }
  shape_out_temp = shape_in_temp;
  return aicpu::KERNEL_STATUS_OK;
}

template <typename T, typename IndexT>
uint32_t ExpandByLayer(std::vector<T> x_indexes,
                       std::vector<T>& out_indexes,
                       std::vector<IndexT> shape_in,
                       std::vector<IndexT>& shape_out) {
  std::vector<IndexT> shape_out_temp;
  bool flag_end = true;
  while (flag_end) {
    if (GetExpandIndex<T, IndexT>(x_indexes, out_indexes, shape_in, shape_out,
                                  shape_out_temp) != aicpu::KERNEL_STATUS_OK) {
      return aicpu::KERNEL_STATUS_PARAM_INVALID;
    }
    bool flag_in = false;
    for (uint64_t i = 0; i < shape_in.size(); i++) {
      if (shape_out[i] != shape_out_temp[i]) {
        flag_in = true;
        break;
      }
    }
    if (!flag_in) {
      flag_end = false;
    } else {
      x_indexes = out_indexes;
      shape_in = shape_out_temp;
    }
  }
  return aicpu::KERNEL_STATUS_OK;
}

template <typename T, typename IndexT>
uint32_t GetExpandShape(std::vector<T>& x_indexes1,
                        std::vector<IndexT>& shape_in,
                        std::vector<IndexT>& shape_out,
                        std::vector<int64_t> size0,
                        T* tensorBase) {
  for (uint64_t i = 0; i < size0.size(); i++) {
    shape_in.push_back(static_cast<IndexT>(size0[i]));
  }

  uint64_t input_num = 1;
  for (uint64_t i = 0; i < shape_in.size(); i++) {
    input_num = input_num * static_cast<uint64_t>(shape_in[i]);
  }

  for (uint64_t i = 0; i < input_num; i++) {
    x_indexes1.push_back(tensorBase[i]);
  }

  uint64_t input_dimension_a = shape_in.size();
  uint64_t output_dimension = shape_out.size();
  if (input_dimension_a < output_dimension) {
    shape_in.insert(shape_in.begin(), output_dimension - input_dimension_a,
                    static_cast<IndexT>(1));
  }

  for (uint64_t i = 0; i < shape_in.size(); i++) {
    if (shape_out[i] == static_cast<IndexT>(1) &&
        shape_in[i] != static_cast<IndexT>(1)) {
      shape_out[i] = shape_in[i];
    }
  }
}

template <typename T, typename IndexT>
uint32_t DoExpandCompute(aicpu::CpuKernelContext& ctx) {
  auto tensorBase =
      static_cast<T*>(ctx.Input(0)->GetData());  // input tensor address
  auto shapeData =
      static_cast<IndexT*>(ctx.Input(1)->GetData());  // input shape address
  auto outTensor =
      static_cast<T*>(ctx.Output(0)->GetData());  // output tensor address

  aicpu::Tensor* input_tensor0 = ctx.Input(0);  // input tensor
  aicpu::Tensor* input_tensor1 = ctx.Input(1);  // output tensor

  std::vector<int64_t> size0 =
      input_tensor0->GetTensorShape()->GetDimSizes();  // input tensor shape
  std::vector<int64_t> size1 =
      input_tensor1->GetTensorShape()->GetDimSizes();  // input shape shape

  if (size0.size() < 0) {
    KERNEL_LOG_ERROR("The dimension of input tensor < 0");
    return aicpu::KERNEL_STATUS_PARAM_INVALID;
  }

  if (size0.size() == 0) {
    size0.push_back(static_cast<int64_t>(1));
  }

  std::vector<T> out_indexes;
  std::vector<T> x_indexes1;
  std::vector<IndexT> shape_in;
  std::vector<IndexT> shape_out;
  for (int64_t i = 0; i < size1[0]; i++) {
    shape_out.push_back(shapeData[i]);
  }
  GetExpandShape(x_indexes1, shape_in, shape_out, size0, tensorBase);

  uint64_t input_flag = 0;
  for (uint64_t i = 0; i < shape_in.size(); i++) {
    if (shape_in[i] != shape_out[i]) {
      input_flag = 1;
    }
  }

  if (input_flag == 0) {
    for (uint64_t i = 0; i < x_indexes1.size(); i++) {
      outTensor[i] = x_indexes1[i];
    }
    return aicpu::KERNEL_STATUS_OK;
  }

  ExpandByLayer<T, IndexT>(x_indexes1, out_indexes, shape_in, shape_out);

  uint64_t output_num = 1;
  for (uint64_t i = 0; i < shape_out.size(); i++) {
    output_num = output_num * static_cast<uint64_t>(shape_out[i]);
  }

  for (uint64_t i = 0; i < out_indexes.size(); i++) {
    outTensor[i] = out_indexes[i];
  }

  return aicpu::KERNEL_STATUS_OK;
}

template <typename IndicesType>
uint32_t IndicesExpandCompute(aicpu::CpuKernelContext& ctx) {
  aicpu::DataType params_type = ctx.Input(0)->GetDataType();
  std::map<int, std::function<uint32_t(aicpu::CpuKernelContext&)>> calls;
  calls[aicpu::DT_FLOAT16] = DoExpandCompute<Eigen::half, IndicesType>;
  calls[aicpu::DT_FLOAT] = DoExpandCompute<float, IndicesType>;
  calls[aicpu::DT_INT8] = DoExpandCompute<int8_t, IndicesType>;
  calls[aicpu::DT_INT32] = DoExpandCompute<int32_t, IndicesType>;
  calls[aicpu::DT_INT64] = DoExpandCompute<int64_t, IndicesType>;
  calls[aicpu::DT_UINT8] = DoExpandCompute<uint8_t, IndicesType>;

  if ((params_type != aicpu::DT_INT32) && (params_type != aicpu::DT_INT8) &&
      (params_type != aicpu::DT_UINT8) && (params_type != aicpu::DT_FLOAT16) &&
      (params_type != aicpu::DT_FLOAT) && (params_type != aicpu::DT_INT64)) {
    return aicpu::KERNEL_STATUS_PARAM_INVALID;
  }

  uint32_t ret = calls[params_type](ctx);
  return ret;
}
}  // namespace expand

namespace aicpu {
const char* kExpand = "Expand";
uint32_t ExpandCpuKernel::Compute(CpuKernelContext& ctx) {
  KERNEL_LOG_INFO("ExpandCpuKernel start.");
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "Check Expand params failed.");

  Tensor* indices = ctx.Input(1);
  DataType indices_type = indices->GetDataType();

  std::map<int, std::function<uint32_t(CpuKernelContext&)>> calls;
  calls[DT_FLOAT16] = expand::IndicesExpandCompute<Eigen::half>;
  calls[DT_FLOAT] = expand::IndicesExpandCompute<float>;
  calls[DT_INT8] = expand::IndicesExpandCompute<int8_t>;
  calls[DT_INT32] = expand::IndicesExpandCompute<int32_t>;
  calls[DT_INT64] = expand::IndicesExpandCompute<int64_t>;
  calls[DT_UINT8] = expand::IndicesExpandCompute<uint8_t>;

  if ((indices_type != DT_INT32) && (indices_type != DT_INT8) &&
      (indices_type != DT_UINT8) && (indices_type != DT_FLOAT16) &&
      (indices_type != DT_FLOAT) && (indices_type != DT_INT64)) {
    return KERNEL_STATUS_PARAM_INVALID;
  }

  uint32_t ret = calls[indices_type](ctx);
  return ret;
}

REGISTER_CPU_KERNEL(kExpand, ExpandCpuKernel);
}  // namespace aicpu