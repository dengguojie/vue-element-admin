/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 */

#include "tile_with_axis.h"

#include <iostream>

#include "cpu_kernel_utils.h"
#include "securec.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kTileWithAxis = "TileWithAxis";
}
namespace aicpu {
template <typename T, int32_t OPTION, int32_t DIMS>
uint32_t TileWithAxisCpuKernel::TileComputeByAxis(CpuKernelContext &ctx) {
  Tensor *input = ctx.Input(kFirstInputIndex);
  Tensor *output = ctx.Output(kFirstOutputIndex);
  
  int32_t axis = 1;
  AttrValue *axis_attr = ctx.GetAttr("axis");
  if(axis_attr != nullptr) {
    axis = axis_attr->GetInt();
    axis = (axis < 0) ? (axis + DIMS) : axis;
  }
  
  // tiles == 1, output跟input一样
  int32_t tiles = ctx.GetAttr("tiles")->GetInt();
  if (tiles == 1) {
    T *input0 = reinterpret_cast<T *>(input->GetData());
    T *output0 = reinterpret_cast<T *>(output->GetData());
    (void)memcpy_s(output0, output->GetDataSize(), input0,
                   output->GetDataSize());

    return KERNEL_STATUS_OK;
  }

  // 获取input和output的shape
  std::vector<int64_t> shape_input = input->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_output = output->GetTensorShape()->GetDimSizes();
  // output赋值, reshape bcast
  Eigen::DSizes<Eigen::DenseIndex, DIMS> in_reshape;
  Eigen::DSizes<Eigen::DenseIndex, DIMS> out_reshape;
  Eigen::array<Eigen::DenseIndex, DIMS> bcast;
  for (int32_t i = 0; i < DIMS; i++) {
    in_reshape[DIMS - i - 1] = shape_input[i];
    out_reshape[DIMS - i - 1] = shape_output[i];
    bcast[i] = (i == (DIMS - axis - 1)) ? tiles : 1;
  }

  Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> input0(
      static_cast<T *>(input->GetData()),
      input->GetTensorShape()->NumElements());
  Eigen::TensorMap<Eigen::Tensor<T, 1>, OPTION> output0(
      static_cast<T *>(output->GetData()),
      output->GetTensorShape()->NumElements());
  output0.reshape(out_reshape) = input0.reshape(in_reshape).broadcast(bcast);

  return KERNEL_STATUS_OK;
}

template <typename T, int32_t OPTION>
uint32_t TileWithAxisCpuKernel::TileComputeInDims(CpuKernelContext &ctx) {
  int32_t dims = ctx.Output(kFirstOutputIndex)->GetTensorShape()->GetDims();
  switch (dims) {
    case 0: {
      T *input0 = reinterpret_cast<T *>(ctx.Input(kFirstInputIndex)->GetData());
      T *output0 =
          reinterpret_cast<T *>(ctx.Output(kFirstOutputIndex)->GetData());
      for (uint32_t i = 0; i < ctx.Output(kFirstOutputIndex)->GetDataSize();
           i++) {
        output0[i] = *input0;
      }
      return KERNEL_STATUS_OK;
    }
    case 1:
      return TileComputeByAxis<T, OPTION, 1>(ctx);
    case 2:
      return TileComputeByAxis<T, OPTION, 2>(ctx);
    case 3:
      return TileComputeByAxis<T, OPTION, 3>(ctx);
    case 4:
      return TileComputeByAxis<T, OPTION, 4>(ctx);
    case 5:
      return TileComputeByAxis<T, OPTION, 5>(ctx);
    case 6:
      return TileComputeByAxis<T, OPTION, 6>(ctx);
    case 7:
      return TileComputeByAxis<T, OPTION, 7>(ctx);
    case 8:
      return TileComputeByAxis<T, OPTION, 8>(ctx);
    default:
      KERNEL_LOG_ERROR("[%s] Rank of output should less than 8 but get [%zu].",
                       ctx.GetOpType().c_str(), dims);
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T>
uint32_t TileWithAxisCpuKernel::TileCompute(CpuKernelContext &ctx) {
  bool flag = AddrAlignedCheck(ctx.Input(kFirstInputIndex)->GetData());
  if (flag) {
    return TileComputeInDims<T, Eigen::Aligned>(ctx);
  } else {
    return TileComputeInDims<T, Eigen::Unaligned>(ctx);
  }
}

uint32_t TileWithAxisCpuKernel::TileParaCheck(CpuKernelContext &ctx) {
  Tensor *input = ctx.Input(kFirstInputIndex);
  Tensor *output = ctx.Output(kFirstOutputIndex);
  int32_t axis = 1;

  // 判断axis是不是在input的轴范围内，可选参数
  AttrValue *axis_attr = ctx.GetAttr("axis");
  if(axis_attr != nullptr) {
    axis = axis_attr->GetInt();
    auto dims = input->GetTensorShape()->GetDims();
    axis = (axis < 0) ? (axis + dims) : axis;
    if (axis < 0 || axis >= dims) {
        KERNEL_LOG_ERROR("TileWithAxis axis[%d] is valid.", axis);
        return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  
  // 判断tiles必须大于0，必选参数
  AttrValue *tiles_attr = ctx.GetAttr("tiles");
  if(tiles_attr == nullptr) {
    KERNEL_LOG_ERROR("TileWithAxis tiles is null.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int32_t tiles = tiles_attr->GetInt();
  if (tiles <= 0) {
    KERNEL_LOG_ERROR("TileWithAxis tiles[%d] is valid.", tiles);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  
  // 判断outshape轴axis的大小是否是input轴axis大小*tiles
  std::vector<int64_t> shape_input = input->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_output = output->GetTensorShape()->GetDimSizes();
  if (shape_output[axis] != shape_input[axis] * tiles) {
    KERNEL_LOG_ERROR("TileWithAxis output_shape[%d] is valid.", axis);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t TileWithAxisCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, 1, 1), "TileWithAxis NormalCheck fail.");
  KERNEL_HANDLE_ERROR(TileParaCheck(ctx), "TileWithAxis TileParaCheck fail.");

  auto data_type =
      static_cast<DataType>(ctx.Input(kFirstInputIndex)->GetDataType());
  switch (data_type) {
    case DT_FLOAT16:
      return TileCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return TileCompute<float>(ctx);
    case DT_INT64:
      return TileCompute<int64_t>(ctx);
    case DT_INT32:
      return TileCompute<int32_t>(ctx);
    case DT_INT16:
      return TileCompute<int16_t>(ctx);
    case DT_INT8:
      return TileCompute<int8_t>(ctx);
    case DT_UINT64:
      return TileCompute<uint64_t>(ctx);
    case DT_UINT32:
      return TileCompute<uint32_t>(ctx);
    case DT_UINT16:
      return TileCompute<uint16_t>(ctx);
    case DT_UINT8:
      return TileCompute<uint8_t>(ctx);
    default:
      KERNEL_LOG_ERROR("TileWithAxis dtype is invalid.");
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

REGISTER_CPU_KERNEL(kTileWithAxis, TileWithAxisCpuKernel);
}  // namespace aicpu