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

#include "get_dynamic_dims.h"

#include "utils/kernel_util.h"
#include "cpu_types.h"
#include "log.h"
#include "securec.h"
#include "status.h"

namespace {
constexpr uint32_t kGetDynamicDimsOutputNum = 1;
constexpr const char *GET_DYNAMIC_DIMS = "GetDynamicDims";
} // namespace

namespace aicpu {
uint32_t GetDynamicDimsCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("GetDynamicDimsCpuKernel::Compute(), OpType:%s.",
                  GET_DYNAMIC_DIMS);
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kDynamicInput, kGetDynamicDimsOutputNum),
                      "%s check params failed.", GET_DYNAMIC_DIMS);

  // parse attr
  AttrValue *n_attr = ctx.GetAttr("N");
  KERNEL_CHECK_NULLPTR(n_attr, KERNEL_STATUS_PARAM_INVALID,
                       "%s get attr:N failed.", GET_DYNAMIC_DIMS);
  int64_t count = n_attr->GetInt();
  KERNEL_LOG_INFO("%s get attr:N: %ld.", GET_DYNAMIC_DIMS, count);

  AttrValue *shape_info_attr = ctx.GetAttr("shape_info");
  KERNEL_CHECK_NULLPTR(shape_info_attr, KERNEL_STATUS_PARAM_INVALID,
                       "%s get attr:shape_info failed.", GET_DYNAMIC_DIMS);
  std::vector<int64_t> shape_info = shape_info_attr->GetListInt();
  KERNEL_LOG_INFO("%s get attr:shape_info: %s.", GET_DYNAMIC_DIMS,
                  VectorToString(shape_info).c_str());
  std::vector<std::vector<int64_t>> shape_infos = GetShapeInfos(shape_info);

  // check inputs size
  uint32_t inputs_size = ctx.GetInputsSize();
  KERNEL_CHECK_FALSE(
      (inputs_size == count), KERNEL_STATUS_PARAM_INVALID,
      "%s inputs size [%zu] is not match attr N [%ld].",
      GET_DYNAMIC_DIMS, inputs_size, count);
  KERNEL_CHECK_FALSE(
      (inputs_size == shape_infos.size()), KERNEL_STATUS_PARAM_INVALID,
      "%s inputs size [%u] is not match shape_infos size [%zu].",
      GET_DYNAMIC_DIMS, inputs_size, shape_infos.size());

  // get input shapes
  std::vector<std::vector<int64_t>> input_shapes;
  KERNEL_HANDLE_ERROR(GetInputShapes(ctx, input_shapes),
                      "%s get input shapes failed.", GET_DYNAMIC_DIMS);

  // find -1 in shape_infos, and record corresponding input_dim into dims
  std::vector<int64_t> dims;
  for (uint32_t i = 0; i < inputs_size; ++i) {
    KERNEL_LOG_INFO("%s shape_infos[%u]: %s.", GET_DYNAMIC_DIMS, i,
                    VectorToString(shape_infos[i]).c_str());
    KERNEL_LOG_INFO("%s get input[%u]'s shape: %s.", GET_DYNAMIC_DIMS, i,
                    VectorToString(input_shapes[i]).c_str());
    KERNEL_CHECK_FALSE(
        (input_shapes[i].size() == shape_infos[i].size()),
        KERNEL_STATUS_PARAM_INVALID,
        "%s input[%u] rank [%zu] is not match shape_infos[%u] rank [%zu].",
        GET_DYNAMIC_DIMS, i, input_shapes[i].size(), i, shape_infos[i].size());

    for (size_t j = 0; j < input_shapes[i].size(); ++j) {
      if (shape_infos[i][j] == -1) {
        dims.push_back(input_shapes[i][j]);
      }
    }
  }
  KERNEL_LOG_INFO("%s unknown dims: %s.", GET_DYNAMIC_DIMS,
                  VectorToString(dims).c_str());

  // fill output data
  Tensor *output_tensor = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output_tensor, KERNEL_STATUS_INNER_ERROR,
                       "%s get output:0 failed.", GET_DYNAMIC_DIMS);
  void *output_data = output_tensor->GetData();
  uint64_t output_size = output_tensor->GetDataSize();

  errno_t cpret = memcpy_s(output_data, output_size, dims.data(),
                            dims.size() * sizeof(int64_t));
  KERNEL_CHECK_FALSE(
      (cpret == EOK), KERNEL_STATUS_INNER_ERROR,
      "%s memcpy_s to output failed, destMax=%ld, count=%zu.",
      GET_DYNAMIC_DIMS, output_size, dims.size() * sizeof(int64_t));
  return KERNEL_STATUS_OK;
}

std::vector<std::vector<int64_t>> GetDynamicDimsCpuKernel::GetShapeInfos(
    std::vector<int64_t> &shape_info) const {
  std::vector<std::vector<int64_t>> shape_infos;
  for (size_t i = 0; i < shape_info.size(); ++i) {
    int64_t rank = shape_info[i];
    std::vector<int64_t> shape;
    for (int64_t j = 0; j < rank; ++j) {
      shape.push_back(shape_info[++i]);
    }
    shape_infos.push_back(shape);
  }

  return shape_infos;
}

uint32_t GetDynamicDimsCpuKernel::GetInputShapes(
    CpuKernelContext &ctx,
    std::vector<std::vector<int64_t>> &input_shapes) const {
  for (uint32_t i = 0; i < ctx.GetInputsSize(); ++i) {
    Tensor *input_tensor = ctx.Input(i);
    KERNEL_CHECK_NULLPTR(input_tensor, KERNEL_STATUS_INNER_ERROR,
                         "%s get input:%u failed.", GET_DYNAMIC_DIMS, i);
    std::vector<int64_t> input_shape;
    int64_t input_size = input_tensor->NumElements();
    switch (input_tensor->GetDataType()) {
      case DT_INT32: {
        int32_t *input_data = static_cast<int32_t *>(input_tensor->GetData());
        input_shape.insert(input_shape.begin(), input_data,
                           input_data + input_size);
        break;
      }
      case DT_INT64: {
        int64_t *input_data = static_cast<int64_t *>(input_tensor->GetData());
        input_shape.insert(input_shape.begin(), input_data,
                           input_data + input_size);
        break;
      }
      default:
        KERNEL_LOG_ERROR("%s input:%u data_tpye must be in {int32 int64}.",
                         GET_DYNAMIC_DIMS, i);
        return KERNEL_STATUS_INNER_ERROR;
        break;
    }
    input_shapes.emplace_back(std::move(input_shape));
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(GET_DYNAMIC_DIMS, GetDynamicDimsCpuKernel);
} // namespace aicpu
