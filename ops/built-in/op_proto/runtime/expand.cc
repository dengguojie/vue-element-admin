/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
/*!
 * \file expand.cc
 * \brief
 */
#include "op_log.h"
#include "runtime_util.h"
#include "register/op_impl_registry.h"
#include "error_util.h"

using namespace ge;
namespace ops {
template <typename T>
static bool ExpandCalDim(const gert::Tensor *shape_tensor, std::vector<int64_t> &vec_dim, std::vector<int64_t> &x_dims,
    int64_t &len_shape)
{
    int64_t len_x = x_dims.size();
    int64_t diff = abs(len_x - len_shape);

    const T *shape_val = shape_tensor->GetData<T>();
    if (len_shape < len_x) {
        for (int64_t i = 0; i < diff; i++) {
            vec_dim.push_back(x_dims[i]);
        }
        for (int64_t i = diff; i < len_x; i++) {
            T dim = shape_val[i - diff];
            if ((x_dims[i] != dim) && (x_dims[i] != 1) && (dim != 1)) {
                return false;
            }
            if (x_dims[i] > dim) {
                vec_dim.push_back(x_dims[i]);
            } else {
                vec_dim.push_back(dim);
            }
        }
    } else {
        for (int64_t i = 0; i < diff; i++) {
            vec_dim.push_back(shape_val[i]);
        }
        for (int64_t i = diff; i < len_shape; i++) {
            T dim = shape_val[i];
            if ((x_dims[i - diff] != dim) && (x_dims[i - diff] != 1) && (dim != 1)) {
                return false;
            }
            if (x_dims[i - diff] > dim) {
                vec_dim.push_back(x_dims[i - diff]);
            } else {
                vec_dim.push_back(dim);
            }
        }
    }
    return true;
}

ge::graphStatus ExpandInferShapeFunc(gert::InferShapeContext *context)
{
    const char *op_name = "Expand";
    auto output_shape = context->GetOutputShape(0);
    
    auto x_shape = context->GetInputShape(0);
    int64_t len_x = (int64_t)x_shape->GetDimNum();
    std::vector<int64_t> x_dims;
    for (int64_t i = 0; i < len_x; i++) {
        x_dims.push_back(x_shape->GetDim(i));
    }

    auto shape_tensor = context->GetInputTensor(1);
    int64_t len_shape = shape_tensor->GetShapeSize();  
    std::vector<int64_t> vec_dim;

    DataType data_type = shape_tensor->GetDataType();
    if (data_type == DT_INT32) {
        if (!ExpandCalDim<int32_t>(shape_tensor, vec_dim, x_dims, len_shape)) {
            OP_LOGE(op_name, "Data shape are not compatible!");
            return GRAPH_FAILED;
        }
    } else if (data_type == DT_INT64) {
        if (!ExpandCalDim<int64_t>(shape_tensor, vec_dim, x_dims, len_shape)) {
            OP_LOGE(op_name, "Data shape are not compatible!");
            return GRAPH_FAILED;
        }
    } else {
        OP_LOGE(op_name, "Data type not supported!");
        return GRAPH_PARAM_INVALID;
    }

    int64_t len_res = vec_dim.size();
    output_shape->SetDimNum(len_res);
    for (int64_t i = 0; i < len_res; i++) {
        output_shape->SetDim(i, vec_dim[i]);
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP(Expand).InputsDataDependency({ 1 }).InferShape(ExpandInferShapeFunc);
} // namespace ops