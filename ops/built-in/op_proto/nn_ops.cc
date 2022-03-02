/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file nn_ops.cpp
 * \brief
 */
#include "inc/nn_ops.h"
#include <cmath>
#include <string>
#include <vector>
#include "util/common_shape_fns.h"
#include "util/util.h"
#include "util/error_util.h"
#include "op_log.h"
#include "graph/utils/op_desc_utils.h"

namespace ge {
bool InTopKV2CheckInput(const Operator& op) {
  Shape shape_prediction = op.GetInputDesc("predictions").GetShape();
  Shape shape_target = op.GetInputDesc("targets").GetShape();
  int prediction_dim = shape_prediction.GetDimNum();
  if (prediction_dim != DIM_SIZE2) {
    OP_LOGE(op.GetName().c_str(), "Predictions must be 2-dimensional, but get [%d]", prediction_dim);
    return false;
  }
  size_t target_dim = shape_target.GetDimNum();
  if (target_dim != DIM_SIZE1) {
    OP_LOGE(op.GetName().c_str(), "Targets must be 1-dimensional but get [%lu]", target_dim);
    return false;
  }
  if (shape_prediction.GetDim(0) != shape_target.GetDim(0)) {
    OP_LOGE(op.GetName().c_str(),
            "First dimension of predictions must match length of targets, but first dimension of predictions get [%ld] "
            "and targets get [%lu]", shape_prediction.GetDim(0), shape_target.GetDim(0));
    return false;
  }
  return true;
}

IMPLEMT_VERIFIER(InTopKV2, InTopKV2Verify) {
  if (!InTopKV2CheckInput(op)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(InTopKV2InferShape) {
  DataType output_dtype = DT_BOOL;
  Shape shape_target = op.GetInputDesc("targets").GetShape();
  TensorDesc tensordesc_output = op.GetOutputDesc("precision");
  tensordesc_output.SetShape(shape_target);
  tensordesc_output.SetDataType(output_dtype);
  if (op.UpdateOutputDesc("precision", tensordesc_output) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(InTopKV2, InTopKV2InferShape);
VERIFY_FUNC_REG(InTopKV2, InTopKV2Verify);

IMPLEMT_INFERFUNC(FusedBatchNormV2, FusedBatchNormV2Infer) {
  Shape xshape;
  if (WithRank(op.GetInputDesc("x"), 4, xshape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Input x rank must be 4");
    return GRAPH_FAILED;
  }
  bool is_training;
  if (op.GetAttr("is_training", is_training) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr is_training failed");
    return GRAPH_FAILED;
  }
  int number_inputs = (is_training) ? 3 : 5;
  string data_format;
  if (op.GetAttr("data_format", data_format) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr data_format failed");
    return GRAPH_FAILED;
  } else {
    if (data_format != "NCHW" && data_format != "NHWC") {
      OP_LOGE(op.GetName().c_str(), "Attr data_format [%s] only support NCHW and NHWC", data_format.c_str());
      return GRAPH_FAILED;
    }
  }
  int64_t channel_dim = 0;
  int channel_dim_index = 0;
  if (data_format == "NHWC") {
    channel_dim_index = 3;
    channel_dim = xshape.GetDim(channel_dim_index);
  }
  if (data_format == "NCHW") {
    channel_dim_index = 1;
    channel_dim = xshape.GetDim(channel_dim_index);
  }
  for (int i = 1; i < number_inputs; ++i) {
    Shape vec;
    if (op.GetInputDesc(i).GetDataType() != DT_FLOAT) {
      OP_LOGE(op.GetName().c_str(), "Input[%d] type must be DT_FLOAT", i);
      return GRAPH_FAILED;
    }
    if (WithRank(op.GetInputDesc(i), 1, vec, op.GetName().c_str()) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "Input[%d] rank must be 1", i);
      return GRAPH_FAILED;
    }
    int64_t dim0 = vec.GetDim(0);
    if (Merge(channel_dim, dim0, channel_dim) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "Channel_dim [%ld] and input[%d]'s dim0 [%ld] should same length", channel_dim, i,
              dim0);
      return GRAPH_FAILED;
    }
  }
  Shape yshape;
  if (ReplaceDim(xshape, channel_dim_index, channel_dim, yshape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to replacedim from xshape");
    return GRAPH_FAILED;
  }
  DataType xtype = op.GetInputDesc("x").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDesc(0);
  tensordesc_output.SetDataType(xtype);
  tensordesc_output.SetShape(yshape);
  if (op.UpdateOutputDesc("y", tensordesc_output) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to update y desc");
    return GRAPH_FAILED;
  }
  Shape vector_shape = Shape({channel_dim});
  tensordesc_output = op.GetOutputDesc(1);
  tensordesc_output.SetDataType(DT_FLOAT);
  tensordesc_output.SetShape(vector_shape);
  if (op.UpdateOutputDesc("batch_mean", tensordesc_output) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to update batch_mean desc");
    return GRAPH_FAILED;
  }
  tensordesc_output = op.GetOutputDesc(2);
  tensordesc_output.SetDataType(DT_FLOAT);
  tensordesc_output.SetShape(vector_shape);
  if (op.UpdateOutputDesc("batch_variance", tensordesc_output) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to update batch_variance desc");
    return GRAPH_FAILED;
  }
  tensordesc_output = op.GetOutputDesc(3);
  tensordesc_output.SetDataType(DT_FLOAT);
  tensordesc_output.SetShape(vector_shape);
  if (op.UpdateOutputDesc("reserve_space_1", tensordesc_output) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to update reserve_space_1 desc");
    return GRAPH_FAILED;
  }
  tensordesc_output = op.GetOutputDesc(4);
  tensordesc_output.SetDataType(DT_FLOAT);
  tensordesc_output.SetShape(vector_shape);
  if (op.UpdateOutputDesc("reserve_space_2", tensordesc_output) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to update reserve_space_1 desc");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(FusedBatchNormV2, FusedBatchNormV2Infer);
//-----------------------SegmentSort-------------------------
IMPLEMT_COMMON_INFERFUNC(SegmentSortInferShape) {
    TensorDesc tensordesc_output = op.GetOutputDesc("output_proposal");
    TensorDesc tensordesc_input = op.GetInputDesc("input_data");

    int64_t data_num = tensordesc_input.GetShape().GetDim(0);
    int64_t k_num = 0;
    if (GRAPH_SUCCESS != op.GetAttr("k_num", k_num)) {
        OP_LOGE(op.GetName().c_str(), "Get attr k_num failed");
        return GRAPH_FAILED;
    }

    const int64_t merge_channel = 4;
    const int64_t core_align_num = 1984;
    const int64_t core_min_num = 7936;
    const int64_t pro_repeat_num = 16;
    const int64_t pro_data_num = 8;
    int64_t ai_core_num = 32;

    int64_t result_data_num = (data_num + ai_core_num - 1) / ai_core_num;
    result_data_num = (result_data_num + core_align_num - 1) / core_align_num * core_align_num;
    if (result_data_num < core_min_num) {
        result_data_num = core_min_num;
    }

    ai_core_num = (data_num  + result_data_num - 1) / result_data_num;
    if (ai_core_num > merge_channel) {
        ai_core_num = (ai_core_num + merge_channel - 1) / merge_channel * merge_channel;
    }

    result_data_num = result_data_num + pro_repeat_num;

    vector<int64_t> output_shape;
    output_shape.push_back(ai_core_num);
    output_shape.push_back(result_data_num);
    output_shape.push_back(pro_data_num);

    tensordesc_output.SetShape(Shape(output_shape));
    tensordesc_output.SetDataType(tensordesc_input.GetDataType());
    (void)op.UpdateOutputDesc("output_proposal", tensordesc_output);
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SegmentSort, SegmentSortInferShape);
//-----------------------SegmentSort END---------------------
// ----------------------MultiMerge----------------------
IMPLEMT_COMMON_INFERFUNC(MultiMergeInferShape) {
    int64_t k_num = 0;
    if (GRAPH_SUCCESS != op.GetAttr("k_num", k_num)) {
        OP_LOGE(op.GetName().c_str(), "Get attr k_num failed");
        return GRAPH_FAILED;
    }
    bool include_index = false;
    op.GetAttr("include_index", include_index);
    if (include_index) {
        TensorDesc tensordesc_output_data = op.GetOutputDesc("output_proposal");
        TensorDesc tensordesc_output_index = op.GetOutputDesc("output_index");
        TensorDesc tensordesc_input = op.GetInputDesc("input_proposal");
        vector<int64_t> output_shape;
        output_shape.push_back(k_num);

        tensordesc_output_data.SetShape(Shape(output_shape));
        tensordesc_output_data.SetDataType(tensordesc_input.GetDataType());
        (void)op.UpdateOutputDesc("output_proposal", tensordesc_output_data);

        tensordesc_output_index.SetShape(Shape(output_shape));
        tensordesc_output_index.SetDataType(DT_INT32);
        (void)op.UpdateOutputDesc("output_index", tensordesc_output_index);
        return GRAPH_SUCCESS;
    } else {
        TensorDesc tensordesc_output = op.GetOutputDesc("output_proposal");
        TensorDesc tensordesc_input = op.GetInputDesc("input_proposal");

        int64_t channel_num = tensordesc_input.GetShape().GetDim(0);
        int64_t data_num = tensordesc_input.GetShape().GetDim(1);


        const int64_t merge_channel = 4;
        const int64_t pro_repeat_num = 16;
        const int64_t pro_data_num = 8;
        int64_t ai_core_num = 32;

        int64_t result_data_num = (data_num - pro_repeat_num) * merge_channel;
        k_num = (k_num + pro_repeat_num - 1) / pro_repeat_num * pro_repeat_num;
        if (k_num < result_data_num) {
            result_data_num = k_num;
        }
        result_data_num = result_data_num + pro_repeat_num;

        ai_core_num = channel_num / 4;
        if (ai_core_num > merge_channel) {
            ai_core_num = (ai_core_num + merge_channel - 1) / merge_channel * merge_channel;
        }

        vector<int64_t> output_shape;
        output_shape.push_back(ai_core_num);
        output_shape.push_back(result_data_num);
        output_shape.push_back(pro_data_num);

        tensordesc_output.SetShape(Shape(output_shape));
        tensordesc_output.SetDataType(tensordesc_input.GetDataType());
        (void)op.UpdateOutputDesc("output_proposal", tensordesc_output);

        TensorDesc tensordesc_output_index = op.GetOutputDesc("output_index");
        vector<int64_t> index_shape;
        index_shape.push_back(1);
        tensordesc_output_index.SetShape(Shape(index_shape));
        tensordesc_output_index.SetDataType(DT_INT32);
        (void)op.UpdateOutputDesc("output_index", tensordesc_output_index);
        return GRAPH_SUCCESS;
    }
}

COMMON_INFER_FUNC_REG(MultiMerge, MultiMergeInferShape);
//-----------------------MultiMerge END---------------------

}//namespace ge
