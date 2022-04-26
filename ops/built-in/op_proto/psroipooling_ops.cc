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
 * \file psroipooling_ops.cpp
 * \brief
 */
#include <cmath>
#include <vector>
#include <string>
#include "util/util.h"
#include "util/error_util.h"
#include "op_log.h"
#include "inc/nn_detect_ops.h"

namespace ge {

IMPLEMT_INFERFUNC(PSROIPooling, PSROIPoolingInferShape) {
    auto output_dim = op.get_attr_output_dim();
    auto group_size = op.get_attr_group_size();
    // The value of group_size must be less than 128
    if (group_size <= 0 || group_size >= 128) {
        std::string err_msg = GetAttrValueErrMsg("group_size", std::to_string(group_size), ConcatString("less than 128 and greater than 0"));
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
    }

    auto x_shape = op.get_input_desc_x().GetShape();
    auto x_dtype = op.get_input_desc_x().GetDataType();
    int64_t pool_h = group_size;
    int64_t pool_w = group_size;

    int64_t c_output_dim = x_shape.GetDim(1) / (group_size * group_size);
    if (c_output_dim != output_dim) {
        string err_msg1 = ConcatString("c_output_dim shoule be equal to output_dim, c_output_dim:",c_output_dim, ", output_dim:",output_dim);
        std::string err_msg = OtherErrMsg(err_msg1);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
    }

    auto rois_shape = op.get_input_desc_rois().GetShape();
    int64_t rois_num = rois_shape.GetDim(0) * rois_shape.GetDim(2);

    vector<int64_t> y_shape({rois_num, output_dim, pool_h, pool_w});

    auto out_desc = op.GetOutputDesc("y");
    out_desc.SetShape(Shape(y_shape));
    out_desc.SetDataType(ge::DataType(x_dtype));
    (void)op.update_output_desc_y(out_desc);

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(PSROIPooling, PSROIPoolingVerify) {
    // input x only support NCHW format
    auto x_shape = op.get_input_desc_x().GetShape();
    if (x_shape.GetDims().size() != 4) {
        std::string err_msg = GetShapeSizeErrMsg(0, ConcatString(x_shape.GetDims().size()), ConcatString(4));
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
    }

    Format x_format = op.get_input_desc_x().GetFormat();
    if (x_format != FORMAT_NCHW) {
        std::string err_msg = GetInputFormatErrMsg ("x_format", ConcatString("NCHW"), ConcatString("x_format"));
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
    }

    // rois shape is (batch, 5, rois_num), shape size is 3
    auto rois_shape = op.get_input_desc_rois().GetShape();
    if (rois_shape.GetDims().size() < 3) {
        std::string err_msg = GetShapeSizeErrMsg(1, ConcatString(rois_shape.GetDims().size()), ConcatString("more than or equal to 3"));
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(PSROIPooling, PSROIPoolingInferShape);
VERIFY_FUNC_REG(PSROIPooling, PSROIPoolingVerify);

IMPLEMT_INFERFUNC(PSROIPoolingV2, PSROIPoolingV2InferShape) {
    auto output_dim = op.get_attr_output_dim();
    auto group_size = op.get_attr_group_size();
    // The value of group_size must be less than 128
    if (group_size <= 0 || group_size >= 128) {
        string correct_size = ConcatString("less than 128 and greater than 0");
        std::string err_msg = GetAttrValueErrMsg("group_size", std::to_string(group_size), correct_size);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
    }

    auto x_shape = op.GetInputDesc("x").GetShape();
    auto x_dtype = op.GetInputDesc("x").GetDataType();
    int64_t pool_h = group_size;
    int64_t pool_w = group_size;
    int64_t c_output_dim = x_shape.GetDim(1) / (group_size * group_size);
    if (c_output_dim != output_dim) {
        string err_msg1 = ConcatString("c_output_dim shoule be equal to output_dim, c_output_dim:",c_output_dim, ", output_dim:",output_dim);
        std::string err_msg = OtherErrMsg(err_msg1);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
    }

    auto rois_shape = op.get_input_desc_rois().GetShape();
    int64_t rois_num = rois_shape.GetDim(0) * rois_shape.GetDim(2);
    vector<int64_t> y_shape({rois_num, output_dim, pool_h, pool_w});

    auto out_desc = op.GetOutputDesc("y");
    out_desc.SetShape(Shape(y_shape));
    out_desc.SetDataType(ge::DataType(x_dtype));
    (void)op.update_output_desc_y(out_desc);

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(PSROIPoolingV2, PSROIPoolingV2Verify) {
    // input x only support NCHW format
    auto x_shape = op.get_input_desc_x().GetShape();
    if (x_shape.GetDims().size() != 4) {
        std::string err_msg = GetShapeSizeErrMsg(0, std::to_string(x_shape.GetDims().size()), ConcatString("4"));
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
    }

    Format x_format = op.get_input_desc_x().GetFormat();
    if (x_format != FORMAT_NCHW) {
        std::string err_msg = GetInputFormatErrMsg("x_format", ConcatString("NCHW"), ConcatString(x_format));
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
    }

    // rois shape is (batch, 5, rois_num), shape size is 3
    auto rois_shape = op.get_input_desc_rois().GetShape();
    if (rois_shape.GetDims().size() < 3) {
        string correct_size = ConcatString("more than or equal to 3");
        std::string err_msg = GetShapeSizeErrMsg(1, std::to_string(rois_shape.GetDims().size()), correct_size);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(PSROIPoolingV2, PSROIPoolingV2InferShape);
VERIFY_FUNC_REG(PSROIPoolingV2, PSROIPoolingV2Verify);

IMPLEMT_INFERFUNC(PSROIPoolingGradV2D, PSROIPoolingGradV2DInferShape) {
    auto output_dim = op.get_attr_output_dim();
    auto group_size = op.get_attr_group_size();
    auto input_size = op.get_attr_input_size();

    // The value of group_size must be less than 128
    if (group_size <= 0 || group_size >= 128) {
        string correct_size = ConcatString("less than 128 and greater than 0");
        std::string err_msg = GetAttrValueErrMsg("group_size", std::to_string(group_size), correct_size);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
    }

    if (input_size.size() != 2) {
        std::string err_msg = GetAttrSizeErrMsg("input_size", std::to_string(input_size.size()), ConcatString("2"));
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
    }

    auto x_shape = op.GetInputDesc("x").GetShape();
    auto x_dtype = op.GetInputDesc("x").GetDataType();

    int64_t c_output_dim = x_shape.GetDim(1);
    if (c_output_dim != output_dim) {
        string err_msg1 = ConcatString("c_output_dim shoule be equal to output_dim, c_output_dim:",c_output_dim, ", output_dim:",output_dim);
        std::string err_msg = OtherErrMsg(err_msg1);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
    }

    auto rois_shape = op.get_input_desc_rois().GetShape();
    int64_t rois_batch = rois_shape.GetDim(0);
    int64_t output_h = input_size[0];
    int64_t output_w = input_size[1];
    int64_t output_c = group_size * group_size * output_dim;
    std::vector<int64_t> y_shape({rois_batch, output_c, output_h, output_w});

    auto out_desc = op.GetOutputDesc("y");
    out_desc.SetShape(Shape(y_shape));
    out_desc.SetDataType(ge::DataType(x_dtype));
    (void)op.UpdateOutputDesc("y", out_desc);

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(PSROIPoolingGradV2D, PSROIPoolingGradV2DVerify) {
    // input x only support NCHW format
    auto x_shape = op.get_input_desc_x().GetShape();
    if (x_shape.GetDims().size() != 4) {
        std::string err_msg = GetShapeSizeErrMsg(0, std::to_string(x_shape.GetDims().size()), ConcatString("4"));
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
    }

    Format x_format = op.get_input_desc_x().GetFormat();
    if (x_format != FORMAT_NCHW) {
        std::string err_msg = GetInputFormatErrMsg("x_format", ConcatString("NCHW"), ConcatString(x_format));
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
    }

    // rois shape is (batch, 5, rois_num), shape size is 3
    auto rois_shape = op.get_input_desc_rois().GetShape();
    if (rois_shape.GetDims().size() < 3) {
        string correct_size = ConcatString("more than or equal to 3");
        std::string err_msg = GetShapeSizeErrMsg(1, std::to_string(rois_shape.GetDims().size()), correct_size);
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(PSROIPoolingGradV2D, PSROIPoolingGradV2DInferShape);
VERIFY_FUNC_REG(PSROIPoolingGradV2D, PSROIPoolingGradV2DVerify);
}  // namespace ge
