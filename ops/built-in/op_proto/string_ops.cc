/* *
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file sparse_ops.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "inc/string_ops.h"
#include "common_shape_fns.h"
#include "op_log.h"

namespace ge {
IMPLEMT_INFERFUNC(StringSplit, StringSplitInfer)
{
    Shape unused;
    auto tensor_input = op.get_input_desc_input();
    auto tensor_sep = op.get_input_desc_delimiter();

    if (WithRank(tensor_input, 1, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "The rank of input must be 1");
        return GRAPH_FAILED;
    }

    if (WithRank(tensor_sep, 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "The rank of delimiter must be 0");
        return GRAPH_FAILED;
    }

    Shape indices_shape;
    Shape values_shape;
    Shape shape;

    auto result = Matrix(ge::UNKNOWN_DIM, 2, indices_shape);
    if (result != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "generate indices_shape failed !");
        return GRAPH_FAILED;
    }
    TensorDesc indices_desc = op.get_output_desc_indices();
    indices_desc.SetShape(indices_shape);
    indices_desc.SetDataType(DT_INT64);
    if (op.UpdateOutputDesc("indices", indices_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update indices desc failed");
        return GRAPH_FAILED;
    }

    result = Vector(ge::UNKNOWN_DIM, values_shape);
    if (result != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "generate values_shape failed !");
        return GRAPH_FAILED;
    }
    TensorDesc values_desc = op.get_output_desc_values();
    values_desc.SetShape(values_shape);
    values_desc.SetDataType(DT_STRING);
    if (op.UpdateOutputDesc("values", values_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update values desc failed");
        return GRAPH_FAILED;
    }

    result = Vector(2, shape);
    if (result != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "generate shape failed !");
        return GRAPH_FAILED;
    }
    TensorDesc shape_desc = op.get_output_desc_shape();
    shape_desc.SetShape(shape);
    shape_desc.SetDataType(DT_INT64);
    if (op.UpdateOutputDesc("shape", shape_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update shape desc failed");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringSplit, StringSplitInfer);

IMPLEMT_INFERFUNC(StringSplitV2, StringSplitV2Infer)
{
    Shape unused;
    auto tensor_input = op.get_input_desc_input();
    auto tensor_sep = op.get_input_desc_sep();

    if (WithRank(tensor_input, 1, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "The rank of input must be 1");
        return GRAPH_FAILED;
    }

    if (WithRank(tensor_sep, 0, unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "The rank of sep must be 0");
        return GRAPH_FAILED;
    }

    Shape indices_shape;
    Shape values_shape;
    Shape shape;

    auto result = Matrix(ge::UNKNOWN_DIM, 2, indices_shape);
    if (result != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "generate indices_shape failed !");
        return GRAPH_FAILED;
    }
    TensorDesc indices_desc = op.get_output_desc_indices();
    indices_desc.SetShape(indices_shape);
    indices_desc.SetDataType(DT_INT64);
    if (op.UpdateOutputDesc("indices", indices_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update indices desc failed");
        return GRAPH_FAILED;
    }

    result = Vector(ge::UNKNOWN_DIM, values_shape);
    if (result != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "generate values_shape failed !");
        return GRAPH_FAILED;
    }
    TensorDesc values_desc = op.get_output_desc_values();
    values_desc.SetShape(values_shape);
    values_desc.SetDataType(DT_STRING);
    if (op.UpdateOutputDesc("values", values_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update values desc failed");
        return GRAPH_FAILED;
    }

    result = Vector(2, shape);
    if (result != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "generate shape failed !");
        return GRAPH_FAILED;
    }
    TensorDesc shape_desc = op.get_output_desc_shape();
    shape_desc.SetShape(shape);
    shape_desc.SetDataType(DT_INT64);
    if (op.UpdateOutputDesc("shape", shape_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update shape desc failed");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringSplitV2, StringSplitV2Infer);

IMPLEMT_INFERFUNC(UnicodeScript, UnicodeScriptInfer)
{
    DataType y_type = op.GetInputDesc("x").GetDataType();
    TensorDesc desc = op.GetOutputDesc("y");
    desc.SetShape(op.GetInputDesc("x").GetShape());
    desc.SetDataType(y_type);

    if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update y desc failed.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(UnicodeScript, UnicodeScriptInfer);

IMPLEMT_INFERFUNC(Substr, SubstrInfer)
{
    auto pos_tensor = op.GetInputDesc(1);
    Shape pos_shape = op.GetInputDesc(1).GetShape();
    Shape len_shape = op.GetInputDesc(2).GetShape();
    Shape unused;
    if (WithRank(pos_tensor, len_shape.GetDimNum(), unused, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "pos and len must have same rank");
        return GRAPH_FAILED;
    }
    for (size_t i = 0; i < pos_shape.GetDimNum(); ++i) {
        auto pos_dim = pos_shape.GetDim(i);
        auto len_dim = len_shape.GetDim(i);
        if (pos_dim != len_dim) {
            OP_LOGE(op.GetName().c_str(), "pos and len must have same dim");
            return GRAPH_FAILED;
        }
    }

    TensorDesc desc = op.GetOutputDesc(0);
    desc.SetDataType(DT_STRING);
    if (op.UpdateOutputDesc("output", desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update output desc failed.");
        return GRAPH_FAILED;
    }

    auto outputFunc = BROADCAST_INFER("input", "pos", "output");
    return outputFunc(op);
}

INFER_FUNC_REG(Substr, SubstrInfer);

IMPLEMT_INFERFUNC(StringToHashBucketFast, StringToHashBucketFastInfer)
{
    TensorDesc desc = op.GetOutputDesc("y");
    desc.SetShape(op.GetInputDesc("x").GetShape());
    desc.SetDataType(DT_INT64);

    if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update y desc failed.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringToHashBucketFast, StringToHashBucketFastInfer);

IMPLEMT_INFERFUNC(StringToHashBucketStrong, StringToHashBucketStrongInfer)
{
    DataType x_type = op.GetInputDesc("x").GetDataType();
    if (x_type != DT_STRING) {
        OP_LOGE(op.GetName().c_str(), " illegal when input type is not DT_STRING");
        return GRAPH_PARAM_INVALID;
    }
    TensorDesc desc = op.GetOutputDesc("y");
    desc.SetShape(op.GetInputDesc("x").GetShape());
    desc.SetDataType(DT_INT64);

    if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update y desc failed.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringToHashBucketStrong, StringToHashBucketStrongInfer);

IMPLEMT_INFERFUNC(StringToHashBucket, StringToHashBucketInfer)
{
    TensorDesc desc = op.GetOutputDesc("y");
    desc.SetShape(op.GetInputDesc(0).GetShape());
    desc.SetDataType(DT_INT64);

    if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update y desc failed.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringToHashBucket, StringToHashBucketInfer);

IMPLEMT_INFERFUNC(StringStrip, StringStripInfer)
{
    TensorDesc desc = op.GetOutputDesc("y");
    desc.SetShape(op.GetInputDesc(0).GetShape());
    desc.SetDataType(DT_STRING);

    if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update y desc failed.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringStrip, StringStripInfer);

IMPLEMT_INFERFUNC(StringLength, StringLengthInfer)
{
    TensorDesc desc = op.GetOutputDesc("y");
    desc.SetShape(op.GetInputDesc(0).GetShape());
    desc.SetDataType(DT_INT32);

    if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update y desc failed.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringLength, StringLengthInfer);

IMPLEMT_INFERFUNC(StringJoin, StringJoinInfer)
{
    size_t input_size = op.GetInputsSize();
    bool all_scalar = true;
    for (size_t i = 0; i < input_size; ++i) {
        if (op.GetInputDesc(i).GetShape().GetDimNum() != 0) {
            all_scalar = false;
        }
    }

    TensorDesc desc = op.GetOutputDesc("y");
    desc.SetDataType(DT_STRING);
    if (all_scalar) {
        desc.SetShape(Shape());
        if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
            OP_LOGE(op.GetName().c_str(), "update y desc failed.");
            return GRAPH_FAILED;
        }
        return GRAPH_SUCCESS;
    }

    Shape out(ge::UNKNOWN_SHAPE);
    for (size_t i = 0; i < input_size; ++i) {
        Shape input_shape = op.GetInputDesc(i).GetShape();
        if ((RankKnown(input_shape)) && (input_shape.GetDimNum() != 0)) {
            if (Merge(out, input_shape, out, op.GetName().c_str()) != GRAPH_SUCCESS) {
                OP_LOGE(op.GetName().c_str(), "merge two dimension error.");
                return GRAPH_FAILED;
            }
        }
    }
    desc.SetShape(out);
    if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update y desc failed.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringJoin, StringJoinInfer);

IMPLEMT_INFERFUNC(StringFormat, StringFormatInfer)
{
    string template_;
    string placeholder;
    if (op.GetAttr("template", template_) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "getattr template failed.");
        return GRAPH_FAILED;
    }
    if (op.GetAttr("placeholder", placeholder) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "getattr placeholder failed.");
        return GRAPH_FAILED;
    }
    if (op.GetInputsSize() == 0) {
        OP_LOGE(op.GetName().c_str(), "input size is 0, it is not illegal.");
        return GRAPH_FAILED;
    }
    std::istringstream str_template(template_);
    std::string token;
    size_t pos = -1;
    size_t num_placeholders = 0;
    while (str_template >> token) {
        while ((pos = token.rfind(placeholder)) != std::string::npos) {
            num_placeholders++;
            token.erase(pos, 1);
        }
    }

    if (op.GetInputsSize() != num_placeholders) {
        OP_LOGE(op.GetName().c_str(), "num placeholders in template and num inputs must match.");
        return GRAPH_FAILED;
    }

    TensorDesc desc = op.GetOutputDesc("y");
    desc.SetDataType(DT_STRING);
    desc.SetShape(Shape());

    if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update y desc failed.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StringFormat, StringFormatInfer);

IMPLEMT_INFERFUNC(RegexFullMatch, RegexFullMatchInfer)
{
    Shape un_used;
    if (WithRank(op.GetInputDesc("pattern"), 0, un_used, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input pattern must be 0-D");
        return GRAPH_FAILED;
    }
    Shape x_shape = op.GetInputDesc("x").GetShape();
    TensorDesc y_desc = op.GetOutputDesc("y");
    y_desc.SetShape(x_shape);
    y_desc.SetDataType(DT_BOOL);
    if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update y failed");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(RegexFullMatch, RegexFullMatchInfer);

IMPLEMT_INFERFUNC(RegexReplace, RegexReplaceInfer)
{
    Shape un_used;
    if (WithRank(op.GetInputDesc("pattern"), 0, un_used, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input pattern must be 0-D");
        return GRAPH_FAILED;
    }
    if (WithRank(op.GetInputDesc("rewrite"), 0, un_used, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input rewrite must be 0-D");
        return GRAPH_FAILED;
    }
    Shape x_shape = op.GetInputDesc("x").GetShape();
    TensorDesc y_desc = op.GetOutputDesc("y");
    y_desc.SetShape(x_shape);
    y_desc.SetDataType(DT_STRING);
    if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update y failed");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(RegexReplace, RegexReplaceInfer);

IMPLEMT_INFERFUNC(AsString, AsStringInfer)
{
    TensorDesc out_desc = op.GetOutputDesc("y");
    out_desc.SetDataType(DT_STRING);
    if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update y failed");
        return GRAPH_FAILED;
    }
    return UnchangedShape(op, "x", "y");
}

INFER_FUNC_REG(AsString, AsStringInfer);

IMPLEMT_INFERFUNC(EncodeBase64, EncodeBase64Infer)
{
    TensorDesc out_desc = op.GetOutputDesc("y");
    out_desc.SetDataType(DT_STRING);
    if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update y failed");
        return GRAPH_FAILED;
    }
    return UnchangedShape(op, "x", "y");
}

INFER_FUNC_REG(EncodeBase64, EncodeBase64Infer);

IMPLEMT_INFERFUNC(DecodeBase64, DecodeBase64Infer)
{
    TensorDesc out_desc = op.GetOutputDesc("y");
    out_desc.SetDataType(DT_STRING);
    if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "update y failed");
        return GRAPH_FAILED;
    }
    return UnchangedShape(op, "x", "y");
}

INFER_FUNC_REG(DecodeBase64, DecodeBase64Infer);
}