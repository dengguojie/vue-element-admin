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
 * \file image_ops.cpp
 * \brief
 */
#include "inc/image_ops.h"

#include <math.h>

#include "util/images_ops_shape_fns.h"
#include "util/util.h"
#include "util/common_shape_fns.h"
#include "util/error_util.h"
#include "op_log.h"
#include "op_const.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "axis_util.h"
#include "inc/graph/utils/type_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
IMPLEMT_INFERFUNC(DecodeGif, DecodeGifInfer) {
  const char *op_name = TbeGetName(op).c_str();
  auto tensor = op.GetInputDesc(0);
  Shape input_shape;
  if (WithRank(tensor, 0, input_shape, op_name) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Input must be 0-D");
    return GRAPH_FAILED;
  }
  TensorDesc y_desc = op.GetOutputDesc(0);
  Shape out = Shape({ge::UNKNOWN_DIM, ge::UNKNOWN_DIM, ge::UNKNOWN_DIM, 3});
  std::vector<std::pair<int64_t, int64_t>> y_range;
  (void)y_range.emplace_back(std::make_pair(1, -1));
  (void)y_range.emplace_back(std::make_pair(1, -1));
  (void)y_range.emplace_back(std::make_pair(1, -1));
  (void)y_range.emplace_back(std::make_pair(3, 3));
  y_desc.SetShape(Shape(out));
  y_desc.SetShapeRange(y_range);
  y_desc.SetDataType(DT_UINT8);
  if (op.UpdateOutputDesc("image", y_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name, "Fail to update output.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DecodeGif, DecodeGifInfer);

IMPLEMT_INFERFUNC(AdjustHue, AdjustHueInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto images_desc = op_desc->MutableInputDesc(0);

  GeShape out;
  if (WithRankAtLeast(images_desc, 3, out, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        0, DebugString(images_desc->GetShape().GetDims()), "at least 3D");
    err_msg = string("failed to call WithRankAtLeast function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<std::pair<int64_t, int64_t>> range;
  if (images_desc->GetShapeRange(range) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  auto y_desc = op_desc->MutableOutputDesc(0);
  y_desc->SetShape(out);
  y_desc->SetShapeRange(range);
  y_desc->SetDataType(images_desc->GetDataType());

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AdjustHue, AdjustHueInfer);

IMPLEMT_INFERFUNC(AdjustSaturation, AdjustSaturationInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto images_desc = op_desc->MutableInputDesc(0);

  GeShape out;
  if (WithRankAtLeast(images_desc, 3, out, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        0, DebugString(images_desc->GetShape().GetDims()), "at least 3D");
    err_msg = string("failed to call WithRankAtLeast function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<std::pair<int64_t, int64_t>> range;
  if (images_desc->GetShapeRange(range) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  auto y_desc = op_desc->MutableOutputDesc(0);
  y_desc->SetShape(out);
  y_desc->SetShapeRange(range);
  y_desc->SetDataType(images_desc->GetDataType());

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AdjustSaturation, AdjustSaturationInfer);

IMPLEMT_INFERFUNC(AdjustContrast, AdjustContrastInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  GeShape shape;
  std::string err_msg;
  auto contrast_factor_desc = op_desc->MutableInputDesc(1);
  if (WithRank(contrast_factor_desc, 0, shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(
        1, DebugString(contrast_factor_desc->GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  auto images_desc = op_desc->MutableInputDesc(0);
  if (WithRankAtLeast(images_desc, 3, shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(0, DebugString(images_desc->GetShape().GetDims()),
                             "at least 3D");
    err_msg = string("failed to call WithRankAtLeast function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_PARAM_INVALID;
  }

  auto y_desc = op_desc->MutableOutputDesc(0);
  y_desc->SetShape(shape);
  y_desc->SetDataType(images_desc->GetDataType());

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AdjustContrast, AdjustContrastInfer);

IMPLEMT_INFERFUNC(CropAndResize, CropAndResizeInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  // unknown shape support
  op_desc->SetOpInferDepends({"crop_size"});

  auto x_desc = op_desc->MutableInputDesc(0);
  const char* op_name = TbeGetName(op).c_str();
  GeShape x_shape;
  if (WithRank(x_desc, 4, x_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(x_desc->GetShape().GetDims()), "4D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
  }

  auto boxes_desc = op_desc->MutableInputDesc(1);
  GeShape boxes_shape;
  if (WithRank(boxes_desc, 2, boxes_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(boxes_desc->GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto box_index_desc = op_desc->MutableInputDesc(2);
  GeShape box_index_shape;
  if (WithRank(box_index_desc, 1, box_index_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2,
        DebugString(box_index_desc->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto crop_size_desc = op_desc->MutableInputDesc(3);
  GeShape crop_size_shape;
  if (WithRank(crop_size_desc, 1, crop_size_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(3,
        DebugString(crop_size_desc->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto x_dims = x_shape.GetDims();
  auto boxes_dims = boxes_shape.GetDims();
  auto box_index_dims = box_index_shape.GetDims();
  auto crop_size_dims = crop_size_shape.GetDims();

  CHECK(boxes_dims.empty() || box_index_dims.empty(),
        AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("the 0th input[x]'s shape and 1st input[boxes]'s shape"
                                           " should not be empty.")),
                                           return GRAPH_FAILED);
  if (boxes_dims[0] != UNKNOWN_DIM &&
      box_index_dims[0] != UNKNOWN_DIM &&
      boxes_dims[0] != box_index_dims[0]) {
      std::string err_msg = ConcatString(
          "the 0th dimension of the 1th input[boxes] and the 2nd input[box_index] must be equal. "
          , boxes_dims[0], " and " , box_index_dims[0]);
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  CHECK(crop_size_dims.empty(), AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), string("empty crop_size dim.")), return GRAPH_FAILED);
  if (crop_size_dims[0] != 2 && crop_size_dims[0] != UNKNOWN_DIM) {
      std::string err_msg = ConcatString(
          "the 3rd input[crop_size] must be a 1-D tensor containing 2 elements, current shape is ", DebugString(crop_size_dims));
    return GRAPH_FAILED;
  }

  int64_t crop_height = UNKNOWN_DIM;
  int64_t crop_width = UNKNOWN_DIM;
  Tensor crop_size_tensor;
  if (op.GetInputConstData("crop_size", crop_size_tensor) == GRAPH_SUCCESS) {
    auto size_data = reinterpret_cast<const int32_t*>(crop_size_tensor.GetData());
    crop_height = static_cast<int64_t>(size_data[0]);
    crop_width = static_cast<int64_t>(size_data[1]);
  }

  vector<int64_t> y_dims;
  Format input_format = op.GetInputDesc(0).GetFormat();
  if (input_format == FORMAT_NHWC && x_dims.size() > 3) {
    y_dims.push_back(boxes_dims[0]);
    y_dims.push_back(crop_height);
    y_dims.push_back(crop_width);
    y_dims.push_back(x_dims[3]);
  } else if (input_format == FORMAT_NCHW && x_dims.size() > 1) {
    y_dims.push_back(boxes_dims[0]);
    y_dims.push_back(x_dims[1]);
    y_dims.push_back(crop_height);
    y_dims.push_back(crop_width);
  } else {
    std::string str_input_format = ge::TypeUtils::FormatToSerialString(input_format);
    std::string err_msg = ConcatString(
          "only supporting NCHW and NHWC, current format is [", str_input_format, "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto y_desc = op_desc->MutableOutputDesc(0);
  GeShape y_shape(y_dims);
  if (!ShapeFullyDefined(y_shape)) {
    std::vector<std::pair<int64_t, int64_t>> y_range;
    for (const int64_t& y_dim : y_dims) {
      y_range.push_back(y_dim == UNKNOWN_DIM ? std::pair<int64_t, int64_t>{1, -1} :
                                               std::pair<int64_t, int64_t>{y_dim, y_dim});

    }
    y_desc->SetShapeRange(y_range);
  }
  y_desc->SetShape(y_shape);
  y_desc->SetDataType(boxes_desc->GetDataType());

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(CropAndResize, CropAndResizeInfer);

IMPLEMT_INFERFUNC(CropAndResizeGradBoxes, CropAndResizeGradBoxesInfer) {
  Shape shape;
  if (WithRank(op.GetInputDesc(0), 4, shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(op.GetInputDesc(0).GetShape().GetDims()), "4D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 4, shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(op.GetInputDesc(1).GetShape().GetDims()), "4D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(2), 2, shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2,
        DebugString(op.GetInputDesc(2).GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(3), 1, shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(3,
        DebugString(op.GetInputDesc(3).GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto grads_shape = op.GetInputDesc(0).GetShape().GetDims();
  auto boxes_shape = op.GetInputDesc(2).GetShape().GetDims();
  auto box_index_shape = op.GetInputDesc(3).GetShape().GetDims();

  if (grads_shape[0] != boxes_shape[0] && boxes_shape[0] != box_index_shape[0]) {
      std::string err_msg = ConcatString(
          "the 0th dimension of the 2th input[boxes], 0th input[grads] and the 3rd"
          " input [box_index] must be equal. ", grads_shape[0], ", " , boxes_shape[0] , " and " ,box_index_shape[0]);
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc desc = op.GetOutputDesc("y");
  desc.SetShape(op.GetInputDesc(2).GetShape());
  desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(CropAndResizeGradBoxes, CropAndResizeGradBoxesInfer);

IMPLEMT_INFERFUNC(CropAndResizeGradImage, CropAndResizeGradImageInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  // unknown shape support
  std::vector<std::string> input_infer_depends = {"image_size"};
  op_desc->SetOpInferDepends(input_infer_depends);

  auto grads_desc = op_desc->MutableInputDesc(0);
  GeShape grads_shape;
  const char* op_name = TbeGetName(op).c_str();
  if (WithRank(grads_desc, 4, grads_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(grads_desc->GetShape().GetDims()), "4D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto boxes_desc = op_desc->MutableInputDesc(1);
  GeShape boxes_shape;
  if (WithRank(boxes_desc, 2, boxes_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(boxes_desc->GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto box_index_desc = op_desc->MutableInputDesc(2);
  GeShape box_index_shape;
  if (WithRank(box_index_desc, 1, box_index_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2,
        DebugString(box_index_desc->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto image_size_desc = op_desc->MutableInputDesc(3);
  GeShape image_size_shape;
  if (WithRank(image_size_desc, 1, image_size_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(3,
        DebugString(image_size_desc->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto grads_dims = grads_shape.GetDims();
  auto boxes_dims = boxes_shape.GetDims();
  auto box_index_dims = box_index_shape.GetDims();
  CHECK(grads_dims.empty() || boxes_dims.empty() || box_index_dims.empty(),
        AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string(
        "the 0th input[grads] , the 1st input[boxes] dims and the 2nd input[box_index], "
        "must not be empty.")),
        return GRAPH_FAILED);
  if (!DimsAllEqualOrUnknown({grads_dims[0], boxes_dims[0], box_index_dims[0]})) {
      std::string err_msg = ConcatString(
                                         "the 0th dimension of the 0th input[grads], the 1st input[boxes]"
                                         " and the 2nd input[box_index] must be equal. "
                                         , grads_dims[0], ", " , boxes_dims[0], " and ", box_index_dims[0]);
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
  }

  auto image_size_dims = image_size_shape.GetDims();
  CHECK(image_size_dims.empty(), AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("the 3rd input[image_size] dims must not be empty.")),
        return GRAPH_FAILED);
  if (image_size_dims[0] != 4 && image_size_dims[0] != UNKNOWN_DIM) {
      std::string err_msg = ConcatString(
          "the 3rd input[image_size] must be a 1-D tensor with 4 elements, current image_size is ", DebugString(image_size_dims));
    return GRAPH_FAILED;
  }

  DataType type;
  if (op.GetAttr("T", type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("get attr[T] failed"));
    return GRAPH_FAILED;
  }

  int64_t batch = UNKNOWN_DIM;
  int64_t image_height = UNKNOWN_DIM;
  int64_t image_width = UNKNOWN_DIM;
  int64_t depth = UNKNOWN_DIM;
  Tensor image_size_tensor;
  if (op.GetInputConstData("image_size", image_size_tensor) == GRAPH_SUCCESS) {
    const int32_t* size_data = reinterpret_cast<const int32_t*>(image_size_tensor.GetData());
    CHECK(image_size_tensor.GetSize() / sizeof(int32_t) < 4,
          AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), string("the 3rd input[image_size]'s data nums less then 4, curent data num is ",
                                                                 image_size_tensor.GetSize() / sizeof(int32_t))),
          return GRAPH_FAILED);
    batch = static_cast<int64_t>(size_data[0]);
    image_height = static_cast<int64_t>(size_data[1]);
    image_width = static_cast<int64_t>(size_data[2]);
    depth = static_cast<int64_t>(size_data[3]);
  }

  std::vector<int64_t> y_dims;
  Format input_format = grads_desc->GetFormat();
  if (input_format == FORMAT_NHWC) {
    y_dims.push_back(batch);
    y_dims.push_back(image_height);
    y_dims.push_back(image_width);
    y_dims.push_back(depth);
  } else if (input_format == FORMAT_NCHW) {
    y_dims.push_back(batch);
    y_dims.push_back(depth);
    y_dims.push_back(image_height);
    y_dims.push_back(image_width);
  } else {
    std::string str_input_format = ge::TypeUtils::FormatToSerialString(input_format);
    std::string err_msg = ConcatString(
          "only supporting NCHW and NHWC, current format is [", str_input_format, "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto y_desc = op_desc->MutableOutputDesc(0);
  GeShape y_shape(y_dims);
  if (!ShapeFullyDefined(y_shape)) {
    std::vector<std::pair<int64_t, int64_t>> y_range;
    for (const int64_t& y_dim : y_dims) {
      y_range.push_back(y_dim == UNKNOWN_DIM ? std::pair<int64_t, int64_t>{1, -1} :
                                               std::pair<int64_t, int64_t>{y_dim, y_dim});

    }
    y_desc->SetShapeRange(y_range);
  }
  y_desc->SetShape(y_shape);
  y_desc->SetDataType(type);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(CropAndResizeGradImage, CropAndResizeGradImageInfer);

IMPLEMT_INFERFUNC(ExtractGlimpse, ExtractGlimpseInfer) {
  Shape x_shape;
  if (WithRank(op.GetInputDesc(0), 4, x_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "input x must be 4-D");
    return GRAPH_PARAM_INVALID;
  }
  Shape offsets_shape;
  if (WithRank(op.GetInputDesc(2), 2, offsets_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "input offsets must be 2-D");
    return GRAPH_PARAM_INVALID;
  }
  auto x_dims = op.GetInputDesc(0).GetShape().GetDims();
  auto offsets_dims = op.GetInputDesc(2).GetShape().GetDims();
  CHECK(x_dims.size() < 4 || offsets_dims.size() < 2, OP_LOGE(TbeGetName(op).c_str(), "invalid x_dims or offsets_dims."),
        return GRAPH_FAILED);
  int64_t batch_dim;
  if (Merge(x_dims[0], offsets_dims[0], batch_dim) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "x dim-0 or offsets dim-0 is invalid");
    return GRAPH_PARAM_INVALID;
  }
  if (offsets_dims[1] != 2) {
    OP_LOGE(TbeGetName(op).c_str(), "offsets dim-1 must be 2");
    return GRAPH_PARAM_INVALID;
  }

  bool uniform_noise = false;
  if (op.GetAttr("uniform_noise", uniform_noise) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "get attr uniform_noise failed");
    return GRAPH_FAILED;
  }
  std::string noise;
  if (op.GetAttr("noise", noise) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "get attr noise failed");
    return GRAPH_FAILED;
  }
  if (uniform_noise && (!noise.empty() && noise != "uniform")) {
    OP_LOGE(TbeGetName(op).c_str(), "The uniform_noise and noise should not be specified at the same time");
    return GRAPH_FAILED;
  }

  TensorDesc desc = op.GetOutputDesc("y");
  desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  auto channel_dim = x_dims[3];
  TensorDesc input_td = op.GetInputDesc(0);
  if (input_td.GetFormat() == FORMAT_NCHW) {
    channel_dim = x_dims[1];
  }
  return SetOutputToSizedImage(op, batch_dim, "size", channel_dim, "y");
}

INFER_FUNC_REG(ExtractGlimpse, ExtractGlimpseInfer);

IMPLEMT_INFERFUNC(HSVToRGB, HSVToRGBInfer) {
  TensorDesc desc = op.GetOutputDesc("y");
  desc.SetDataType(op.GetInputDesc(0).GetDataType());
  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return ColorspaceShapeFn(op, "y");
}

INFER_FUNC_REG(HSVToRGB, HSVToRGBInfer);

IMPLEMT_INFERFUNC(QuantizedResizeBilinear, QuantizedResizeBilinearInfer) {
  Shape min_shape;
  if (WithRank(op.GetInputDesc(2), 0, min_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "input min must be a scalar");
    return GRAPH_FAILED;
  }

  Shape max_shape;
  if (WithRank(op.GetInputDesc(3), 0, max_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "input max must be a scalar");
    return GRAPH_FAILED;
  }

  auto status = ResizeShapeFn(op, "images", "size", "resized_images");
  if (status != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "resize images shape failed");
    return GRAPH_FAILED;
  }

  TensorDesc y_min = op.GetOutputDesc("y_min");
  y_min.SetShape(Shape());
  y_min.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("y_min", y_min) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "fail to update output y_min.");
    return GRAPH_FAILED;
  }

  TensorDesc y_max = op.GetOutputDesc("y_max");
  y_max.SetShape(Shape());
  y_max.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("y_max", y_max) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "fail to update output y_max.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(QuantizedResizeBilinear, QuantizedResizeBilinearInfer);

IMPLEMT_INFERFUNC(ResizeArea, ResizeAreaInfer) {
  TensorDesc desc = op.GetOutputDesc("y");
  desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return ResizeShapeFn(op, "images", "size", "y");
}

INFER_FUNC_REG(ResizeArea, ResizeAreaInfer);

IMPLEMT_INFERFUNC(ResizeBicubicGrad, ResizeBicubicGradInfer) {
  TensorDesc desc = op.GetOutputDesc("y");
  Format input_format = op.GetInputDesc(0).GetFormat();
  vector<int64_t> grads_shape = op.GetInputDesc(0).GetShape().GetDims();
  vector<int64_t> org_images_shape = op.GetInputDesc(1).GetShape().GetDims();
  vector<int64_t> y_shape;
  if (input_format == FORMAT_NHWC && grads_shape.size() > 3
      && org_images_shape.size() > 2) {
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(org_images_shape[1]);
    y_shape.push_back(org_images_shape[2]);
    y_shape.push_back(grads_shape[3]);
  } else if (input_format == FORMAT_NCHW && grads_shape.size() > 1
             && org_images_shape.size() > 3) {
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(grads_shape[1]);
    y_shape.push_back(org_images_shape[2]);
    y_shape.push_back(org_images_shape[3]);
  } else {
    std::string str_input_format = ge::TypeUtils::FormatToSerialString(input_format);
    std::string err_msg = ConcatString(
        "only supporting NCHW and NHWC, current format is [", str_input_format, "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
  }
  desc.SetShape(ge::Shape(y_shape));
  auto type = op.GetInputDesc(1).GetDataType();
  desc.SetDataType(type);
  return op.UpdateOutputDesc("y", desc);
}

INFER_FUNC_REG(ResizeBicubicGrad, ResizeBicubicGradInfer);

IMPLEMT_INFERFUNC(ResizeBicubic, ResizeBicubicInfer) {
  TensorDesc desc = op.GetOutputDesc("y");
  desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return ResizeShapeFn(op, "images", "size", "y");
}

INFER_FUNC_REG(ResizeBicubic, ResizeBicubicInfer);

IMPLEMT_INFERFUNC(ResizeNearestNeighborV2Grad, ResizeNearestNeighborV2GradInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto y_desc = op_desc->MutableOutputDesc(0);
  auto size_desc = op_desc->MutableInputDesc(1);
  auto grads_desc = op_desc->MutableInputDesc(0);
  if (op.GetInputDesc(0).GetShape().GetDims() == UNKNOWN_RANK ||
      op.GetInputDesc(1).GetShape().GetDims() == UNKNOWN_RANK) {
    y_desc->SetShape(GeShape(UNKNOWN_RANK));
    y_desc->SetDataType(grads_desc->GetDataType());
    return GRAPH_SUCCESS;
  }
  // unknown shape support
  std::vector<std::string> input_infer_depends = {"size"};
  op_desc->SetOpInferDepends(input_infer_depends);

  const char* op_name = TbeGetName(op).c_str();
  GeShape grads_shape;
  if (WithRank(grads_desc, 4, grads_shape, op_name) != GRAPH_SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), "Input grads must be 4-D, real rank is [%lu]", grads_desc->GetShape().GetDimNum());
    return GRAPH_PARAM_INVALID;
  }

  GeShape size_shape;
  if (WithRank(size_desc, 1, size_shape, op_name) != GRAPH_SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), "Input size must be 1-D, real rank is [%lu]", size_desc->GetShape().GetDimNum());
    return GRAPH_PARAM_INVALID;
  }

  auto size_dims = size_shape.GetDims();
  if (size_dims[0] != 2 && size_dims[0] != UNKNOWN_DIM) {
    OP_LOGE(op_desc->GetName().c_str(), "Input size must be 1-D of 2 elements, real dim size is [%ld]", size_dims[0]);
    return GRAPH_PARAM_INVALID;
  }

  int64_t size_height = UNKNOWN_DIM;
  int64_t size_width = UNKNOWN_DIM;
  Tensor size_tensor;
  if (op.GetInputConstData("size", size_tensor) == GRAPH_SUCCESS) {
    auto size_data = reinterpret_cast<const int32_t*>(size_tensor.GetData());
    if (size_data == nullptr) {
      OP_LOGE(op_desc->GetName().c_str(), "Get size data failed");
      return GRAPH_PARAM_INVALID;
    }
    size_height = static_cast<int64_t>(size_data[0]);
    size_width = static_cast<int64_t>(size_data[1]);
  }

  std::vector<int64_t> output_dims;
  auto grads_dims = grads_shape.GetDims();
  auto input_format = grads_desc->GetFormat();
  if (input_format == FORMAT_NCHW) {
    output_dims.push_back(grads_dims[0]);
    output_dims.push_back(grads_dims[1]);
    output_dims.push_back(size_height);
    output_dims.push_back(size_width);
  } else if (input_format == FORMAT_NHWC) {
    output_dims.push_back(grads_dims[0]);
    output_dims.push_back(size_height);
    output_dims.push_back(size_width);
    output_dims.push_back(grads_dims[3]);
  } else {
    OP_LOGE(op_desc->GetName().c_str(), "Not supported this format: [%d]", input_format);
    return GRAPH_PARAM_INVALID;
  }
  GeShape output_shape(output_dims);
  if (ShapeFullyDefined(output_shape) == false) {
    std::vector<std::pair<int64_t, int64_t>> output_range;
    for (const int64_t& output_dim : output_dims) {
      output_range.push_back(output_dim == UNKNOWN_DIM ? std::pair<int64_t, int64_t>{1, -1} :
                                               std::pair<int64_t, int64_t>{output_dim, output_dim});

    }
    y_desc->SetShapeRange(output_range);
  }
  y_desc->SetDataType(grads_desc->GetDataType());
  y_desc->SetShape(output_shape);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ResizeNearestNeighborV2Grad, ResizeNearestNeighborV2GradInfer);

// ---------------ResizeNearestNeighborV2GradD Op Start-------------------
IMPLEMT_INFERFUNC(ResizeNearestNeighborV2GradD, ResizeNearestNeighborV2GradDInfer) {
  vector<int64_t> grads_shape = op.GetInputDesc("grads").GetShape().GetDims();
  vector<int64_t> size_out;
  if (op.GetAttr("size", size_out) == ge::GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("size");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (size_out.size() != DIM_SIZE2) {
    std::string err_msg = GetAttrSizeErrMsg("size_out", ConcatString(size_out.size()), ConcatString(DIM_SIZE2));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  Format input_format = op.GetInputDesc("grads").GetFormat();
  TensorDesc td = op.GetOutputDesc("y");
  vector<int64_t> y_shape;
  if (input_format == FORMAT_NHWC && grads_shape.size() > 3) {
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(size_out[0]);
    y_shape.push_back(size_out[1]);
    y_shape.push_back(grads_shape[3]);
  } else if (input_format == FORMAT_NCHW && grads_shape.size() > 1) {
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(grads_shape[1]);
    y_shape.push_back(size_out[0]);
    y_shape.push_back(size_out[1]);
  } else {
    string expected_format_list = ConcatString("FORMAT_NHWC, FORMAT_NHWC");
    std::string err_msg = GetInputFormatNotSupportErrMsg("input_format", expected_format_list, ConcatString(input_format));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  td.SetShape(Shape(y_shape));
  td.SetDataType(DT_FLOAT);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ResizeNearestNeighborV2GradD, ResizeNearestNeighborV2GradDInfer);

IMPLEMT_INFERFUNC(RGBToHSV, RGBToHSVInfer) {
  TensorDesc desc = op.GetOutputDesc("y");
  desc.SetDataType(op.GetInputDesc(0).GetDataType());
  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), std::string("update output[y] desc failed"));
    return GRAPH_FAILED;
  }
  return ColorspaceShapeFn(op, "y");
}

INFER_FUNC_REG(RGBToHSV, RGBToHSVInfer);

IMPLEMT_INFERFUNC(SampleDistortedBoundingBox, SampleDistortedBoundingBoxInfer) {
  bool judge = false;

  Shape image_size;
  judge = (WithRank(op.get_input_desc_image_size(), 1, image_size, TbeGetName(op).c_str()) != GRAPH_SUCCESS);
  if (judge) {
    std::string err_msg = ConcatString(
        "failed to call WithRank function, input[image_size] rank must be 1, "
        "got rank[", op.get_input_desc_image_size().GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  Shape bounding_boxes;
  judge = (WithRank(op.get_input_desc_bounding_boxes(), 3, bounding_boxes, TbeGetName(op).c_str()) != GRAPH_SUCCESS);
  if (judge) {
    std::string err_msg = ConcatString(
        "failed to call WithRank function, input[bounding_boxes] rank must be 3, "
        "got rank[", op.get_input_desc_bounding_boxes().GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t image_size_unused_dim;
  int64_t bounding_boxes_unused_dim2;
  const int64_t kImageSizeDimValue = image_size.GetDim(0);
  const int64_t kBoundingBoxesDim2Value = bounding_boxes.GetDim(2);
  if (WithValue(kImageSizeDimValue, 3, image_size_unused_dim, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call WithValue function, input[image_size] first "
        "dimention must be 3, got dim[", kImageSizeDimValue, "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (WithValue(kBoundingBoxesDim2Value, 4, bounding_boxes_unused_dim2, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call WithValue function, input[bounding_boxes] third "
        "dimention must be 4, got dim[", kBoundingBoxesDim2Value, "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc begin_desc = op.GetOutputDesc("begin");
  begin_desc.SetShape(Shape({3}));
  begin_desc.SetDataType(op.GetInputDesc("image_size").GetDataType());
  if (op.UpdateOutputDesc("begin", begin_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("fail to update output[begin] desc."));
    return GRAPH_FAILED;
  }

  TensorDesc size_desc = op.GetOutputDesc("size");
  size_desc.SetShape(Shape({3}));
  size_desc.SetDataType(op.GetInputDesc("image_size").GetDataType());
  if (op.UpdateOutputDesc("size", size_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("fail to update output[size] desc."));
    return GRAPH_FAILED;
  }

  TensorDesc bboxes_desc = op.GetOutputDesc("bboxes");
  bboxes_desc.SetShape(Shape({1, 1, 4}));
  bboxes_desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("bboxes", bboxes_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("fail to update output[bboxes] desc."));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SampleDistortedBoundingBox, SampleDistortedBoundingBoxInfer);

IMPLEMT_INFERFUNC(SampleDistortedBoundingBoxExt2, SampleDistortedBoundingBoxExt2Infer) {
  bool judge = false;

  Shape image_size;
  judge = (WithRank(op.get_input_desc_image_size(), 1, image_size, TbeGetName(op).c_str()) != GRAPH_SUCCESS);
  if (judge) {
    std::string err_msg = ConcatString(
        "failed to call WithRank function, input[image_size] rank must be 1 ,"
        "got rank[", op.get_input_desc_image_size().GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  Shape bounding_boxes;
  judge = (WithRank(op.get_input_desc_bounding_boxes(), 3, bounding_boxes, TbeGetName(op).c_str()) != GRAPH_SUCCESS);
  if (judge) {
    std::string err_msg = ConcatString(
        "failed to call WithRank function, input[bounding_boxes] rank must be 3 ,"
        "got rank[", op.get_input_desc_image_size().GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  Shape min_object_covered;
  judge =
      (WithRank(op.get_input_desc_min_object_covered(), 0, min_object_covered, TbeGetName(op).c_str()) != GRAPH_SUCCESS);
  if (judge) {
    std::string err_msg = ConcatString(
        "failed to call WithRank function, input[min_object_covered] rank must "
        "be scalar, got rank[",
        op.get_input_desc_image_size().GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  const int64_t image_size_dim_value = op.get_input_desc_image_size().GetShape().GetDim(0);
  const int64_t bounding_boxes_dim2_value = op.get_input_desc_bounding_boxes().GetShape().GetDim(2);
  if (((image_size_dim_value != 3) && (image_size_dim_value != -1)) ||
     ((bounding_boxes_dim2_value != 4) && (bounding_boxes_dim2_value != -1))) {
    std::string err_msg = ConcatString(
        "0th dim of input[image_size] must be 3 or -1, got[", image_size_dim_value,
        "] and 2nd dim of input[bounding_boxes] must be 4 or -1, got[",
        bounding_boxes_dim2_value, "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc begin_desc = op.GetOutputDesc("begin");
  begin_desc.SetShape(Shape({3}));
  begin_desc.SetDataType(op.GetInputDesc("image_size").GetDataType());
  if (op.UpdateOutputDesc("begin", begin_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("fail to update output[begin] desc."));
    return GRAPH_FAILED;
  }

  TensorDesc size_desc = op.GetOutputDesc("size");
  size_desc.SetShape(Shape({3}));
  size_desc.SetDataType(op.GetInputDesc("image_size").GetDataType());
  if (op.UpdateOutputDesc("size", size_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("fail to update output[size] desc."));
    return GRAPH_FAILED;
  }

  TensorDesc bboxes_desc = op.GetOutputDesc("bboxes");
  bboxes_desc.SetShape(Shape({1, 1, 4}));
  bboxes_desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("bboxes", bboxes_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("fail to update output[bboxes] desc."));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SampleDistortedBoundingBoxExt2, SampleDistortedBoundingBoxExt2Infer);

IMPLEMT_INFERFUNC(DrawBoundingBoxes, DrawBoundingBoxesInfer) {
  Shape images;

  if (WithRank(op.GetInputDesc(0), 4, images, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
        ConcatString("call WithRank function failed, ",
            GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "4D")));
    return GRAPH_FAILED;
  }

  Format input_format = op.GetInputDesc(0).GetFormat();
  int64_t depth = images.GetDim(3);
  if (input_format == FORMAT_NCHW) {
    depth = images.GetDim(1);
  }
  if (depth != ge::UNKNOWN_DIM) {
    if (!(depth == 1 || depth == 3 || depth == 4)) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
          ConcatString("invalid 3th dim[", depth, "] of input[images], should be 1, 3 or 4"));
      return GRAPH_FAILED;
    }
  }

  Shape boxes;
  if (WithRank(op.GetInputDesc(1), 3, boxes, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
        ConcatString("call WithRank function failed, ",
            GetShapeErrMsg(1, DebugString(op.GetInputDesc(1).GetShape().GetDims()), "3D")));
    return GRAPH_FAILED;
  }
  if ((boxes.GetDim(2) != 4) && (boxes.GetDim(2) != -1)) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        ConcatString("invalid 2th dim[", boxes.GetDim(2),
            "] of input[boxes], should 4 or -1"));
    return GRAPH_FAILED;
  }
  DataType type = op.GetInputDesc("images").GetDataType();
  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetDataType(type);
  y_desc.SetShape(images);
  return op.UpdateOutputDesc("y", y_desc);
}

INFER_FUNC_REG(DrawBoundingBoxes, DrawBoundingBoxesInfer);

IMPLEMT_INFERFUNC(NonMaxSuppression, NonMaxSuppressionInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  
  GeShape boxes_shape;
  const char* op_name = TbeGetName(op).c_str();
  auto boxes_desc = op_desc->MutableInputDesc(0);
  if (WithRank(boxes_desc, 2, boxes_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(boxes_desc->GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  GeShape scores_shape;
  auto scores_desc = op_desc->MutableInputDesc(1);
  if (WithRank(scores_desc, 1, scores_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(scores_desc->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  GeShape max_output_size_shape;
  auto max_output_size_desc = op_desc->MutableInputDesc(2);
  if (WithRank(max_output_size_desc, 0, max_output_size_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2,
        DebugString(max_output_size_desc->GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t unused_dim;
  if (Merge(boxes_shape.GetDim(0), scores_shape.GetDim(0), unused_dim) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call Merge function, 0th dim[",
        boxes_shape.GetDim(0), "] of input[boxes] not equal 0th dim[",
        scores_shape.GetDim(0), "] of input[scores]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (boxes_shape.GetDim(1) != 4 && boxes_shape.GetDim(1) != UNKNOWN_DIM) {
    std::string err_msg = ConcatString(
        "0th dim[", boxes_shape.GetDim(1), "] of input[boxes] not equal 4");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto selected_indices_desc = op_desc->MutableOutputDesc("selected_indices");
  selected_indices_desc->SetShape(GeShape({UNKNOWN_DIM}));
  selected_indices_desc->SetShapeRange({std::pair<int64_t, int64_t>(0, -1)});
  selected_indices_desc->SetDataType(DT_INT32);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(NonMaxSuppression, NonMaxSuppressionInfer);

IMPLEMT_INFERFUNC(NonMaxSuppressionV2, NonMaxSuppressionV2Infer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  GeShape boxes_shape;
  auto boxes_desc = op_desc->MutableInputDesc(0);
  const char* op_name = TbeGetName(op).c_str();
  if (WithRank(boxes_desc, 2, boxes_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(boxes_desc->GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  GeShape scores_shape;
  auto scores_desc = op_desc->MutableInputDesc(1);
  if (WithRank(scores_desc, 1, scores_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(scores_desc->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  GeShape max_output_size_shape;
  auto max_output_size_desc = op_desc->MutableInputDesc(2);
  if (WithRank(max_output_size_desc, 0, max_output_size_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2,
        DebugString(max_output_size_desc->GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  GeShape iou_threshold_shape;
  auto iou_threshold_desc = op_desc->MutableInputDesc(3);
  if (WithRank(iou_threshold_desc, 0, iou_threshold_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(3,
        DebugString(iou_threshold_desc->GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t unused_dim;
  if (Merge(boxes_shape.GetDim(0), scores_shape.GetDim(0), unused_dim) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call Merge function, 0th dim[",
        boxes_shape.GetDim(0), "] of input[boxes] not equal 0th dim[",
        scores_shape.GetDim(0), "] of input[scores]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (boxes_shape.GetDim(1) != 4 && boxes_shape.GetDim(1) != UNKNOWN_DIM) {
    std::string err_msg = ConcatString("1th dim[", boxes_shape.GetDim(1), "] of input[boxes] not equal 4.");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto selected_indices_desc = op_desc->MutableOutputDesc("selected_indices");
  selected_indices_desc->SetShape(GeShape({UNKNOWN_DIM}));
  selected_indices_desc->SetShapeRange({std::pair<int64_t, int64_t>(0, -1)});
  selected_indices_desc->SetDataType(DT_INT32);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(NonMaxSuppressionV2, NonMaxSuppressionV2Infer);

IMPLEMT_INFERFUNC(NonMaxSuppressionV3, NonMaxSuppressionV3Infer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  GeShape boxes_shape;
  auto boxes_desc = op_desc->MutableInputDesc(0);
  const char* op_name = TbeGetName(op).c_str();
  if (WithRank(boxes_desc, 2, boxes_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(boxes_desc->GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  GeShape scores_shape;
  auto scores_desc = op_desc->MutableInputDesc(1);
  if (WithRank(scores_desc, 1, scores_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(scores_desc->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  GeShape max_output_size_shape;
  auto max_output_size_desc = op_desc->MutableInputDesc(2);
  if (WithRank(max_output_size_desc, 0, max_output_size_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        2,
        DebugString(max_output_size_desc->GetShape().GetDims()),
        "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  GeShape iou_threshold_shape;
  auto iou_threshold_desc = op_desc->MutableInputDesc(3);
  if (WithRank(iou_threshold_desc, 0, iou_threshold_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        3,
        DebugString(iou_threshold_desc->GetShape().GetDims()),
        "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  GeShape score_threshold_shape;
  auto score_threshold_desc = op_desc->MutableInputDesc(4);
  if (WithRank(score_threshold_desc, 0, score_threshold_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(4,
        DebugString(score_threshold_desc->GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t unused_dim;
  if (Merge(boxes_shape.GetDim(0), scores_shape.GetDim(0), unused_dim) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call Merge function, 0th dim[",
        boxes_shape.GetDim(0), "] of input[boxes] not equal 0th dim[",
        scores_shape.GetDim(0), "] of input[scores]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (boxes_shape.GetDim(1) != 4 && boxes_shape.GetDim(1) != UNKNOWN_DIM) {
    std::string err_msg = ConcatString(
        "1th dim[", boxes_shape.GetDim(1), "] of input[boxes] not equal 4.");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  auto selected_indices_desc = op_desc->MutableOutputDesc(0);
  selected_indices_desc->SetDataType(DT_INT32);
  selected_indices_desc->SetShape(GeShape({UNKNOWN_DIM}));
  selected_indices_desc->SetShapeRange({std::pair<int64_t, int64_t>(0, -1)});

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(NonMaxSuppressionV3, NonMaxSuppressionV3Infer);

IMPLEMT_INFERFUNC(NonMaxSuppressionV4, NonMaxSuppressionV4Infer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  const char *op_name = TbeGetName(op).c_str();
  // unknwon shape support
  std::vector<std::string> input_infer_depends = {"max_output_size"};
  op_desc->SetOpInferDepends(input_infer_depends);

  GeShape boxes_shape;
  auto boxes_desc = op_desc->MutableInputDesc(0);
  if (WithRank(boxes_desc, 2, boxes_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, 
                                         DebugString(boxes_desc->GetShape().GetDims()),
                                         "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }


  GeShape scores_shape;
  auto scores_desc = op_desc->MutableInputDesc(1);
  if (WithRank(scores_desc, 1, scores_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
                                         DebugString(scores_desc->GetShape().GetDims()), 
                                         "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  GeShape max_output_size_shape;
  auto max_output_size_desc = op_desc->MutableInputDesc(2);
  if (WithRank(max_output_size_desc, 0, max_output_size_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2, 
                                         DebugString(max_output_size_desc->GetShape().GetDims()),
                                         "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  GeShape iou_threshold_shape;
  auto iou_threshold_desc = op_desc->MutableInputDesc(3);
  if (WithRank(iou_threshold_desc, 0, iou_threshold_shape, op_name) != GRAPH_SUCCESS) {
     std::string err_msg = GetShapeErrMsg(3,
                                          DebugString(iou_threshold_desc->GetShape().GetDims()), 
                                          "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  GeShape score_threshold_shape;
  auto score_threshold_desc = op_desc->MutableInputDesc(4);
  if (WithRank(score_threshold_desc, 0, score_threshold_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(4, 
                                         DebugString(score_threshold_desc->GetShape().GetDims()), 
                                         "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t unused_dim;
  if (Merge(boxes_shape.GetDim(0), scores_shape.GetDim(0), unused_dim) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call Merge function, 0th dim[",
                                       boxes_shape.GetDim(0), "] of input[boxes] not equal 0th dim[",
                                       scores_shape.GetDim(0), "] of input[scores]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithValue(boxes_shape.GetDim(1), 4, unused_dim, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call WithValue function, 1th dim[",
                                       boxes_shape.GetDim(1), "] of input[boxes] not equal 4");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> selected_indices_dims{UNKNOWN_DIM};
  bool pad_to_max = false;
  if (op.GetAttr("pad_to_max_output_size", pad_to_max) != ge::GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("get attr[pad_to_max_output_size] failed"));
    return GRAPH_FAILED;
  }
  if (pad_to_max) {
    Tensor selected_indices_tensor;
    if (op.GetInputConstData("max_output_size", selected_indices_tensor) == GRAPH_SUCCESS) {
      const int32_t *selected_indices_data =
          reinterpret_cast<const int32_t*>(selected_indices_tensor.GetData());
      int32_t selected_indices_data_0 = *selected_indices_data;
      if (selected_indices_data_0 < 0) {
        std::string err_msg = ConcatString("0th data[", selected_indices_data_0,
                                           "] of input[max_output_size] at least 0.");
        AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
      }
      selected_indices_dims[0] = selected_indices_data_0;
    }
  }
  auto selected_indices_desc = op_desc->MutableOutputDesc("selected_indices");
  (void)FillOpDesc(selected_indices_desc, GeShape(selected_indices_dims), ge::DT_INT32);

  auto valid_outputs_desc = op_desc->MutableOutputDesc("valid_outputs");
  (void)FillOpDesc(valid_outputs_desc, GeShape(), ge::DT_INT32);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(NonMaxSuppressionV4, NonMaxSuppressionV4Infer);

IMPLEMT_INFERFUNC(NonMaxSuppressionWithOverlaps, NonMaxSuppressionWithOverlapsInfer) {
  Shape overlaps_shape = op.GetInputDesc("overlaps").GetShape();
  Shape scores_shape = op.GetInputDesc("scores").GetShape();
  Shape max_output_size_shape = op.GetInputDesc("max_output_size").GetShape();
  Shape overlap_threshold_shape = op.GetInputDesc("overlap_threshold").GetShape();
  Shape score_threshold_shape = op.GetInputDesc("score_threshold").GetShape();
  if (WithRank(op.GetInputDesc("overlaps"), 2, overlaps_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call WithRank function, ",
      "input[overlaps] rank must be 2, but got rank[",
      op.GetInputDesc("overlaps").GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc("scores"), 1, scores_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call WithRank function, ",
      "input[scores] rank must be 1, but got rank[",
      op.GetInputDesc("scores").GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc("max_output_size"), 0, max_output_size_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call WithRank function, ",
      "input[max_output_size] rank must be 0, but got rank[",
      op.GetInputDesc("max_output_size").GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc("overlap_threshold"), 0, overlap_threshold_shape, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call WithRank function, ",
      "input[overlap_threshold] rank must be 0, but got rank[",
      op.GetInputDesc("overlap_threshold").GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc("score_threshold"), 0, score_threshold_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call WithRank function, ",
      "input[score_threshold] rank must be 0, but got rank[",
      op.GetInputDesc("score_threshold").GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  int64_t unused_dim = 0;
  if (Merge(overlaps_shape.GetDim(0), scores_shape.GetDim(0), unused_dim) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call Merge function to merge the input[overlaps] 0th dim",
        "[" , overlaps_shape.GetDim(0), "] and the input[scores]'s 0th dim [", 
        scores_shape.GetDim(0), "]");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (Merge(overlaps_shape.GetDim(0), overlaps_shape.GetDim(1), unused_dim) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call Merge function to merge the input[overlaps] 0th dim",
        "[" , overlaps_shape.GetDim(0), "] and the input[overlaps]'s 1th dim [", 
        overlaps_shape.GetDim(1), "]");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc selected_indices_desc = op.GetOutputDesc("selected_indices");
  Shape selecte_indices_shape;
  Vector(ge::UNKNOWN_DIM, selecte_indices_shape);
  selected_indices_desc.SetDataType(DT_INT32);
  selected_indices_desc.SetShape(selecte_indices_shape);
  if (op.UpdateOutputDesc("selected_indices", selected_indices_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
      std::string("update output[selected_indices] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(NonMaxSuppressionWithOverlaps, NonMaxSuppressionWithOverlapsInfer);

IMPLEMT_INFERFUNC(EncodePng, EncodePngInfer) {
  return EncodeImageShapeFn(op);
}

INFER_FUNC_REG(EncodePng, EncodePngInfer);



IMPLEMT_INFERFUNC(DecodePng, DecodePngInfer) {
  return DecodeImageShapeFn(op);
}

INFER_FUNC_REG(DecodePng, DecodePngInfer);

IMPLEMT_INFERFUNC(DecodeBmp, DecodeBmpInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc == nullptr) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("get op desc failed, op desc is nullptr."));
    return GRAPH_FAILED;
  }
  TensorDesc contents = op.GetInputDesc(0);
  TensorDesc image = op.GetOutputDesc(0);
  DataType input_data = contents.GetDataType();
  std::string err_msg;
  if (input_data != DT_STRING) {
    std::string input_dt = DTypeStr(input_data);
    err_msg = ConcatString(
        "input[contents] data type must be string, data type[", input_dt, "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t channels;
  if (op.GetAttr("channels", channels) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("get attr[channels] failed."));
    return GRAPH_FAILED;
  }
  if (channels != 0 && channels != 1 && channels != 3 && channels != 4) {
    err_msg =
        ConcatString("attr[channels] must be 0, 1, 3, 4, got [", channels, "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (channels == 0) {
    channels = UNKNOWN_DIM;
    OP_LOGI(TbeGetName(op).c_str(), "attr[channels] is 0, use unknowdim");
  }
  image.SetDataType(DT_UINT8);
  std::vector<int64_t> image_shape({UNKNOWN_DIM, UNKNOWN_DIM, channels});
  image.SetShape(Shape(image_shape));
  if (op.UpdateOutputDesc("image", image) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("fail to update output[image]."));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(DecodeBmp, DecodeBmpInfer);

IMPLEMT_INFERFUNC(DecodeAndCropJpeg, DecodeAndCropJpegInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc == nullptr) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("get op desc failed, op desc is nullptr."));
    return GRAPH_FAILED;
  }

  // unknown shape support
  op_desc->SetOpInferDepends({"crop_window"});

  GeShape contents_shape;
  auto contents_desc = op_desc->MutableInputDesc(0);
  if (contents_desc == nullptr) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("get input[contents] desc failed, input[contents] "
                             "desc is nullptr."));
    return GRAPH_FAILED;
  }
  std::string err_msg;
  if (WithRank(contents_desc, 0, contents_shape, TbeGetName(op).c_str())
      != GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[contents] rank must be 0, got "
        "rank[",
        contents_desc->GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_PARAM_INVALID;
  }

  int64_t channels_dim = UNKNOWN_DIM;
  int64_t height = UNKNOWN_DIM;
  int64_t width = UNKNOWN_DIM;

  int32_t channels;
  if (op.GetAttr("channels", channels) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
                                       string("failed to get attr[channels]."));
    return GRAPH_PARAM_INVALID;
  }
  if (channels != 0) {
    if (channels < 0) {
      err_msg = ConcatString("attr[channels] must be non-negative, got[",
                             channels, "]");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_PARAM_INVALID;
    }
    channels_dim = channels;
  }

  GeShape crop_window_shape;
  auto crop_window_desc = op_desc->MutableInputDesc(1);
  if (crop_window_desc == nullptr) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), string("get input[crop_window] desc failed, "
                             "input[crop_window] desc is nullptr."));
    return GRAPH_FAILED;
  }
  if (WithRank(crop_window_desc, 1, crop_window_shape, TbeGetName(op).c_str())
      != GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[crop_window] rank must be 1, "
        "got rank[",
        crop_window_desc->GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_PARAM_INVALID;
  }
  int64_t unused_dim;
  if (WithValue(crop_window_shape.GetDim(0), 4, unused_dim,
                op_desc->GetName().c_str()) != GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithValue function, dim[0] of input[crop_window] must "
        "be 4, got[",
        crop_window_shape.GetDim(0), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_PARAM_INVALID;
  }

  Tensor crop_window_tensor;
  if (op.GetInputConstData("crop_window", crop_window_tensor) ==
      GRAPH_SUCCESS) {
    const int32_t* crop_window_data =
        reinterpret_cast<int32_t*>(crop_window_tensor.GetData());
    height = *(crop_window_data + 2);
    width = *(crop_window_data + 3);
  }

  auto image_desc = op_desc->MutableOutputDesc("image");
  (void)FillOpDesc(image_desc, GeShape({height, width, channels_dim}),
                   DT_UINT8);

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(DecodeAndCropJpeg, DecodeAndCropJpegInfer);

bool ResizeConstInferShape(const Operator& op, const std::pair<uint32_t, std::string> image_info,
                           const std::pair<uint32_t, std::string> size_info,
                           const std::pair<uint32_t, std::string> output_info) {
  static const size_t output_len = 4;
  static const size_t size_len = 2;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_desc == nullptr, OP_LOGE(TbeGetName(op).c_str(), "op desc is null."), return false);

  auto input_desc_x = op_desc->MutableInputDesc(image_info.first);
  CHECK(input_desc_x == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), OtherErrMsg("input x is null.")), return false);
  auto output_desc_y = op_desc->MutableOutputDesc(output_info.first);
  CHECK(output_desc_y == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), OtherErrMsg("output y is null.")), return false);

  // infer dtype start
  output_desc_y->SetDataType(input_desc_x->GetDataType());
  // infer dtype end

  // infer shape start
  const GeShape& x_shape = input_desc_x->MutableShape();
  auto input_format = input_desc_x->GetFormat();
  OP_LOGD(TbeGetName(op).c_str(), "get the format is %s", TypeUtils::FormatToSerialString(input_format).c_str());
  CHECK(input_format != FORMAT_NHWC && input_format != FORMAT_NCHW,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), OtherErrMsg("The input format is valid")),
        return false);
  const int64_t image_n_idx = 0;
  // format is NHWC, c_idx = 3, format is NCHW, c_idx = 1,
  const int64_t image_c_idx = input_format == FORMAT_NHWC ? 3 : 1;
  const int64_t image_h_idx = input_format == FORMAT_NHWC ? 1 : 2;
  const int64_t image_w_idx = input_format == FORMAT_NHWC ? 2 : 3;
  // get const value
  bool is_size_const = true;
  vector<int64_t> size_out;
  if (!ops::GetConstIntData(op, size_info.first, size_out)) {
    OP_LOGW(TbeGetName(op).c_str(), "get const value of input size failed, set out hw = -1, -1");
    size_out = {-1, -1};
    is_size_const = false;
  }

  // the size num must be 2, mean output h, output w
  OP_LOGD(TbeGetName(op).c_str(), "the size num must be 2. get the num is %zu", size_out.size());
  CHECK(size_out.size() != size_len,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), OtherErrMsg("the input size num must be 2.")),
        return false);

  // get y shape
  GeShape& y_shape = output_desc_y->MutableShape();
  y_shape.SetDimNum(output_len);
  if (!x_shape.IsUnknownDimNum()) {
    OP_LOGD(TbeGetName(op).c_str(), "the input shape size must be 4. get shape size is %zu", x_shape.GetDimNum());
    CHECK(x_shape.GetDimNum() != output_len,
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), OtherErrMsg("The dim of input x is not 4")),
          return false);
    y_shape.SetDim(image_n_idx, x_shape.GetDim(image_n_idx));
    y_shape.SetDim(image_c_idx, x_shape.GetDim(image_c_idx));
  } else {
    OP_LOGW(TbeGetName(op).c_str(), "the input is unkown rank, will set the out nc = -1, -1");
    y_shape.SetDim(image_n_idx, -1);
    y_shape.SetDim(image_c_idx, -1);
  }
  y_shape.SetDim(image_h_idx, size_out[0]);
  y_shape.SetDim(image_w_idx, size_out[1]);
  // infer shape end

  // charge whether is dynamic, when output is static shape, return true
  CHECK(!y_shape.IsUnknownShape(), OP_LOGD(TbeGetName(op).c_str(), "the output is static shape. infer succ"),
        return true);

  OP_LOGD(TbeGetName(op).c_str(), "the output is dynamic shape. will infer range");
  // infer shape_range start
  std::vector<std::pair<int64_t, int64_t>> x_range;
  vector<int64_t> image_shape{-1, -1, -1, -1};
  // check whether is -2 case
  if (!x_shape.IsUnknownDimNum()) {
    image_shape = x_shape.GetDims();
    (void)input_desc_x->GetShapeRange(x_range);
  }
  MakeUpShapeRange(image_shape, x_range);
  OP_LOGD(TbeGetName(op).c_str(), "the input range size must be 4. get size is %zu", x_range.size());
  CHECK(x_range.size() != output_len,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), OtherErrMsg("the x range size is not equal 4")),
        return false);
  if (!is_size_const) {
    std::vector<std::pair<int64_t, int64_t>> size_value_range;
    auto input_size_x = op_desc->MutableInputDesc(size_info.first);
    CHECK(input_size_x == nullptr,
          VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), OtherErrMsg("input size is null.")),
          return false);
    // means no const value, will get the value range
    (void)input_size_x->GetValueRange(size_value_range);
    // the size num must be 2, so the value range num must be 2
    if (size_value_range.size() != size_len) {
      x_range[image_h_idx] = std::pair<int64_t, int64_t>(0, -1);
      x_range[image_w_idx] = std::pair<int64_t, int64_t>(0, -1);
    } else {
      x_range[image_h_idx] = size_value_range[0];
      x_range[image_w_idx] = size_value_range[1];
    }
  } else {
    x_range[image_h_idx] = std::pair<int64_t, int64_t>(size_out[0], size_out[0]);
    x_range[image_w_idx] = std::pair<int64_t, int64_t>(size_out[1], size_out[1]);
  }

  output_desc_y->SetShapeRange(x_range);
  // infer shape_range end
  return true;
}

IMPLEMT_COMMON_INFERFUNC(ResizeInferShape) {
  vector<int64_t> images_shape = op.GetInputDesc("x").GetShape().GetDims();
  vector<int64_t> size_out;
  if (op.GetAttr("size", size_out) == ge::GRAPH_FAILED) {
    OP_LOGE(TbeGetName(op).c_str(), "GetOpAttr ConstValue size failed!");
    return GRAPH_FAILED;
  }

  if (size_out.size() != DIM_SIZE2) {
    OP_LOGE(TbeGetName(op).c_str(), "length of size_out must be equal to 2");
    return GRAPH_FAILED;
  }
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  Format input_format = op.GetInputDesc("x").GetFormat();
  TensorDesc td = op.GetOutputDesc("y");
  vector<int64_t> y_shape;
  if (input_format == FORMAT_NHWC && images_shape.size() > 3) {
    y_shape.push_back(images_shape[0]);
    y_shape.push_back(size_out[0]);
    y_shape.push_back(size_out[1]);
    y_shape.push_back(images_shape[3]);
  } else if (input_format == FORMAT_NCHW && images_shape.size() > 1) {
    y_shape.push_back(images_shape[0]);
    y_shape.push_back(images_shape[1]);
    y_shape.push_back(size_out[0]);
    y_shape.push_back(size_out[1]);
  } else {
    OP_LOGE(TbeGetName(op).c_str(), "Not supported this format");
  }
  td.SetShape(ge::Shape(y_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

bool SyncResizeInferShape(const Operator& op) {
  vector<int64_t> size_out;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc_x = op_desc->MutableInputDesc("x");
  auto output_desc_y = op_desc->MutableOutputDesc("y");
  auto image_shape = input_desc_x->MutableShape().GetDims();
  auto input_format = input_desc_x->GetFormat();

  if (op.GetAttr("split_size", size_out) == ge::GRAPH_FAILED) {
    OP_LOGE(TbeGetName(op).c_str(), "GetOpAttr split_size failed!");
    return false;
  }

  if (size_out.size() != DIM_SIZE2) {
    OP_LOGE(TbeGetName(op).c_str(), "length of size_out must be equal to 2");
    return false;
  }

  std::vector<std::pair<int64_t, int64_t>> x_range;
  // check whether is -2 case
  bool is_unkown_rank = image_shape == UNKNOWN_RANK ? true : false;
  if (is_unkown_rank) {
    OP_LOGW(TbeGetName(op).c_str(), "the input os unkown rank, will set the input -1, -1, -1 , -1");
    image_shape = {-1, -1, -1, -1};
  } else {
    input_desc_x->GetShapeRange(x_range);
  }
  MakeUpShapeRange(image_shape, x_range);

  std::vector<std::pair<int64_t, int64_t>> output_range;
  output_range.push_back(std::pair<int64_t, int64_t>{size_out[0], size_out[0]});
  output_range.push_back(std::pair<int64_t, int64_t>{size_out[1], size_out[1]});
  // get input shape range
  std::vector<std::pair<int64_t, int64_t>> result_range;

  vector<int64_t> y_shape;
  if (input_format == FORMAT_NHWC && image_shape.size() > 3) {
    y_shape.push_back(image_shape[0]);
    y_shape.push_back(size_out[0]);
    y_shape.push_back(size_out[1]);
    y_shape.push_back(image_shape[3]);
    result_range.push_back(x_range[0]);
    result_range.push_back(output_range[0]);
    result_range.push_back(output_range[1]);
    result_range.push_back(x_range[3]);
  } else if (input_format == FORMAT_NCHW && image_shape.size() > 1) {
    y_shape.push_back(image_shape[0]);
    y_shape.push_back(image_shape[1]);
    y_shape.push_back(size_out[0]);
    y_shape.push_back(size_out[1]);
    result_range.push_back(x_range[0]);
    result_range.push_back(x_range[1]);
    result_range.push_back(output_range[0]);
    result_range.push_back(output_range[1]);
  } else {
    OP_LOGE(TbeGetName(op).c_str(), "Not supported this format %d", input_format);
    return false;
  }

  output_desc_y->SetShape(GeShape(y_shape));
  output_desc_y->SetOriginShape(GeShape(y_shape));
  auto input_dtype = input_desc_x->GetDataType();
  output_desc_y->SetDataType(input_dtype);
  output_desc_y->SetShapeRange(result_range);
  return true;
}

// ---------------ResizeBilinearV2 Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(ResizeBilinearV2InferShape) {
  static const std::pair<uint32_t, std::string> input_x{0, "x"};
  static const std::pair<uint32_t, std::string> input_size{1, "size"};
  static const std::pair<uint32_t, std::string> output_y{0, "y"};
  const vector<string> depends{input_size.second};
  PREPARE_DYNAMIC_SHAPE(depends);
  if (!ResizeConstInferShape(op, input_x, input_size, output_y)) {
    return GRAPH_FAILED;
  }

  auto op_desc_info = OpDescUtils::GetOpDescFromOperator(op);
  auto output_desc_y = op_desc_info->MutableOutputDesc(output_y.first);

  DataType attr_dtype;
  CHECK(op.GetAttr("dtype", attr_dtype) != GRAPH_SUCCESS,
        AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), string("Get attr[dtype] failed.")),
        return GRAPH_FAILED);
  output_desc_y->SetDataType(attr_dtype);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ResizeBilinearV2, ResizeBilinearV2InferShape);
INFER_VALUE_RANGE_DEFAULT_REG(ResizeBilinearV2);
// ---------------ResizeBilinearV2 Op End-------------------

// ---------------SyncResizeBilinearV2 Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(SyncResizeBilinearV2InferShape) {
  vector<int64_t> split_size;
  op.GetAttr("split_size", split_size);
    if (!SyncResizeInferShape(op)) {
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SyncResizeBilinearV2, SyncResizeBilinearV2InferShape);
// ---------------SyncResizeBilinearV2 Op End-------------------

// ---------------ResizeBilinearV2D Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(ResizeBilinearV2DInferShape) {

  vector<int64_t> images_shape = op.GetInputDesc("x").GetShape().GetDims();
  vector<int64_t> size_out;
  if (op.GetAttr("size", size_out) == ge::GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("size");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (size_out.size() != DIM_SIZE2) {
    std::string err_msg = GetAttrSizeErrMsg("size_out", ConcatString(size_out.size()), ConcatString(DIM_SIZE2));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  Format input_format = op.GetInputDesc("x").GetFormat();
  TensorDesc td = op.GetOutputDesc("y");
  vector<int64_t> y_shape;
  if (input_format == FORMAT_NHWC && images_shape.size() > 3) {
    y_shape.push_back(images_shape[0]);
    y_shape.push_back(size_out[0]);
    y_shape.push_back(size_out[1]);
    y_shape.push_back(images_shape[3]);
  } else if (input_format == FORMAT_NCHW && images_shape.size() > 1) {
    y_shape.push_back(images_shape[0]);
    y_shape.push_back(images_shape[1]);
    y_shape.push_back(size_out[0]);
    y_shape.push_back(size_out[1]);
  } else {
    string expected_format_list = ConcatString("FORMAT_NHWC, FORMAT_NCHW");
    std::string err_msg = GetInputFormatNotSupportErrMsg("input_format", expected_format_list, ConcatString(input_format));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
  }
  td.SetShape(ge::Shape(y_shape));
  td.SetDataType(DT_FLOAT);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ResizeBilinearV2D, ResizeBilinearV2DInferShape);
// ---------------ResizeBilinearV2D Op End-------------------

// ----------------------KeepRatioResizeBilinear----------------------
IMPLEMT_VERIFIER(KeepRatioResizeBilinear, KeepRatioResizeBilinearVerify) {
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(KeepRatioResizeBilinearInferShape) {
  std::int64_t minDims = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("min_dimension", minDims)) {
    OP_LOGE(TbeGetName(op).c_str(), "get attr min_dimension failed");
    return GRAPH_FAILED;
  }
  std::int64_t maxDims = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("max_dimension", maxDims)) {
    OP_LOGE(TbeGetName(op).c_str(), "get attr max_dimension failed");
    return GRAPH_FAILED;
  }
  CHECK(minDims == 0 || maxDims == 0, OP_LOGE(TbeGetName(op).c_str(), "min_dimension and max_dimension should not be 0."),
        return GRAPH_FAILED);
  float minDimsFloat = static_cast<float>(minDims);
  float maxDimsFloat = static_cast<float>(maxDims);
  std::int64_t batchDIms = 0;
  std::int64_t heightDIms = 0;
  std::int64_t widthDims = 0;
  std::int64_t channelDIms = 0;
  auto inputImagesShape = op.GetInputDesc("images").GetShape().GetDims();

  if (inputImagesShape.size() != DIM_SIZE4) {
    OP_LOGE(TbeGetName(op).c_str(), "length of size_out must be equal to 4");
    return GRAPH_FAILED;
  }

  Format inputFormat = op.GetInputDesc("images").GetFormat();
  if (inputFormat == FORMAT_NHWC) {
    batchDIms = inputImagesShape[0];
    heightDIms = inputImagesShape[1];
    widthDims = inputImagesShape[2];
    channelDIms = inputImagesShape[3];
  } else if (inputFormat == FORMAT_NCHW) {
    batchDIms = inputImagesShape[0];
    heightDIms = inputImagesShape[2];
    widthDims = inputImagesShape[3];
    channelDIms = inputImagesShape[1];
  } else {
    string expectedFormatList = ConcatString("FORMAT_NHWC, FORMAT_NCHW");
    OP_LOGE(TbeGetName(op).c_str(), "Not supported this format");
    return GRAPH_FAILED;
  }
  std::int64_t minShapeDims = std::min(heightDIms, widthDims);
  std::int64_t maxShapeDims = std::max(heightDIms, widthDims);
  float minShapeDimsFloat = static_cast<float>(minShapeDims);
  float maxShapeDimsFloat = static_cast<float>(maxShapeDims);

  // get min scale
  float resizeScale = minShapeDimsFloat / minDimsFloat;
  float minNewShapeH = floor((heightDIms / resizeScale) + 0.5);
  float minNewShapeW = floor((widthDims / resizeScale) + 0.5);
  float minNewShapeMaxDim = std::max(minNewShapeH, minNewShapeW);

  // get max scale
  resizeScale = maxShapeDimsFloat / maxDimsFloat;
  float maxNewShapeH = floor((heightDIms / resizeScale) + 0.5);
  float maxNewShapeW = floor((widthDims / resizeScale) + 0.5);

  vector<int64_t> outputShapeVec;
  if (minNewShapeMaxDim > maxDimsFloat) {
    outputShapeVec.push_back(static_cast<int64_t>(maxNewShapeH));
    outputShapeVec.push_back(static_cast<int64_t>(maxNewShapeW));
  } else {
    outputShapeVec.push_back(static_cast<int64_t>(minNewShapeH));
    outputShapeVec.push_back(static_cast<int64_t>(minNewShapeW));
  }

  vector<int64_t> yShape;
  if (inputFormat == FORMAT_NHWC) {
    yShape.push_back(batchDIms);
    yShape.push_back(outputShapeVec[0]);
    yShape.push_back(outputShapeVec[1]);
    yShape.push_back(channelDIms);
  } else {
    yShape.push_back(batchDIms);
    yShape.push_back(channelDIms);
    yShape.push_back(outputShapeVec[0]);
    yShape.push_back(outputShapeVec[1]);
  }

  TensorDesc td = op.GetOutputDesc("y");
  td.SetShape(ge::Shape(yShape));
  td.SetDataType(DT_FLOAT);
  (void)op.UpdateOutputDesc("y", td);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(KeepRatioResizeBilinear, KeepRatioResizeBilinearInferShape);
VERIFY_FUNC_REG(KeepRatioResizeBilinear, KeepRatioResizeBilinearVerify);
// ----------------------KeepRatioResizeBilinear END----------------------

// ---------------ResizeD Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(ResizeDInferShape) {
    TensorDesc output_desc = op.GetOutputDesc("y");
    DataType input_dtype = op.GetInputDesc("x").GetDataType();
    Format input_format = op.GetInputDesc("x").GetFormat();
    Shape input_shape = op.GetInputDesc("x").GetShape();
    std::vector<int64_t> input_dims = input_shape.GetDims();
    std::vector<int64_t> sizes;
    std::string mode = "nearest";
    op.GetAttr("sizes", sizes);
    op.GetAttr("mode", mode);
    if (mode == "cubic") {
        if (input_dims.size() != 4) {
            return GRAPH_FAILED;
        }

        if (sizes.size() != 2) {
            return GRAPH_FAILED;
        }

        std::vector<int64_t> dim_vec;
        for (size_t i = 0; i < 2; i++) {
            int64_t dims = input_dims[i];
            dim_vec.push_back(dims);
        }
        for (size_t i = 0; i < 2; i++) {
            int64_t dims = sizes[i];
            dim_vec.push_back(dims);
        }

        Shape output_shape = Shape(dim_vec);
        output_desc.SetShape(output_shape);
        output_desc.SetDataType(input_dtype);
        output_desc.SetFormat(input_format);
        op.UpdateOutputDesc("y", output_desc);

        return GRAPH_SUCCESS;
    } else if (mode == "linear") {
        if (input_dims.size() != 4)
        {
            return GRAPH_FAILED;
        }
        if (sizes.size() != 1)
        {
            return GRAPH_FAILED;
        }

        int64_t n = input_dims[0];
        int64_t c = input_dims[1];
        int64_t h = input_dims[2];
        int64_t output_w = sizes[0];

        std::vector<int64_t> dim_vec;
        dim_vec.push_back(n);
        dim_vec.push_back(c);
        dim_vec.push_back(h);
        dim_vec.push_back(output_w);
        Shape output_shape = Shape(dim_vec);

        output_desc.SetShape(output_shape);
        output_desc.SetDataType(input_dtype);
        output_desc.SetFormat(input_format);
        op.UpdateOutputDesc("y", output_desc);
        return GRAPH_SUCCESS;
    } else {
        return GRAPH_FAILED;
    }
}

IMPLEMT_VERIFIER(ResizeD, ResizeDVerify) {
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ResizeD, ResizeDInferShape);
VERIFY_FUNC_REG(ResizeD, ResizeDVerify);
// ---------------ResizeD Op End-------------------

// ---------------ResizeGradD Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(ResizeGradDInferShape) {
    TensorDesc out_put_desc = op.GetOutputDesc("y");
    DataType input_dtype = op.GetInputDesc("grads").GetDataType();
    Format input_format = op.GetInputDesc("grads").GetFormat();
    Shape input_shape = op.GetInputDesc("grads").GetShape();
    std::vector<int64_t> input_dims = input_shape.GetDims();
    std::vector<int64_t> original_size;
    op.GetAttr("original_size", original_size);

    std::string mode = "nearest";
    op.GetAttr("mode", mode);

    if (mode == "linear") {
        if (input_dims.size() != 4 || input_dims[2] != 1)
        {
            return GRAPH_FAILED;
        }
        if (original_size.size() != 3) {
            return GRAPH_FAILED;
        }

        int64_t N = input_dims[0];
        int64_t C = input_dims[1];
        int64_t H = input_dims[2];
        int64_t output_W = original_size[2];

        std::vector < int64_t > dim_vec;
        dim_vec.push_back(N);
        dim_vec.push_back(C);
        dim_vec.push_back(H);
        dim_vec.push_back(output_W);
        Shape output_shape = Shape(dim_vec);

        out_put_desc.SetShape(output_shape);
        out_put_desc.SetDataType(input_dtype);
        out_put_desc.SetFormat(input_format);
        op.UpdateOutputDesc("y", out_put_desc);

        return GRAPH_SUCCESS;
    } else if (mode == "cubic") {
        std::vector<int64_t> dim_vec(original_size);
        out_put_desc.SetShape(Shape(dim_vec));
        out_put_desc.SetDataType(input_dtype);
        out_put_desc.SetFormat(input_format);
        op.UpdateOutputDesc("y", out_put_desc);

        return GRAPH_SUCCESS;
    }

    return GRAPH_FAILED;
}

IMPLEMT_VERIFIER(ResizeGradD, ResizeGradDVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ResizeGradD, ResizeGradDInferShape);
VERIFY_FUNC_REG(ResizeGradD, ResizeGradDVerify);
// ---------------ResizeGradD Op End-------------------


// ---------------ResizeNearestNeighborV2D Op Start-------------------
COMMON_INFER_FUNC_REG(ResizeNearestNeighborV2D, ResizeInferShape);
// ---------------ResizeNearestNeighborV2D Op End-------------------

// ---------------ResizeNearestNeighborV2 Op Start-------------------
// ---------------ResizeBilinearV2 Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(ResizeNearestNeighborV2InferShape) {
  static const std::pair<uint32_t, std::string> input_x{0, "x"};
  static const std::pair<uint32_t, std::string> input_size{1, "size"};
  static const std::pair<uint32_t, std::string> output_y{0, "y"};
  const vector<string> depends{input_size.second};
  PREPARE_DYNAMIC_SHAPE(depends);
  if (!ResizeConstInferShape(op, input_x, input_size, output_y)) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(ResizeNearestNeighborV2, ResizeNearestNeighborV2InferShape);
INFER_VALUE_RANGE_DEFAULT_REG(ResizeNearestNeighborV2);
// ---------------ResizeNearestNeighborV2 Op End-------------------

// ---------------ResizeBilinearV2Grad Op Start-------------------
IMPLEMT_INFERFUNC(ResizeBilinearV2Grad, ResizeBilinearV2GradInfer) {
  auto op_desc_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc_grad = op_desc_info->MutableInputDesc("grads");
  auto input_desc_image = op_desc_info->MutableInputDesc("original_image");
  vector<int64_t> grads_shape = input_desc_grad->MutableShape().GetDims();
  vector<int64_t> images_shape = input_desc_image->MutableShape().GetDims();
  DataType input_dtype = input_desc_image->GetDataType();
  Format input_format = input_desc_grad->GetFormat();
  auto output_desc_y = op_desc_info->MutableOutputDesc("y");
  std::vector<std::pair<int64_t, int64_t>> grads_range;
  std::vector<std::pair<int64_t, int64_t>> image_range;

  bool is_unkown_rank_grads = grads_shape == UNKNOWN_RANK ? true : false;
  bool is_unkown_rank_images = images_shape == UNKNOWN_RANK ? true : false;

  if (is_unkown_rank_grads) {
    OP_LOGW(TbeGetName(op).c_str(), "the input os unkown rank, will set the input -1, -1, -1 , -1");
    grads_shape = {-1, -1, -1, -1};
  } else {
    input_desc_grad->GetShapeRange(grads_range);
  }
  if (is_unkown_rank_images) {
    OP_LOGW(TbeGetName(op).c_str(), "the input os unkown rank, will set the input -1, -1, -1 , -1");
    images_shape = {-1, -1, -1, -1};
  } else {
    input_desc_image->GetShapeRange(image_range);
  }

  MakeUpShapeRange(grads_shape, grads_range);
  MakeUpShapeRange(images_shape, image_range);

  std::vector<std::pair<int64_t, int64_t>> y_range;
  vector<int64_t> y_shape;
  if (input_format == FORMAT_NHWC && grads_shape.size() > 3 && images_shape.size() > 2) {
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(images_shape[1]);
    y_shape.push_back(images_shape[2]);
    y_shape.push_back(grads_shape[3]);
    y_range.push_back(grads_range[0]);
    y_range.push_back(image_range[1]);
    y_range.push_back(image_range[2]);
    y_range.push_back(grads_range[3]);
  } else if (input_format == FORMAT_NCHW && grads_shape.size() > 1 && images_shape.size() > 3) {
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(grads_shape[1]);
    y_shape.push_back(images_shape[2]);
    y_shape.push_back(images_shape[3]);
    y_range.push_back(grads_range[0]);
    y_range.push_back(grads_range[1]);
    y_range.push_back(image_range[2]);
    y_range.push_back(image_range[3]);
  } else {
    string expected_format_list = ConcatString("FORMAT_NHWC, FORMAT_NCHW");
    std::string err_msg = GetInputFormatNotSupportErrMsg("input_format", expected_format_list, ConcatString(input_format));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
  }
  output_desc_y->SetShape(GeShape(y_shape));
  output_desc_y->SetOriginShape(GeShape(y_shape));
  output_desc_y->SetShapeRange(y_range);
  output_desc_y->SetDataType(input_dtype);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ResizeBilinearV2Grad, ResizeBilinearV2GradInfer);
// ---------------ResizeBilinearV2Grad Op End-------------------
// ---------------SyncResizeBilinearV2Grad Op Start-------------------
IMPLEMT_INFERFUNC(SyncResizeBilinearV2Grad, SyncResizeBilinearV2GradInfer) {
  auto op_desc_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc_grad = op_desc_info->MutableInputDesc("grads");
  auto input_desc_image = op_desc_info->MutableInputDesc("original_image");
  vector<int64_t> grads_shape = input_desc_grad->MutableShape().GetDims();
  vector<int64_t> images_shape = input_desc_image->MutableShape().GetDims();
  DataType input_dtype = input_desc_image->GetDataType();
  Format input_format = input_desc_grad->GetFormat();
  auto output_desc_y = op_desc_info->MutableOutputDesc("y");
  std::vector<std::pair<int64_t, int64_t>> grads_range;
  std::vector<std::pair<int64_t, int64_t>> image_range;

  bool is_unkown_rank_grads = grads_shape == UNKNOWN_RANK ? true : false;
  bool is_unkown_rank_images = images_shape == UNKNOWN_RANK ? true : false;

  if (is_unkown_rank_grads) {
    OP_LOGW(TbeGetName(op).c_str(), "the input os unkown rank, will set the input -1, -1, -1 , -1");
    grads_shape = {-1, -1, -1, -1};
  } else {
    input_desc_grad->GetShapeRange(grads_range);
  }
  if (is_unkown_rank_images) {
    OP_LOGW(TbeGetName(op).c_str(), "the input os unkown rank, will set the input -1, -1, -1 , -1");
    images_shape = {-1, -1, -1, -1};
  } else {
    input_desc_image->GetShapeRange(image_range);
  }

  MakeUpShapeRange(grads_shape, grads_range);
  MakeUpShapeRange(images_shape, image_range);

  std::vector<std::pair<int64_t, int64_t>> y_range;
  vector<int64_t> y_shape;
  if (input_format == FORMAT_NHWC && grads_shape.size() > 3 && images_shape.size() > 2) {
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(images_shape[1]);
    y_shape.push_back(images_shape[2]);
    y_shape.push_back(grads_shape[3]);
    y_range.push_back(grads_range[0]);
    y_range.push_back(image_range[1]);
    y_range.push_back(image_range[2]);
    y_range.push_back(grads_range[3]);
  } else if (input_format == FORMAT_NCHW && grads_shape.size() > 1 && images_shape.size() > 3) {
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(grads_shape[1]);
    y_shape.push_back(images_shape[2]);
    y_shape.push_back(images_shape[3]);
    y_range.push_back(grads_range[0]);
    y_range.push_back(grads_range[1]);
    y_range.push_back(image_range[2]);
    y_range.push_back(image_range[3]);
  } else {
    string expected_format_list = ConcatString("FORMAT_NHWC, FORMAT_NCHW");
    std::string err_msg = GetInputFormatNotSupportErrMsg("input_format", expected_format_list, ConcatString(input_format));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
  }
  output_desc_y->SetShape(GeShape(y_shape));
  output_desc_y->SetOriginShape(GeShape(y_shape));
  output_desc_y->SetShapeRange(y_range);
  output_desc_y->SetDataType(input_dtype);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SyncResizeBilinearV2Grad, SyncResizeBilinearV2GradInfer);
// ---------------SyncResizeBilinearV2Grad Op End-------------------

IMPLEMT_INFERFUNC(EncodeJpeg, EncodeJpegInfer) {
  return EncodeImageShapeFn(op);
}

INFER_FUNC_REG(EncodeJpeg, EncodeJpegInfer);

IMPLEMT_INFERFUNC(ExtractJpegShape, ExtractJpegShapeInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc(0), 0, unused_shape, TbeGetName(op).c_str())
      != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(op.GetInputDesc(0).GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  DataType output_type;
  if (op.GetAttr("output_type", output_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
                                      string("get attr[output_type] failed"));
    return GRAPH_FAILED;
  }
  Shape output_shape;
  Vector(3, output_shape);
  TensorDesc image_shape_desc = op.GetOutputDesc("image_shape");
  image_shape_desc.SetShape(output_shape);
  image_shape_desc.SetDataType(output_type);
  image_shape_desc.SetFormat(FORMAT_NHWC);
  if (op.UpdateOutputDesc("image_shape", image_shape_desc)
      != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
                                      string("update output[image_shape] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ExtractJpegShape, ExtractJpegShapeInfer);

IMPLEMT_INFERFUNC(DrawBoundingBoxesV2, DrawBoundingBoxesV2Infer) {
  auto imagesTensor = op.get_input_desc_images();

  Shape images;
  if (WithRankAtLeast(imagesTensor, 3, images, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
        ConcatString("call WithRankAtLeast function failed, ",
            GetShapeErrMsg(0, DebugString(imagesTensor.GetShape().GetDims()),
            "at least 3D")));
    return GRAPH_FAILED;
  }

  DataType type = op.GetInputDesc("images").GetDataType();
  TensorDesc outputDesc = op.GetOutputDesc("y");
  outputDesc.SetDataType(type);
  outputDesc.SetShape(images);
  return op.UpdateOutputDesc("y", outputDesc);
}

INFER_FUNC_REG(DrawBoundingBoxesV2, DrawBoundingBoxesV2Infer);

IMPLEMT_INFERFUNC(NonMaxSuppressionV5, NonMaxSuppressionV5Infer) {
  Shape boxes;
  Shape scores;
  Shape max_output_size;
  Shape iouThreshold;
  Shape scoreThreshold;
  Shape softNmsSigma;

  if (WithRank(op.GetInputDesc(0), 2, boxes, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, 
                                         DebugString(op.GetInputDesc(0).GetShape().GetDims()),
                                         "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(1), 1, scores, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1, 
                                         DebugString(op.GetInputDesc(1).GetShape().GetDims()),
                                         "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(2), 0, max_output_size, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2, 
                                         DebugString(op.GetInputDesc(2).GetShape().GetDims()), 
                                         "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(3), 0, iouThreshold, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(3, 
                                         DebugString(op.GetInputDesc(3).GetShape().GetDims()),
                                         "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(4), 0, scoreThreshold, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(4, 
                                         DebugString(op.GetInputDesc(4).GetShape().GetDims()), 
                                         "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(5), 0, softNmsSigma, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(5, 
                                         DebugString(op.GetInputDesc(5).GetShape().GetDims()), 
                                         "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t un_used;

  if (Merge(boxes.GetDim(0), scores.GetDim(0), un_used) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call Merge function, 0th dim[",
                                       boxes.GetDim(0), "] of input[boxes] not equal 0th dim[",
                                       scores.GetDim(0), "] of input[scores]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (boxes.GetDim(1) != 4) {
    if (boxes.GetDim(1) != UNKNOWN_DIM) {
      std::string err_msg = ConcatString("1th dim[", 
                                         boxes.GetDim(1), 
                                         "] of input[boxes] not equal 4 or -1.");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }

  bool pad_to_max;
  if (ge::GRAPH_SUCCESS != op.GetAttr("pad_to_max_output_size", pad_to_max)) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("get attr[pad_to_max_output_size] failed"));
    return GRAPH_FAILED;
  }

  TensorDesc out_desc = op.GetOutputDesc("selected_indices");
  TensorDesc out_desc_scores = op.GetOutputDesc("selected_scores");
  out_desc.SetDataType(DT_INT32);
  DataType type;
  if (op.GetAttr("T", type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("get attr[T] failed"));
    return GRAPH_FAILED;
  }
  out_desc_scores.SetDataType(type);

  if (!pad_to_max) {
    out_desc.SetShape(Shape({ge::UNKNOWN_DIM}));
    out_desc_scores.SetShape(Shape({ge::UNKNOWN_DIM}));
    if (op.UpdateOutputDesc("selected_indices", out_desc) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), string("update description for output[selected_indices] failed"));
      return GRAPH_FAILED;
    }
    if (op.UpdateOutputDesc("selected_scores", out_desc_scores) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), string("update description for output[selected_scores] failed"));
      return GRAPH_FAILED;
    }
  } else {
    Tensor in_tensor;
    if (op.GetInputConstData("max_output_size", in_tensor) != GRAPH_SUCCESS) {
      out_desc.SetShape(Shape({ge::UNKNOWN_DIM}));
      out_desc_scores.SetShape(Shape({ge::UNKNOWN_DIM}));
      (void)op.UpdateOutputDesc("selected_indices", out_desc);
      (void)op.UpdateOutputDesc("selected_scores", out_desc_scores);
    } else {
      const int32_t* size_data = reinterpret_cast<const int32_t*>(in_tensor.GetData());
      if (*size_data < 0) {
        std::string err_msg = ConcatString("0th data[", *size_data, "] of input[max_output_size] at least 0");
        AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
      }
      out_desc.SetShape(Shape({*size_data}));
      (void)op.UpdateOutputDesc("selected_indices", out_desc);
      out_desc_scores.SetShape(Shape({*size_data}));
      (void)op.UpdateOutputDesc("selected_scores", out_desc_scores);
    }
  }

  TensorDesc out_desc1 = op.GetOutputDesc("valid_outputs");
  out_desc1.SetShape(Shape());
  out_desc1.SetDataType(ge::DT_INT32);
  (void)op.UpdateOutputDesc("valid_outputs", out_desc1);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(NonMaxSuppressionV5, NonMaxSuppressionV5Infer);

IMPLEMT_INFERFUNC(ScaleAndTranslate, ScaleAndTranslateInfer) {
  TensorDesc desc = op.GetOutputDesc("y");
  desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        string("update description for output[y] failed."));
    return GRAPH_FAILED;
  }
  return ResizeShapeFn(op, "images", "size", "y");
}

INFER_FUNC_REG(ScaleAndTranslate, ScaleAndTranslateInfer);

IMPLEMT_INFERFUNC(ScaleAndTranslateGrad, ScaleAndTranslateGradInfer) {
  TensorDesc desc = op.GetOutputDesc("y");
  Format input_format = op.GetInputDesc(0).GetFormat();
  vector<int64_t> grads_shape = op.GetInputDesc(0).GetShape().GetDims();
  vector<int64_t> org_images_shape = op.GetInputDesc(1).GetShape().GetDims();
  vector<int64_t> y_shape;
  if (input_format == FORMAT_NHWC && grads_shape.size() > 3 && org_images_shape.size() > 2) {
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(org_images_shape[1]);
    y_shape.push_back(org_images_shape[2]);
    y_shape.push_back(grads_shape[3]);
  } else if (input_format == FORMAT_NCHW && grads_shape.size() > 1 && org_images_shape.size() > 3) {
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(grads_shape[1]);
    y_shape.push_back(org_images_shape[2]);
    y_shape.push_back(org_images_shape[3]);
  } else {
    if (grads_shape.size() < 4) {
      std::string err_msg = ConcatString(
        "the 0th input[grads]'s rank should not be less than 4, ",
        "current rank is ", grads_shape.size());
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
    if (org_images_shape.size() < 2) {
      std::string err_msg = ConcatString(
        "the 1th input[original_images]'s rank should not be less than 2, ",
        "current rank is ", org_images_shape.size());
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(org_images_shape[1]);
    y_shape.push_back(org_images_shape[2]);
    y_shape.push_back(grads_shape[3]);
    OP_LOGI(TbeGetName(op).c_str(), "Real format is %d", input_format);
  }

  desc.SetShape(ge::Shape(y_shape));
  desc.SetDataType(DT_FLOAT);
  return op.UpdateOutputDesc("y", desc);
}

INFER_FUNC_REG(ScaleAndTranslateGrad, ScaleAndTranslateGradInfer);

// ---------------IMGWarp Op start-------------------
IMPLEMT_COMMON_INFERFUNC(IMGWarpInferShape) {
  OP_LOGI(TbeGetName(op).c_str(), "start to infershape for IMGWarp.");
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_info == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid OpDesc.")), return GRAPH_FAILED);
  auto image_desc = op_info->MutableInputDesc("img");
  auto offset_desc = op_info->MutableInputDesc("warp_offset");
  auto image_dtype = image_desc->GetDataType();
  vector<int64_t> image_shape = image_desc->MutableShape().GetDims();
  vector<int64_t> offset_shape = offset_desc->MutableShape().GetDims();

  // check image_shape//offset_shape must be 4dims
  if (image_shape.size() != DIM_SIZE4) {
    std::string err_msg = GetAttrSizeErrMsg("img", ConcatString(image_shape.size()), ConcatString(DIM_SIZE4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (offset_shape.size() != DIM_SIZE4) {
    std::string err_msg = GetAttrSizeErrMsg("warp_offset", ConcatString(offset_shape.size()), ConcatString(DIM_SIZE4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  vector<int64_t> output_shape = image_shape;
  output_shape[2] = offset_shape[2];
  output_shape[3] = offset_shape[3];
  auto output_desc = op_info->MutableOutputDesc("warp_img");
  output_desc->SetShape(GeShape(output_shape));
  output_desc->SetOriginShape(GeShape(output_shape));
  output_desc->SetDataType(image_dtype);
  OP_LOGI(TbeGetName(op).c_str(), "end to infershape for IMGWarp.");
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(IMGWarp, IMGWarpInferShape);
// ----------------IMGWarp END---------------------

// ---------------Remap Op start-------------------
IMPLEMT_COMMON_INFERFUNC(RemapInferShape) {
  OP_LOGI(TbeGetName(op).c_str(), "start to infershape for Remap.");
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_info == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), OtherErrMsg("invalid OpDesc.")), return GRAPH_FAILED);
  auto image_desc = op_info->MutableInputDesc("img");
  auto offset_desc = op_info->MutableInputDesc("map_offset");
  auto image_dtype = image_desc->GetDataType();
  vector<int64_t> image_shape = image_desc->MutableShape().GetDims();
  vector<int64_t> offset_shape = offset_desc->MutableShape().GetDims();

  // check image_shape//offset_shape must be 4dims
  if (image_shape.size() != DIM_SIZE4) {
    std::string err_msg = GetAttrSizeErrMsg("img", ConcatString(image_shape.size()), ConcatString(DIM_SIZE4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (offset_shape.size() != DIM_SIZE4) {
    std::string err_msg = GetAttrSizeErrMsg("map_offset", ConcatString(offset_shape.size()), ConcatString(DIM_SIZE4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  vector<int64_t> output_shape = image_shape;
  output_shape[1] = offset_shape[1];
  output_shape[2] = offset_shape[2];
  auto output_desc = op_info->MutableOutputDesc("map_img");
  output_desc->SetShape(GeShape(output_shape));
  output_desc->SetOriginShape(GeShape(output_shape));
  output_desc->SetDataType(image_dtype);
  OP_LOGI(TbeGetName(op).c_str(), "end to infershape for Remap.");
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(Remap, RemapInferShape);
// ----------------Remap END---------------------

IMPLEMT_INFERFUNC(CombinedNonMaxSuppression, CombinedNonMaxSuppressionInfer) {
  DYNAMIC_SHAPE_NOT_SUPPORTED(op);
  Shape boxes;
  Shape scores;
  Shape max_output_size_per_class;
  Shape max_total_size;
  Shape unused_shape;

  if (WithRank(op.GetInputDesc(0), 4, boxes, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
        GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "4D"));
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 3, scores, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
        GetShapeErrMsg(1, DebugString(op.GetInputDesc(1).GetShape().GetDims()), "3D"));
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(2), 0, max_output_size_per_class, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
        GetShapeErrMsg(2, DebugString(op.GetInputDesc(2).GetShape().GetDims()), "scalar"));
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(3), 0, max_total_size, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
        GetShapeErrMsg(3, DebugString(op.GetInputDesc(3).GetShape().GetDims()), "scalar"));
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(4), 0, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
        GetShapeErrMsg(4, DebugString(op.GetInputDesc(4).GetShape().GetDims()), "scalar"));
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(5), 0, unused_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
        GetShapeErrMsg(5, DebugString(op.GetInputDesc(5).GetShape().GetDims()), "scalar"));
    return GRAPH_FAILED;
  }

  int64_t unused = 0;
  int64_t dim1 = boxes.GetDim(0);
  int64_t dim2 = scores.GetDim(0);
  if (Merge(dim1, dim2, unused) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
        ConcatString("call Merge function failed to merge 0th dim of input[boxes]"
        " and input[scores], ", dim1, " and ", dim2));
    return GRAPH_FAILED;
  }
  int64_t dim3 = boxes.GetDim(1);
  int64_t dim4 = scores.GetDim(1);
  if (Merge(dim3, dim4, unused) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
        ConcatString("call Merge function failed to merge 1th dim of input[boxes]"
        " and input[scores], ", dim3, " and ", dim4));
    return GRAPH_FAILED;
  }

  if (boxes.GetDim(3) != 4) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        ConcatString("invalid 3th dim value[", boxes.GetDim(3), "], it should be 4"));
    return GRAPH_FAILED;
  }

  Shape boxes_shape = op.GetInputDesc(0).GetShape();
  Shape scores_shape = op.GetInputDesc(1).GetShape();
  if (ValueKnown(boxes_shape, 2) && ValueKnown(scores_shape, 2)) {
    if (boxes_shape.GetDim(2) != 1 && boxes_shape.GetDim(2) != scores_shape.GetDim(2)) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
          ConcatString("2th dim of input[boxes] and input[scores] are not equal, ",
              boxes_shape.GetDim(2), " and ", scores_shape.GetDim(2)));
      return GRAPH_FAILED;
    }
  }

  Tensor maxTotalSizeTensor;
  if (op.GetInputConstData("max_total_size", maxTotalSizeTensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("get const data from input[max_total_size] failed"));
    return GRAPH_FAILED;
  }
  int64_t maxTotalSize;
  if (MakeDimForScalarInput(maxTotalSizeTensor, maxTotalSize, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
        ConcatString("call MakeDimForScalarInput failed to get value from input[max_total_size] tensor"));
    return GRAPH_FAILED;
  }
  if (maxTotalSize <= 0) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        ConcatString("invalid value[", maxTotalSize, "] of input[max_total_size], should be > 0"));
    return GRAPH_FAILED;
  }

  Tensor maxOutputSizePerClassTensor;
  if (op.GetInputConstData("max_output_size_per_class", maxOutputSizePerClassTensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
        std::string("get const data from input[max_output_size_per_class] failed"));
    return GRAPH_FAILED;
  }
  int64_t maxOutputSizePerClass;
  if (MakeDimForScalarInput(maxOutputSizePerClassTensor, maxOutputSizePerClass, TbeGetName(op).c_str()) !=
      GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op),
        ConcatString("call MakeDimForScalarInput failed to get value from input[max_output_size_per_class] tensor"));
    return GRAPH_FAILED;
  }

  int64_t output_size;
  bool pad_per_class;
  if (op.GetAttr("pad_per_class", pad_per_class) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(),
        std::string("get attr[pad_per_class] failed"));
    return GRAPH_FAILED;
  }
  if (!pad_per_class) {
    output_size = maxTotalSize;
  } else {
    if (maxOutputSizePerClass <= 0) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op),
          ConcatString("invalid value[", maxOutputSizePerClass,
              "] of input[max_output_size_per_class], should be > 0"));
      return GRAPH_FAILED;
    }
    if (maxTotalSize <= maxOutputSizePerClass * scores_shape.GetDim(2)) {
      output_size = maxTotalSize;
    } else {
      output_size = maxOutputSizePerClass * scores_shape.GetDim(2);
    }
  }

  int64_t batch_dim = boxes.GetDim(0);
  Shape shape1({batch_dim, output_size, 4});
  Shape shape2({batch_dim, output_size});
  Shape shape3({batch_dim, output_size});
  Shape shape4({batch_dim});

  TensorDesc desc1 = op.GetOutputDesc("nmsed_boxes");
  desc1.SetShape(shape1);
  desc1.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("nmsed_boxes", desc1) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(),
        std::string("update output[nmsed_boxes] desc failed"));
    return GRAPH_FAILED;
  }
  TensorDesc desc2 = op.GetOutputDesc("nmsed_scores");
  desc2.SetShape(shape2);
  desc2.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("nmsed_scores", desc2) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(),
        std::string("update output[nmsed_scores] desc failed"));
    return GRAPH_FAILED;
  }
  TensorDesc desc3 = op.GetOutputDesc("nmsed_classes");
  desc3.SetShape(shape3);
  desc3.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("nmsed_classes", desc3) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(),
        std::string("update output[nmsed_classes] desc failed"));
    return GRAPH_FAILED;
  }
  TensorDesc desc4 = op.GetOutputDesc("valid_detections");
  desc4.SetShape(shape4);
  desc4.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("valid_detections", desc4) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(),
        std::string("update output[valid_detections] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(CombinedNonMaxSuppression, CombinedNonMaxSuppressionInfer);

IMPLEMT_INFERFUNC(SpatialTransformerD, SpatialTransformerDInferShape) {
  auto x_shape = op.get_input_desc_x().GetShape();
  auto x_dtype = op.get_input_desc_x().GetDataType();

  std::vector<int64_t> output_size = op.get_attr_output_size();
  CHECK(output_size.size() == 1,  OP_LOGE(TbeGetName(op).c_str(), "invalid output size in attr."), return GRAPH_FAILED);
  if (output_size.empty()) {
    output_size.push_back(x_shape.GetDim(2));
    output_size.push_back(x_shape.GetDim(3));
  } else {
    output_size[0] = (output_size[0] == -1) ? x_shape.GetDim(2) : output_size[0];
    output_size[1] = (output_size[1] == -1) ? x_shape.GetDim(3) : output_size[1];
  }

  vector<int64_t> y_shape({x_shape.GetDim(0), x_shape.GetDim(1), output_size[0], output_size[1]});

  auto out_desc = op.GetOutputDesc("y");
  out_desc.SetShape(Shape(y_shape));
  out_desc.SetDataType(ge::DataType(x_dtype));
  (void)op.update_output_desc_y(out_desc);

  // record original channel
  op.SetAttr("stn_ori_channel", x_shape.GetDim(1));

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SpatialTransformerD, SpatialTransformerDInferShape);

IMPLEMT_INFERFUNC(SpatialTransformer, SpatialTransformerInferShape) {
  auto x_shape = op.get_input_desc_x().GetShape();
  auto x_dtype = op.get_input_desc_x().GetDataType();

  std::vector<int64_t> output_size = op.get_attr_output_size();
  CHECK(output_size.size() == 1,  OP_LOGE(TbeGetName(op).c_str(), "invalid output size in attr."), return GRAPH_FAILED);
  if (output_size.empty()) {
    output_size.push_back(x_shape.GetDim(2));
    output_size.push_back(x_shape.GetDim(3));
  } else {
    output_size[0] = (output_size[0] == -1) ? x_shape.GetDim(2) : output_size[0];
    output_size[1] = (output_size[1] == -1) ? x_shape.GetDim(3) : output_size[1];
  }

  vector<int64_t> y_shape({x_shape.GetDim(0), x_shape.GetDim(1), output_size[0], output_size[1]});

  auto out_desc = op.GetOutputDesc("y");
  out_desc.SetShape(Shape(y_shape));
  out_desc.SetDataType(ge::DataType(x_dtype));
  (void)op.update_output_desc_y(out_desc);

  // record original channel
  op.SetAttr("stn_ori_channel", x_shape.GetDim(1));

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SpatialTransformer, SpatialTransformerInferShape);

// ----------------Resize Begin-------------------
static bool GetConstValueFloat(const Operator& op, const Tensor& const_tensor,
                               const DataType& dtype,
                               std::vector<float_t>& const_data) {
  size_t size = 0;
  if (dtype == ge::DT_FLOAT) {
    const float_t* const_data_ptr =
        reinterpret_cast<const float_t*>(const_tensor.GetData());
    size = const_tensor.GetSize() / sizeof(float_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back((float_t)((*(const_data_ptr + i))));
      OP_LOGD(TbeGetName(op).c_str(), "const data int32 fusion pass ====== %d",
              (float_t)(*(const_data_ptr + i)));
    }
  } else {
    std::string err_msg = OtherErrMsg("Not support this type");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return false;
  }
  return true;
}

// ---------------ResizeNearest Op start-------------------
IMPLEMT_INFERFUNC(Resize, ResizeNearestInferShape) {
  OP_LOGI(TbeGetName(op).c_str(), "Resize Start Infer Shape!");
  auto op_desc_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc_x = op_desc_info->MutableInputDesc("x");
  auto input_sizes = op_desc_info->MutableInputDesc("sizes");
  auto input_scales = op_desc_info->MutableInputDesc("scales");
  auto output_desc = op_desc_info->MutableOutputDesc("y");
  DataType input_dtype = input_desc_x->GetDataType();
  Format input_format = input_desc_x->GetFormat();
  vector<int64_t> x_shape = input_desc_x->MutableShape().GetDims();
  std::vector<std::pair<int64_t, int64_t>> x_range;
  // check format and get rank num
  int64_t dim_num;
  
  dim_num = x_shape.size();
  if (dim_num == 4) {
    input_format = FORMAT_NCHW;
    input_desc_x->SetFormat(input_format);
  } else if (dim_num == 5) {
    input_format = FORMAT_NCDHW;
    input_desc_x->SetFormat(input_format);
  } else {
    OP_LOGE(TbeGetName(op).c_str(), "Input format not support");
    return GRAPH_FAILED;
  }
  // unknown rank
  bool is_unknown_rank_x = x_shape == UNKNOWN_RANK;
  if (is_unknown_rank_x) {
    OP_LOGE(TbeGetName(op).c_str(), "Unknown rank not support");
    return GRAPH_FAILED;
  }
  input_desc_x->GetShapeRange(x_range);
  // get x_shape and x_range
  MakeUpShapeRange(x_shape, x_range);
  if (static_cast<int64_t>(x_shape.size()) != dim_num) {
    OP_LOGE(TbeGetName(op).c_str(), "Rank of x_shape not support");
    return GRAPH_FAILED;
  }

  // infer y_range and y_shape
  std::vector<std::pair<int64_t, int64_t>> y_range;
  vector<int64_t> y_shape;

  Tensor sizes_tensor;
  Tensor scales_tensor;
  vector<int64_t> sizes_out;
  vector<float_t> scales_out;
  if (input_sizes != nullptr) {
    // infer shape by sizes
    const vector<string> depend_names = {"sizes"};
    PREPARE_DYNAMIC_SHAPE(depend_names);
    if (op.GetInputConstData("sizes", sizes_tensor) == GRAPH_SUCCESS) {
      DataType sizes_dtype = op.GetInputDesc("sizes").GetDataType();
      GetConstValue(op, sizes_tensor, sizes_dtype, sizes_out);
      if (static_cast<int64_t>(sizes_out.size()) != dim_num) {
        OP_LOGE(TbeGetName(op).c_str(), "Rank of sizesnot support");
        return GRAPH_FAILED;
      }
      for (int64_t i = 0; i < dim_num; i++) {
        y_shape.push_back(sizes_out[i]);
        y_range.push_back({ sizes_out[i], sizes_out[i] });
      }
    }
  } else if (input_scales != nullptr) {
    // infer shape by scales
    const vector<string> depend_names = {"scales"};
    PREPARE_DYNAMIC_SHAPE(depend_names);
    if (op.GetInputConstData("scales", scales_tensor) == GRAPH_SUCCESS) {
      DataType scales_dtype = op.GetInputDesc("scales").GetDataType();
      GetConstValueFloat(op, scales_tensor, scales_dtype, scales_out);
      if (static_cast<int64_t>(scales_out.size()) != dim_num) {
        OP_LOGE(TbeGetName(op).c_str(), "Rank of scales support");
        return GRAPH_FAILED;
      }
      for (int64_t i = 0; i < dim_num; i++) {
        if (x_shape[i] == -1) {
          y_shape.push_back(-1);
          y_range.push_back({ 1, -1 });
        } else {
          int64_t output_dim_num = floor(x_shape[i] * scales_out[i]);
          y_shape.push_back(output_dim_num);
          y_range.push_back({ output_dim_num, output_dim_num });
        }
      }
    }
  } else {
    OP_LOGE(TbeGetName(op).c_str(), "Can not get sizes and scales");
    return GRAPH_FAILED;
  }
  if (y_shape.size() == 0) {
    for (int64_t i = 0; i < dim_num; i++) {
      y_shape.push_back(-1);
      y_range.push_back({ 0, -1 });
    }
  }
  output_desc->SetShape(GeShape(y_shape));
  output_desc->SetOriginShape(GeShape(y_shape));
  output_desc->SetShapeRange(y_range);
  output_desc->SetDataType(input_dtype);
  output_desc->SetFormat(input_format);
  OP_LOGI(TbeGetName(op).c_str(), "Resize Infer Shape Success!");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Resize, ResizeNearestInferShape);
// ---------------ResizeNearest Op End-------------------
// ----------------Resize END---------------------

// ----------------DecodeJpeg Op Start--------------------
IMPLEMT_INFERFUNC(DecodeJpeg, DecodeJpegInfer) {
  TensorDesc contents = op.GetInputDesc(0);
  TensorDesc image = op.GetOutputDesc(0);
  DataType input_data = contents.GetDataType();
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  if (input_data != DT_STRING) {
    OP_LOGE(op_desc->GetName().c_str(), "Input data type must be string, dataType is [%d]", input_data);
    return GRAPH_FAILED;
  }
  image.SetDataType(DT_UINT8);
  std::vector<int64_t> image_shape({UNKNOWN_DIM, UNKNOWN_DIM, 3});
  std::vector<std::pair<int64_t, int64_t>> image_range;
  (void)image_range.emplace_back(std::make_pair(1, -1));
  (void)image_range.emplace_back(std::make_pair(1, -1));
  (void)image_range.emplace_back(std::make_pair(3, 3));
  image.SetShape(Shape(image_shape));
  image.SetShapeRange(image_range);
  if (op.UpdateOutputDesc("image", image) != GRAPH_SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), "Fail to update output image.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DecodeJpeg, DecodeJpegInfer);
//-----------------DecodeJpeg End--------------------------

//-----------------DenseImageWarp Op Start-----------------
IMPLEMT_INFERFUNC(DenseImageWarp, DenseImageWarpInfer) {
  auto image_desc = op.GetInputDesc("image");
  auto image_shape = image_desc.GetShape();
  auto image_dtype = image_desc.GetDataType();
  auto image_format = image_desc.GetFormat();

  auto y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(image_shape);
  y_desc.SetDataType(image_dtype);
  y_desc.SetFormat(image_format);

  std::vector<std::pair<int64_t, int64_t>> image_range;
  if (image_desc.GetShapeRange(image_range) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Fail to get input_image range");
    return GRAPH_FAILED;
  }
  y_desc.SetShapeRange(image_range);

  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Fail to update output y_desc");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(DenseImageWarp, DenseImageWarpVerify) {
  auto image_desc = op.GetInputDesc("image");
  auto flow_desc = op.GetInputDesc("flow");
  auto image_shape = image_desc.GetShape().GetDims();
  auto flow_shape = flow_desc.GetShape().GetDims();
  auto image_format = image_desc.GetFormat();

  if (image_format != FORMAT_NHWC && image_format != FORMAT_NCHW) {
    OP_LOGE(TbeGetName(op).c_str(), "Input image should be NHWC or NCHW format, actual is [%s]",
            TypeUtils::FormatToSerialString(image_format).c_str());
    return GRAPH_FAILED;
  }

  if (image_shape.size() != 4 || flow_shape.size() != 4) {
    OP_LOGE(TbeGetName(op).c_str(),
            "Input image and flow both should be 4d, actual are [image:%zu, flow:%zu]",
            image_shape.size(), flow_shape.size());
    return GRAPH_FAILED;
  }

  std::string image_format_str;
  if (image_format == FORMAT_NHWC) {
    image_format_str = "NHWC";
  } else {
    image_format_str = "NCHW";
  }
  int32_t pos_h = image_format_str.find("H");
  int32_t pos_w = image_format_str.find("W");
  int32_t pos_c = image_format_str.find("C");

  if (flow_shape[pos_c] != 2) {
    OP_LOGE(TbeGetName(op).c_str(),
            "Input flow channel should be 2, actual is %ld", flow_shape[3]);
    return GRAPH_FAILED;
  }

  if (flow_shape[0] != image_shape[0] || flow_shape[pos_h] != image_shape[pos_h] ||
      flow_shape[pos_w] != image_shape[pos_w]) {
    OP_LOGE(TbeGetName(op).c_str(),
            "Input flow batch, height and width should be same as image, actually flow:[%ld, %ld, %ld], image:[%ld, "
            "%ld, %ld]",
            flow_shape[0], flow_shape[pos_h], flow_shape[pos_w], image_shape[0], image_shape[pos_h],
            image_shape[pos_w]);
    return GRAPH_FAILED;
  }

  if (image_shape[pos_h] < 2 || image_shape[pos_w] < 2) {
    OP_LOGE(TbeGetName(op).c_str(),
            "Input image height and width should not be less than 2");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DenseImageWarp, DenseImageWarpInfer);
VERIFY_FUNC_REG(DenseImageWarp, DenseImageWarpVerify);

IMPLEMT_INFERFUNC(DenseImageWarpGrad, DenseImageWarpGradInfer) {
  auto image_desc = op.GetInputDesc("image");
  auto image_shape = image_desc.GetShape();
  auto image_dtype = image_desc.GetDataType();
  auto flow_desc = op.GetInputDesc("flow");
  auto flow_shape = flow_desc.GetShape();
  auto flow_dtype = flow_desc.GetDataType();

  auto grad_image_desc = op.GetOutputDesc("grad_image");
  grad_image_desc.SetShape(image_shape);
  grad_image_desc.SetDataType(image_dtype);
  auto grad_flow_desc = op.GetOutputDesc("grad_flow");
  grad_flow_desc.SetShape(flow_shape);
  grad_flow_desc.SetDataType(flow_dtype);

  std::vector<std::pair<int64_t, int64_t>> image_range;
  std::vector<std::pair<int64_t, int64_t>> flow_range;
  if ((image_desc.GetShapeRange(image_range) != GRAPH_SUCCESS) ||
      (flow_desc.GetShapeRange(flow_range) != GRAPH_SUCCESS)) {
    OP_LOGE(TbeGetName(op).c_str(), "Fail to get input_image or input_flow range");
    return GRAPH_FAILED;
  }
  grad_image_desc.SetShapeRange(image_range);
  grad_flow_desc.SetShapeRange(flow_range);

  if (op.UpdateOutputDesc("grad_image", grad_image_desc) != GRAPH_SUCCESS ||
      op.UpdateOutputDesc("grad_flow", grad_flow_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Fail to update output desc.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(DenseImageWarpGrad, DenseImageWarpGradVerify) {
  auto grad_desc = op.GetInputDesc("grad");
  auto grad_shape = grad_desc.GetShape().GetDims();
  auto grad_format = grad_desc.GetFormat();

  auto image_desc = op.GetInputDesc("image");
  auto image_shape = image_desc.GetShape().GetDims();
  auto image_format = image_desc.GetFormat();

  if (grad_format != image_format) {
    OP_LOGE(TbeGetName(op).c_str(), "Grad format should be same as image format, actually grad: [%s], image: [%s]",
            TypeUtils::FormatToSerialString(grad_format).c_str(), TypeUtils::FormatToSerialString(image_format).c_str());
    return GRAPH_FAILED;
  }

  if (grad_shape.size() != 4 || image_shape.size() != 4) {
    OP_LOGE(TbeGetName(op).c_str(), "Grad shape and image shape should both be 4d, acutally grad: [%zu], image: [%zu]",
            grad_shape.size(), image_shape.size());
    return GRAPH_FAILED;
  }

  if (grad_shape != image_shape) {
    OP_LOGE(
        TbeGetName(op).c_str(),
        "The shape of grad and image should be the same, acutally grad:[%ld, %ld, %ld, %ld], image[%ld, %ld, %ld, %ld]",
        grad_shape[0], grad_shape[1], grad_shape[2], grad_shape[3], image_shape[0], image_shape[1], image_shape[2],
        image_shape[3]);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DenseImageWarpGrad, DenseImageWarpGradInfer);
VERIFY_FUNC_REG(DenseImageWarpGrad, DenseImageWarpGradVerify);
//-----------------DenseImageWarp Op End-------------------
// ---------------GridSampler2D Op start-------------------
IMPLEMT_INFERFUNC(GridSampler2D, GridSampler2DInferShape) {
    vector<int64_t> grid_shape = op.GetInputDesc("grid").GetShape().GetDims();
    vector<int64_t> x_shape = op.GetInputDesc("x").GetShape().GetDims();
    DataType x_dtype = op.GetInputDesc("x").GetDataType();
    Format x_format = op.GetInputDesc("x").GetFormat();

    if (x_shape.size() != 4 || grid_shape.size() != 4) {
        OP_LOGW(TbeGetName(op).c_str(), "Expected dim of x and grid should be 4. x dim is %d. grid dim is %d.",
                x_shape.size(), grid_shape.size());
        return GRAPH_FAILED;
    }

    x_shape[2] = grid_shape[1];
    x_shape[3] = grid_shape[2];
    TensorDesc output_desc_y = op.GetOutputDesc("y");
    output_desc_y.SetShape(ge::Shape(x_shape));
    output_desc_y.SetDataType(x_dtype);
    output_desc_y.SetFormat(x_format);
    (void)op.UpdateOutputDesc("y", output_desc_y);
    return GRAPH_SUCCESS;
}
INFER_FUNC_REG(GridSampler2D, GridSampler2DInferShape);
// ----------------GridSampler2D END---------------------

// ---------------GridSampler2DGrad Op start-------------------
IMPLEMT_INFERFUNC(GridSampler2DGrad, GridSampler2DGradInferShape) {
  AscendString op_name;
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  const int64_t input_grad_id = 0;
  const int64_t input_x_id = 1;
  const int64_t input_grid_id = 2;
  const int64_t output_dx_id = 0;
  const int64_t output_dgrid_id = 1;
  auto grad_desc = op_desc->MutableInputDesc(input_grad_id);
  auto x_desc = op_desc->MutableInputDesc(input_x_id);
  auto grid_desc = op_desc->MutableInputDesc(input_grid_id);
  const GeShape &grad_shape = grad_desc->MutableShape();
  const GeShape &x_shape = x_desc->MutableShape();
  const GeShape &grid_shape = grid_desc->MutableShape();
  const int64_t GridSampler2DGradInputSizeLimit = 4;
  
  if (grad_shape.GetDimNum() != GridSampler2DGradInputSizeLimit) {
    OP_LOGE(TbeGetName(op), "Expected dim of grad should be 4. real value is %lu.", grad_shape.GetDimNum());
    return GRAPH_FAILED;
  }

  if (x_shape.GetDimNum() != GridSampler2DGradInputSizeLimit) {
    OP_LOGE(TbeGetName(op), "Expected dim of x should be 4. real value is %lu.", x_shape.GetDimNum());
    return GRAPH_FAILED;
  }

  if (grid_shape.GetDimNum() != GridSampler2DGradInputSizeLimit) {
    OP_LOGE(TbeGetName(op), "Expected dim of grid should be 4. real value is %lu.", grid_shape.GetDimNum());
    return GRAPH_FAILED;
  }
  
  auto dx_desc = op_desc->MutableOutputDesc(output_dx_id);
  auto dgrid_desc = op_desc->MutableOutputDesc(output_dgrid_id);
  dx_desc->SetShape(x_shape);
  dx_desc->SetDataType(x_desc->GetDataType());
  dgrid_desc->SetShape(grid_shape);
  dgrid_desc->SetDataType(grid_desc->GetDataType());
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(GridSampler2DGrad, GridSampler2DGradInferShape);
// ---------------GridSampler2DGrad Op END-------------------

// ---------------GridUnnormal Op start-------------------
IMPLEMT_INFERFUNC(GridUnnormal, GridUnnormalInferShape) {
    vector<int64_t> grid_shape = op.GetInputDesc("grid").GetShape().GetDims();
    vector<int64_t> x_shape = op.GetInputDesc("assist").GetShape().GetDims();
    DataType grid_dtype = op.GetInputDesc("grid").GetDataType();
    Format grid_format = op.GetInputDesc("grid").GetFormat();

    if (x_shape.size() != 4 || grid_shape.size() != 4) {
        OP_LOGW(TbeGetName(op).c_str(), "Expected dim of assist and grid should be 4. assist dim is %d. grid dim is %d.",
                x_shape.size(), grid_shape.size());
        return GRAPH_FAILED;
    }

    if (grid_shape[3] != 2) {
        OP_LOGW(TbeGetName(op).c_str(), "Expected last dim of grid should be 2. last dim of grid is %d.", grid_shape[3]);
        return GRAPH_FAILED;
    }

    TensorDesc diff_desc = op.GetOutputDesc("diff");
    diff_desc.SetShape(ge::Shape(grid_shape));
    diff_desc.SetDataType(grid_dtype);
    diff_desc.SetFormat(grid_format);
    (void)op.UpdateOutputDesc("diff", diff_desc);

    TensorDesc pos_desc = op.GetOutputDesc("position");
    pos_desc.SetShape(ge::Shape(grid_shape));
    pos_desc.SetDataType(DT_INT32);
    pos_desc.SetFormat(grid_format);
    (void)op.UpdateOutputDesc("position", pos_desc);
    return GRAPH_SUCCESS;
}
INFER_FUNC_REG(GridUnnormal, GridUnnormalInferShape);
// ----------------GridUnnormal END---------------------

// ---------------ImageUnfold Op start-------------------
IMPLEMT_INFERFUNC(ImageUnfold, ImageUnfoldInferShape) {
    // N,C,Hin,Win
    vector<int64_t> x_shape = op.GetInputDesc("x").GetShape().GetDims();
    // N,Hout,Wout,4
    vector<int64_t> pos_shape = op.GetInputDesc("position").GetShape().GetDims();
    DataType x_dtype = op.GetInputDesc("x").GetDataType();
    Format x_format = op.GetInputDesc("x").GetFormat();

    if (x_shape.size() != 4 || pos_shape.size() != 4) {
        OP_LOGW(TbeGetName(op).c_str(), "Expected dim of x and position should be 4. x dim is %d. position dim is %d.",
                x_shape.size(), pos_shape.size());
        return GRAPH_FAILED;
    }

    vector<int64_t> output_shape = x_shape;
    output_shape[2] = pos_shape[1];
    output_shape[3] = pos_shape[2];
    TensorDesc output_desc_y = op.GetOutputDesc("y");
    output_desc_y.SetShape(ge::Shape(output_shape));
    output_desc_y.SetDataType(x_dtype);
    output_desc_y.SetFormat(x_format);
    (void)op.UpdateOutputDesc("y", output_desc_y);
    return GRAPH_SUCCESS;
}
INFER_FUNC_REG(ImageUnfold, ImageUnfoldInferShape);
// ----------------ImageUnfold END---------------------

// ---------------IMGWarpOffsets Op start-------------------
IMPLEMT_INFERFUNC(IMGWarpOffsets, IMGWarpOffsetsInferShape) {
  std::string op_name = TbeGetName(op);
  // N,H,W,3
  vector<int64_t> images_shape =
      op.GetInputDescByName("images").GetShape().GetDims();

  // N,4,h,w
  vector<int64_t> offsets_shape =
      op.GetInputDescByName("offsets").GetShape().GetDims();

  // N,4,h,w,3
  vector<int64_t> output_shape = UNKNOWN_RANK;
  if ((images_shape != UNKNOWN_RANK) && (offsets_shape != UNKNOWN_RANK)) {
    Shape unused;
    if (WithRank(op.GetInputDesc(0), 4, unused, op_name.c_str()) !=
        GRAPH_SUCCESS) {
      std::string err_msg =
          GetShapeErrMsg(0, DebugString(images_shape), "[N, H, W, 3]");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op_name, err_msg);
      return GRAPH_FAILED;
    }
    // image channels: 3
    if ((images_shape[3] != 3) && (images_shape[3] != UNKNOWN_DIM)) {
      std::string err_msg = ConcatString(
          "input[0] last dim should be 3, but got [", images_shape[3], "]");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg);
      return GRAPH_FAILED;
    }

    if (WithRank(op.GetInputDesc(1), 4, unused, op_name.c_str()) !=
        GRAPH_SUCCESS) {
      std::string err_msg =
          GetShapeErrMsg(1, DebugString(offsets_shape), "[N, 4, H, W]");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op_name, err_msg);
      return GRAPH_FAILED;
    }
    // four points: 4
    if ((offsets_shape[1] != 4) && (offsets_shape[1] != UNKNOWN_DIM)) {
      std::string err_msg = ConcatString(
          "input[1] second dim should be 4, but got [", offsets_shape[1], "]");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg);
      return GRAPH_FAILED;
    }

    if (images_shape[0] != offsets_shape[0]) {
      std::string err_msg = ConcatString(
          "input[0] first dim[", images_shape[0],
          "] should be equel to input[1] first dim[", offsets_shape[0], "]");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg);
      return GRAPH_FAILED;
    }
    output_shape.clear();
    output_shape.emplace_back(offsets_shape[0]);
    output_shape.emplace_back(offsets_shape[1]);
    output_shape.emplace_back(offsets_shape[2]);
    output_shape.emplace_back(offsets_shape[3]);
    output_shape.emplace_back(images_shape[3]);
  }

  DataType images_dtype = op.GetInputDescByName("images").GetDataType();
  TensorDesc output_desc = op.GetOutputDescByName("warp_images");
  output_desc.SetShape(ge::Shape(output_shape));
  output_desc.SetDataType(images_dtype);
  return op.UpdateOutputDesc("warp_images", output_desc);
}
INFER_FUNC_REG(IMGWarpOffsets, IMGWarpOffsetsInferShape);
// ----------------IMGWarpOffsets END---------------------

// ---------------GridSampler3D Op start-------------------
IMPLEMT_INFERFUNC(GridSampler3D, GridSampler3DInferShape) {
  TensorDesc x_desc = op.GetInputDescByName("x");
  TensorDesc grid_desc = op.GetInputDescByName("grid");
  vector<int64_t> grid_shape = grid_desc.GetShape().GetDims();  // NDHW3
  vector<int64_t> x_shape = x_desc.GetShape().GetDims();        // NCDHW
  DataType x_dtype = x_desc.GetDataType();
  Format x_format = x_desc.GetFormat();

  if (x_shape.size() != 5) {
    OP_LOGE(TbeGetName(op).c_str(), "Expected dim of x should be 5. x dim is %ld.", x_shape.size());
    return GRAPH_FAILED;
  }

  if (grid_shape.size() != 5) {
    OP_LOGE(TbeGetName(op).c_str(), "Expected dim of grid should be 5. grid dim is %ld.", grid_shape.size());
    return GRAPH_FAILED;
  }

  if (grid_shape[4] != 3) {
    OP_LOGE(TbeGetName(op).c_str(), "Expected dim of last axis of grid should be 3. real value is %ld.", grid_shape[4]);
    return GRAPH_FAILED;
  }

  x_shape[2] = grid_shape[1];
  x_shape[3] = grid_shape[2];
  x_shape[4] = grid_shape[3];
  TensorDesc output_desc = op.GetOutputDescByName("y");
  output_desc.SetShape(ge::Shape(x_shape));
  output_desc.SetDataType(x_dtype);
  output_desc.SetFormat(x_format);
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(GridSampler3D, GridSampler3DInferShape);
// ----------------GridSampler3D END---------------------

// ---------------GridSampler3DGrad Op start-------------------
IMPLEMT_INFERFUNC(GridSampler3DGrad, GridSampler3DGradInferShape) {
  vector<int64_t> grad_shape = op.GetInputDescByName("grad").GetShape().GetDims();  // NCDHW
  TensorDesc x_desc = op.GetInputDescByName("x");
  TensorDesc grid_desc = op.GetInputDescByName("grid");
  vector<int64_t> grid_shape = grid_desc.GetShape().GetDims();
  vector<int64_t> x_shape = x_desc.GetShape().GetDims();

  if (x_shape.size() != 5) {
    OP_LOGE(TbeGetName(op).c_str(), "Expected dim of x should be 5. real value is %ld.", x_shape.size());
    return GRAPH_FAILED;
  }

  if (grid_shape.size() != 5) {
    OP_LOGE(TbeGetName(op).c_str(), "Expected dim of grid should be 5. real value is %ld.", grid_shape.size());
    return GRAPH_FAILED;
  }

  if (grad_shape.size() != 5) {
    OP_LOGE(TbeGetName(op).c_str(), "Expected dim of grad should be 5. real value is %ld.", grad_shape.size());
    return GRAPH_FAILED;
  }

  (void)op.UpdateOutputDesc("dx", x_desc);
  (void)op.UpdateOutputDesc("dgrid", grid_desc);
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(GridSampler3DGrad, GridSampler3DGradInferShape);
// ----------------GridSampler3DGrad END---------------------

// ---------------Upsample3dForward Op START-------------------
static bool Upasmple3dForwardInferShape(Operator& op) {
  AscendString op_name_str;
  op.GetName(op_name_str);
  const char *op_name = op_name_str.GetString();
  OP_LOGI(op_name, "Enter proto inferfunction!");
  TensorDesc input_desc = op.GetInputDescByName("x");
  auto input_shape_dims = input_desc.GetShape().GetDims();
  DataType input_dtype = input_desc.GetDataType();
  constexpr int THREEDIMS = 3;
  std::vector<int64_t> output_shape;
  
 if (input_dtype != DT_FLOAT16 && input_dtype != DT_FLOAT && input_dtype != DT_DOUBLE)
  {
    OP_LOGE(op_name, "input datatype must be float16,float or double!");
    return false;
  }
  
  if (input_shape_dims.size() != 5) {
    OP_LOGE(op_name, "Expected dim of input x should be 5. but get %ld.", input_shape_dims.size());
    return false;
  }
  output_shape.emplace_back(input_shape_dims[0]);
  output_shape.emplace_back(input_shape_dims[1]);

  std::vector<int64_t> output_size; 
  op.GetAttr("output_size", output_size);
  std::vector<float> scales; 
  op.GetAttr("scales", scales);

  if (!output_size.empty() && scales.empty())
  { 
    if (output_size.size() != THREEDIMS) {
      OP_LOGE(op_name,"attr::output_size dims must be 3, but get %ld.", output_size.size());
      return false;
    }
    output_shape.insert(output_shape.end(),output_size.begin(),output_size.end());
  } else if (output_size.empty() && !scales.empty()) {
    if (scales.size() != THREEDIMS) {
      OP_LOGE(op_name,"attr::scales dims must be 3, but get %ld.", scales.size());
      return false;
    }
    for (int i = 0; i < THREEDIMS; i++) {
      output_shape.emplace_back(int64_t(floor(input_shape_dims[i+2] * scales[i])));
    }
  } else {
    OP_LOGE(op_name,
            "only one of attr::output_size or attr::scales should be defined as a non-empty value.");
    return false;
  }

  Shape output_desc_shape(output_shape);
  TensorDesc output_desc_y = op.GetOutputDescByName("y");
  output_desc_y.SetShape(output_desc_shape);
  output_desc_y.SetDataType(input_dtype);
  op.UpdateOutputDesc("y", output_desc_y);
  return true;
}

// ---------------UpsampleNearest3d Op START-------------------
IMPLEMT_INFERFUNC(UpsampleNearest3d, UpsampleNearest3dInferShape) {
  if (Upasmple3dForwardInferShape(op)) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
INFER_FUNC_REG(UpsampleNearest3d, UpsampleNearest3dInferShape);
// ----------------UpsampleNearest3d END---------------------

// ---------------UpsampleTrilinear3d Op START-------------------
IMPLEMT_INFERFUNC(UpsampleTrilinear3d, UpsampleTrilinear3dInferShape) {
  if (Upasmple3dForwardInferShape(op)) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
INFER_FUNC_REG(UpsampleTrilinear3d, UpsampleTrilinear3dInferShape);
// ----------------UpsampleTrilinear3d END---------------------
// ---------------Upsample3dForward Op END-------------------

// ---------------Upsample3dBackward Op START-------------------
static bool Upsample3dBackwardInferShape(Operator& op) {
  AscendString op_name_str;
  op.GetName(op_name_str);
  const char *op_name = op_name_str.GetString();
  OP_LOGI(op_name, "Enter proto inferfunction!");
  TensorDesc inputDesc = op.GetInputDescByName("grad_output");
  auto input_dtype = inputDesc.GetDataType();
  auto input_shape_dims = inputDesc.GetShape().GetDims();
  constexpr int FIVEDIMS = 5;
  constexpr int THREEDIMS = 3;

 if (input_dtype != DT_FLOAT16 && input_dtype != DT_FLOAT && input_dtype != DT_DOUBLE)
  {
    OP_LOGE(op_name, "input datatype must be float16,float or double!");
    return false;
  }

  if (input_shape_dims.size() != FIVEDIMS) {
    OP_LOGE(op_name, "Expected dim of input x should be 5. but get %ld.", input_shape_dims.size());
    return false;
  }

  std::vector<int64_t> input_size;
  if (GRAPH_SUCCESS != op.GetAttr("input_size", input_size)) {
    OP_LOGE(op_name, "get attr::input_size faild!");
    return false;
  } 
  if (input_size.size() != FIVEDIMS) {
    OP_LOGE(op_name,"attr::input_size dims must be 5, but get %ld.", input_size.size());
    return false;
  }

  std::vector<int64_t> output_size; 
  op.GetAttr("output_size", output_size);
  std::vector<float> scales; 
  op.GetAttr("scales", scales);

  if (!output_size.empty() && scales.empty())
  { 
    if (output_size.size() != THREEDIMS) {
      OP_LOGE(op_name,"attr::output_size dims must be 3, but get %lu.", output_size.size());
      return false;
    }
    for (int i = 0; i < THREEDIMS; i++) {
      if (output_size[i] != input_shape_dims[i+2])
      {
        OP_LOGE(op_name,"attr::output_size[%d](get %ld) != input::grad_output_size[%d](get %ld).", 
                i, output_size[i], i+2, input_shape_dims[i+2]);
        return false;
      }
    }
  } else if (output_size.empty() && !scales.empty()) {
    if (scales.size() != THREEDIMS) {
      OP_LOGE(op_name,"attr::scales dims must be 3, but get %lu.", scales.size());
      return false;
    }
    for (int i = 0; i < THREEDIMS; i++) {
      int64_t tmp = int64_t(floor(input_size[i+2] * scales[i]));
      if (tmp != input_shape_dims[i+2])
      {
        OP_LOGE(op_name,"input_size[%d]*scales[%d](get %ld) != grad_output_size[%d](get %ld).", 
                i+2, i, tmp, i+2, input_shape_dims[i+2]);
        return false;
      }
    }
  } else {
    OP_LOGE(op_name,
            "only one of attr::output_size or attr::scales should be defined as a non-empty value.");
    return false;
  }

  Shape output_desc_shape(input_size);
  TensorDesc output_desc_y = op.GetOutputDescByName("y");
  output_desc_y.SetShape(output_desc_shape);
  output_desc_y.SetDataType(input_dtype);
  op.UpdateOutputDesc("y", output_desc_y);
  return true;
}
// ---------------Upsample3dBackward Op END------------------------

// ---------------UpsampleNearest3dGrad Op START-------------------
IMPLEMT_INFERFUNC(UpsampleNearest3dGrad, UpsampleNearest3dGradInferShape) {
  if (Upsample3dBackwardInferShape(op)) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
INFER_FUNC_REG(UpsampleNearest3dGrad, UpsampleNearest3dGradInferShape);
// ----------------UpsampleNearest3dGrad END---------------------

// ---------------UpsampleTrilinear3dGrad Op START-------------------
IMPLEMT_INFERFUNC(UpsampleTrilinear3dGrad, UpsampleTrilinear3dGradInferShape) {
  if (Upsample3dBackwardInferShape(op)) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
INFER_FUNC_REG(UpsampleTrilinear3dGrad, UpsampleTrilinear3dGradInferShape);
// ----------------UpsampleTrilinear3dGrad END---------------------

// ---------------UpsampleNearest1d Op START-------------------
IMPLEMT_INFERFUNC(UpsampleNearest1d, UpsampleNearest1dInferShape) {
  OP_LOGD(TbeGetName(op).c_str(), "Enter UpsampleNearest1d inferfunction!");
  TensorDesc input_desc = op.GetInputDesc("x");
  auto input_shape_dims = input_desc.GetShape().GetDims();
  DataType input_dtype = input_desc.GetDataType();
  
  std::vector<int64_t> output_shape = input_shape_dims;
  
  if (input_shape_dims.size() != 3) {
    OP_LOGE(TbeGetName(op).c_str(), "Expected dim of input x should be 3. but get %lu.", input_shape_dims.size());
    return GRAPH_FAILED;
  }

  std::vector<int64_t> output_size; 
  op.GetAttr("output_size", output_size);
  std::vector<float> scales; 
  op.GetAttr("scales", scales);
 
  if (!output_size.empty() && scales.empty())
  { 
    if (output_size.size() != 1) {
      OP_LOGE(TbeGetName(op).c_str(),"attr::output_size dims must be 1, but get %lu.", output_size.size());
      return GRAPH_FAILED;
    }
    output_shape[2] = output_size[0];
  } else if (output_size.empty() && !scales.empty()) {
    if (scales.size() != 1) {
      OP_LOGE(TbeGetName(op).c_str(),"attr::scales dims must be 1, but get %lu.", scales.size());
      return GRAPH_FAILED;
    }
    output_shape[2] = input_shape_dims[2] * scales[0];
  } else {
    OP_LOGE(TbeGetName(op).c_str(),
            "only one of attr::output_size or attr::scales should be defined as a non-empty value.");
    return GRAPH_FAILED;
  }

  Shape output_desc_shape(output_shape);
  TensorDesc output_desc_y = op.GetOutputDesc("y");
  output_desc_y.SetShape(output_desc_shape);
  output_desc_y.SetDataType(input_dtype);
  op.UpdateOutputDesc("y", output_desc_y);
  
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(UpsampleNearest1d, UpsampleNearest1dInferShape);
// ----------------UpsampleNearest1d END---------------------

// ---------------UpsampleNearest1dGrad Op START-------------------
IMPLEMT_INFERFUNC(UpsampleNearest1dGrad, UpsampleNearest1dGradInferShape) {
  OP_LOGD(TbeGetName(op).c_str(), "Enter UpsampleNearest1dGrad inferfunction!");
  TensorDesc inputDesc = op.GetInputDesc("grad_output");
  auto input_dtype = inputDesc.GetDataType();
  auto grad_output_dims = inputDesc.GetShape().GetDims();
 
  if (grad_output_dims.size() != 3) {
    OP_LOGE(TbeGetName(op).c_str(), "Expected dim of grad_output should be 3. but get %lu.", grad_output_dims.size());
    return GRAPH_FAILED;
  }

  std::vector<int64_t> input_size;
  if (GRAPH_SUCCESS != op.GetAttr("input_size", input_size)) {
    OP_LOGE(TbeGetName(op).c_str(), "get attr::input_size faild!");
    return GRAPH_FAILED;
  } 
  if (input_size.size() != 3) {
    OP_LOGE(TbeGetName(op).c_str(),"attr::input_size dims must be 3, but get %lu.", input_size.size());
    return GRAPH_FAILED;
  }

  Shape output_desc_shape(input_size);
  TensorDesc output_desc_y = op.GetOutputDesc("y");
  output_desc_y.SetShape(output_desc_shape);
  output_desc_y.SetDataType(input_dtype);
  op.UpdateOutputDesc("y", output_desc_y);

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(UpsampleNearest1dGrad, UpsampleNearest1dGradInferShape);
// ----------------UpsampleNearest1dGrad END---------------------

// ---------------EncodeJpegVariableQuality Op START-------------------
IMPLEMT_INFERFUNC(EncodeJpegVariableQuality, EncodeJpegVariableQualityInferShape) {
  OP_LOGD(TbeGetName(op).c_str(), "Enter EncodeJpegVariableQuality inferfunction!");
  Shape image_shape;
  if (WithRank(op.GetInputDesc(0), 3, image_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Expected dim of image should be 3. but get %lu.", image_shape.GetDimNum());
    return GRAPH_FAILED;
  }
  Shape quality_shape;
  if (WithRank(op.GetInputDesc(1), 0, quality_shape, TbeGetName(op).c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Expected dim of image should be 0. but get %lu.", quality_shape.GetDimNum());
    return GRAPH_FAILED;
  }

  Shape shape;
  (void)Scalar(shape);
  TensorDesc output_desc = op.GetOutputDesc(0);
  output_desc.SetShape(shape);
  output_desc.SetDataType(DT_STRING);
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(EncodeJpegVariableQuality, EncodeJpegVariableQualityInferShape);
// ----------------EncodeJpegVariableQuality END---------------------

// ---------------ImageProjectiveTransform Op START-------------------
IMPLEMT_INFERFUNC(ImageProjectiveTransform, ImageProjectiveTransformInferShape) {
  OP_LOGD(TbeGetName(op).c_str(), "Enter ImageProjectiveTransform inferfunction!");
  auto opname = TbeGetName(op).c_str();
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto image_descptr = op_desc->MutableInputDesc("images");
  GeShape image_shape;
  if (WithRank(image_descptr, 4, image_shape, opname) != GRAPH_SUCCESS) {
    OP_LOGE(opname, "Expected images should be 4-D. but get %lu.", image_descptr->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }
  
  auto outputshape_descptr = op_desc->MutableInputDesc("output_shape");
  GeShape outputshape_shape;
  if (WithRank(outputshape_descptr, 1, outputshape_shape, opname) != GRAPH_SUCCESS) {
    OP_LOGE(opname, "output_shape should be 1. but get %lu.", outputshape_descptr->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  int64_t unused_dim = 0;
  if (WithValue(outputshape_shape.GetDim(0), 2, unused_dim, opname) != GRAPH_SUCCESS) {
    OP_LOGE(opname, "output_shape's dim[0] should be 2. but get %lu.", outputshape_shape.GetDim(0));
    return GRAPH_FAILED;
  }
  
  std::string interpolation; // required attr
  if (op.GetAttr("interpolation", interpolation) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "get attr interpolation failed");
    return GRAPH_FAILED;
  }

  op_desc->SetOpInferDepends({"output_shape"});
  int64_t new_width = UNKNOWN_DIM;
  int64_t new_height = UNKNOWN_DIM;
  Tensor outputshape_tensor;
  if (op.GetInputConstData("output_shape", outputshape_tensor) == GRAPH_SUCCESS) {
    auto output_data = reinterpret_cast<const int32_t*>(outputshape_tensor.GetData());
    new_width = static_cast<int64_t>(output_data[0]);
    new_height = static_cast<int64_t>(output_data[1]);
  }
  auto output_descptr = op_desc->MutableOutputDesc("transformed_images");
  FillOpDesc(output_descptr,
             GeShape({image_shape.GetDim(0), new_height, new_width, image_shape.GetDim(3)}),
             image_descptr->GetDataType()); // NHWC
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(ImageProjectiveTransform, ImageProjectiveTransformInferShape);
// ----------------ImageProjectiveTransform END---------------------

std::vector<std::pair<int64_t, int64_t>> GetOpShapeRangeByAttr(Operator& op)
{
  std::vector<std::pair<int64_t, int64_t>> maxShapes;
  std::string maxShapeAttr;
  if (op.GetAttr(ge::ATTR_NAME_OP_MAX_SHAPE, maxShapeAttr) == GRAPH_SUCCESS) {
    const size_t maxShapeAttrSize = maxShapeAttr.size();
    size_t startIndex = 0;
    while (startIndex < maxShapeAttrSize) {
      int64_t maxShape = -1;
      size_t nextPos = 0;
      try
      {
        maxShape = std::stoll(maxShapeAttr.substr(startIndex, maxShapeAttrSize), &nextPos);
      }
      catch(const std::exception& e)
      {
        OP_LOGI(TbeGetName(op).c_str(), "ge::ATTR_NAME_OP_MAX_SHAPE[%s] is invalid", maxShapeAttr.c_str());
        break;
      }
      maxShapes.emplace_back(1, maxShape);
      startIndex = maxShapeAttr.find(',', startIndex + nextPos);
      if (startIndex == string::npos) {
        break;
      }
      startIndex++;
    }
  }
  return maxShapes;
}

graphStatus DecodeImageV3ShapeFn(Operator& op) {
  int channels;
  if (op.GetAttr("channels", channels) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), string("Get attr[chanels] failed"));
    return GRAPH_FAILED;
  }
  if (channels != 0 && channels != 1 && channels != 3 && channels != 4) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), string("attr[Channels] must be 0,1,3,or 4"));
    return GRAPH_FAILED;
  }

  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), string("Get attr[dtype] failed"));
    return GRAPH_FAILED;
  }

  bool expandAnimations = true;
  TensorDesc outputTensor = op.GetOutputDesc(0);
  op.GetAttr("expand_animations", expandAnimations);
  if (expandAnimations) {
    outputTensor.SetShape(Shape(ge::UNKNOWN_RANK));
    op.UpdateOutputDesc("image", outputTensor);
    return GRAPH_SUCCESS;
  }

  std::vector<int64_t> dims;
  if (channels == 0) {
    dims = {ge::UNKNOWN_DIM, ge::UNKNOWN_DIM, ge::UNKNOWN_DIM};
  } else {
    dims = {ge::UNKNOWN_DIM, ge::UNKNOWN_DIM, channels};
  }

  Shape outputShape(dims);
  outputTensor.SetDataType(dtype);
  outputTensor.SetShape(outputShape);

  std::vector<std::pair<int64_t, int64_t>> shapeRange = GetOpShapeRangeByAttr(op);
  if (shapeRange.size() == dims.size()) {
    outputTensor.SetShapeRange(shapeRange);
  }
  if (op.UpdateOutputDesc("image", outputTensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), string("Update OutputDesc[image] failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(DecodeImage, DecodeImageV3Infer) {
  return DecodeImageV3ShapeFn(op);
}

INFER_FUNC_REG(DecodeImage, DecodeImageV3Infer);

}  // namespace ge
