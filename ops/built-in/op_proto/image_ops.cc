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
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "axis_util.h"
#include "inc/graph/utils/type_utils.h"
namespace ge {
IMPLEMT_INFERFUNC(DecodeGif, DecodeGifInfer) {
  const char *op_name = op.GetName().c_str();
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
  if (WithRankAtLeast(images_desc, 3, out, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        0, DebugString(images_desc->GetShape().GetDims()), "at least 3D");
    err_msg = string("failed to call WithRankAtLeast function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
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
  if (WithRankAtLeast(images_desc, 3, out, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(
        0, DebugString(images_desc->GetShape().GetDims()), "at least 3D");
    err_msg = string("failed to call WithRankAtLeast function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
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
  if (WithRank(contrast_factor_desc, 0, shape, op.GetName().c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(
        1, DebugString(contrast_factor_desc->GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  auto images_desc = op_desc->MutableInputDesc(0);
  if (WithRankAtLeast(images_desc, 3, shape, op.GetName().c_str()) !=
      GRAPH_SUCCESS) {
    err_msg = GetShapeErrMsg(0, DebugString(images_desc->GetShape().GetDims()),
                             "at least 3D");
    err_msg = string("failed to call WithRankAtLeast function, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
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
  const char* op_name = op.GetName().c_str();
  GeShape x_shape;
  if (WithRank(x_desc, 4, x_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(x_desc->GetShape().GetDims()), "4D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
  }

  auto boxes_desc = op_desc->MutableInputDesc(1);
  GeShape boxes_shape;
  if (WithRank(boxes_desc, 2, boxes_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(boxes_desc->GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  auto box_index_desc = op_desc->MutableInputDesc(2);
  GeShape box_index_shape;
  if (WithRank(box_index_desc, 1, box_index_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2,
        DebugString(box_index_desc->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  auto crop_size_desc = op_desc->MutableInputDesc(3);
  GeShape crop_size_shape;
  if (WithRank(crop_size_desc, 1, crop_size_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(3,
        DebugString(crop_size_desc->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  auto x_dims = x_shape.GetDims();
  auto boxes_dims = boxes_shape.GetDims();
  auto box_index_dims = box_index_shape.GetDims();
  auto crop_size_dims = crop_size_shape.GetDims();

  CHECK(boxes_dims.empty() || box_index_dims.empty(),
        AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), string("the 0th input[x]'s shape and 1st input[boxes]'s shape"
                                           " should not be empty.")),
                                           return GRAPH_FAILED);
  if (boxes_dims[0] != UNKNOWN_DIM &&
      box_index_dims[0] != UNKNOWN_DIM &&
      boxes_dims[0] != box_index_dims[0]) {
      std::string err_msg = ConcatString(
          "the 0th dimension of the 1th input[boxes] and the 2nd input[box_index] must be equal. "
          , boxes_dims[0], " and " , box_index_dims[0]);
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  CHECK(crop_size_dims.empty(), AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), string("empty crop_size dim.")), return GRAPH_FAILED);
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
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
  y_desc->SetDataType(DT_FLOAT);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(CropAndResize, CropAndResizeInfer);

IMPLEMT_INFERFUNC(CropAndResizeGradBoxes, CropAndResizeGradBoxesInfer) {
  Shape shape;
  if (WithRank(op.GetInputDesc(0), 4, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(op.GetInputDesc(0).GetShape().GetDims()), "4D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 4, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(op.GetInputDesc(1).GetShape().GetDims()), "4D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(2), 2, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2,
        DebugString(op.GetInputDesc(2).GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(3), 1, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(3,
        DebugString(op.GetInputDesc(3).GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  auto grads_shape = op.GetInputDesc(0).GetShape().GetDims();
  auto boxes_shape = op.GetInputDesc(2).GetShape().GetDims();
  auto box_index_shape = op.GetInputDesc(3).GetShape().GetDims();

  if (grads_shape[0] != boxes_shape[0] && boxes_shape[0] != box_index_shape[0]) {
      std::string err_msg = ConcatString(
          "the 0th dimension of the 2th input[boxes], 0th input[grads] and the 3rd"
          " input [box_index] must be equal. ", grads_shape[0], ", " , boxes_shape[0] , " and " ,box_index_shape[0]);
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
  const char* op_name = op.GetName().c_str();
  if (WithRank(grads_desc, 4, grads_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(grads_desc->GetShape().GetDims()), "4D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  auto boxes_desc = op_desc->MutableInputDesc(1);
  GeShape boxes_shape;
  if (WithRank(boxes_desc, 2, boxes_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(boxes_desc->GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  auto box_index_desc = op_desc->MutableInputDesc(2);
  GeShape box_index_shape;
  if (WithRank(box_index_desc, 1, box_index_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2,
        DebugString(box_index_desc->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  auto image_size_desc = op_desc->MutableInputDesc(3);
  GeShape image_size_shape;
  if (WithRank(image_size_desc, 1, image_size_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(3,
        DebugString(image_size_desc->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  auto grads_dims = grads_shape.GetDims();
  auto boxes_dims = boxes_shape.GetDims();
  auto box_index_dims = box_index_shape.GetDims();
  CHECK(grads_dims.empty() || boxes_dims.empty() || box_index_dims.empty(),
        AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), string(
        "the 0th input[grads] , the 1st input[boxes] dims and the 2nd input[box_index], "
        "must not be empty.")),
        return GRAPH_FAILED);
  if (!DimsAllEqualOrUnknown({grads_dims[0], boxes_dims[0], box_index_dims[0]})) {
      std::string err_msg = ConcatString(
                                         "the 0th dimension of the 0th input[grads], the 1st input[boxes]"
                                         " and the 2nd input[box_index] must be equal. "
                                         , grads_dims[0], ", " , boxes_dims[0], " and ", box_index_dims[0]);
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
  }

  auto image_size_dims = image_size_shape.GetDims();
  CHECK(image_size_dims.empty(), AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), string("the 3rd input[image_size] dims must not be empty.")),
        return GRAPH_FAILED);
  if (image_size_dims[0] != 4 && image_size_dims[0] != UNKNOWN_DIM) {
      std::string err_msg = ConcatString(
          "the 3rd input[image_size] must be a 1-D tensor with 4 elements, current image_size is ", DebugString(image_size_dims));
    return GRAPH_FAILED;
  }

  DataType type;
  if (op.GetAttr("T", type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), string("get attr[T] failed"));
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
          AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), string("the 3rd input[image_size]'s data nums less then 4, curent data num is ",
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
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
  if (WithRank(op.GetInputDesc(0), 4, x_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input x must be 4-D");
    return GRAPH_PARAM_INVALID;
  }
  Shape offsets_shape;
  if (WithRank(op.GetInputDesc(2), 2, offsets_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input offsets must be 2-D");
    return GRAPH_PARAM_INVALID;
  }
  auto x_dims = op.GetInputDesc(0).GetShape().GetDims();
  auto offsets_dims = op.GetInputDesc(2).GetShape().GetDims();
  CHECK(x_dims.size() < 4 || offsets_dims.size() < 2, OP_LOGE(op.GetName().c_str(), "invalid x_dims or offsets_dims."),
        return GRAPH_FAILED);
  int64_t batch_dim;
  if (Merge(x_dims[0], offsets_dims[0], batch_dim) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "x dim-0 or offsets dim-0 is invalid");
    return GRAPH_PARAM_INVALID;
  }
  if (offsets_dims[1] != 2) {
    OP_LOGE(op.GetName().c_str(), "offsets dim-1 must be 2");
    return GRAPH_PARAM_INVALID;
  }

  bool uniform_noise = false;
  if (op.GetAttr("uniform_noise", uniform_noise) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr uniform_noise failed");
    return GRAPH_FAILED;
  }
  std::string noise;
  if (op.GetAttr("noise", noise) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr noise failed");
    return GRAPH_FAILED;
  }
  std::string info = "The uniform_noise and noise should not be specified ";
  if (uniform_noise && (!noise.empty() && noise != "uniform")) {
    OP_LOGE(op.GetName().c_str(), info + "at the same time");
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
  if (WithRank(op.GetInputDesc(2), 0, min_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input min must be a scalar");
    return GRAPH_FAILED;
  }

  Shape max_shape;
  if (WithRank(op.GetInputDesc(3), 0, max_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input max must be a scalar");
    return GRAPH_FAILED;
  }

  auto status = ResizeShapeFn(op, "images", "size", "resized_images");
  if (status != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "resize images shape failed");
    return GRAPH_FAILED;
  }

  TensorDesc y_min = op.GetOutputDesc("y_min");
  y_min.SetShape(Shape());
  y_min.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("y_min", y_min) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "fail to update output y_min.");
    return GRAPH_FAILED;
  }

  TensorDesc y_max = op.GetOutputDesc("y_max");
  y_max.SetShape(Shape());
  y_max.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("y_max", y_max) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "fail to update output y_max.");
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
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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

  const char* op_name = op.GetName().c_str();
  GeShape grads_shape;
  if (WithRank(grads_desc, 4, grads_shape, op_name) != GRAPH_SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), "Input grads must be 4-D, real rank is [%lld]", grads_desc->GetShape().GetDimNum());
    return GRAPH_PARAM_INVALID;
  }

  GeShape size_shape;
  if (WithRank(size_desc, 1, size_shape, op_name) != GRAPH_SUCCESS) {
    OP_LOGE(op_desc->GetName().c_str(), "Input size must be 1-D, real rank is [%lld]", size_desc->GetShape().GetDimNum());
    return GRAPH_PARAM_INVALID;
  }

  auto size_dims = size_shape.GetDims();
  if (size_dims[0] != 2 && size_dims[0] != UNKNOWN_DIM) {
    OP_LOGE(op_desc->GetName().c_str(), "Input size must be 1-D of 2 elements, real dim size is [%lld]", size_dims[0]);
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
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (size_out.size() != DIM_SIZE2) {
    std::string err_msg = GetAttrSizeErrMsg("size_out", ConcatString(size_out.size()), ConcatString(DIM_SIZE2));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
        op.GetName(), std::string("update output[y] desc failed"));
    return GRAPH_FAILED;
  }
  return ColorspaceShapeFn(op, "y");
}

INFER_FUNC_REG(RGBToHSV, RGBToHSVInfer);

IMPLEMT_INFERFUNC(SampleDistortedBoundingBox, SampleDistortedBoundingBoxInfer) {
  bool judge = false;

  Shape image_size;
  judge = (WithRank(op.get_input_desc_image_size(), 1, image_size, op.GetName().c_str()) != GRAPH_SUCCESS);
  if (judge) {
    std::string err_msg = ConcatString(
        "failed to call WithRank function, input[image_size] rank must be 1, "
        "got rank[", op.get_input_desc_image_size().GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  Shape bounding_boxes;
  judge = (WithRank(op.get_input_desc_bounding_boxes(), 3, bounding_boxes, op.GetName().c_str()) != GRAPH_SUCCESS);
  if (judge) {
    std::string err_msg = ConcatString(
        "failed to call WithRank function, input[bounding_boxes] rank must be 3, "
        "got rank[", op.get_input_desc_bounding_boxes().GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t image_size_unused_dim;
  int64_t bounding_boxes_unused_dim2;
  const int64_t kImageSizeDimValue = image_size.GetDim(0);
  const int64_t kBoundingBoxesDim2Value = bounding_boxes.GetDim(2);
  if (WithValue(kImageSizeDimValue, 3, image_size_unused_dim, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call WithValue function, input[image_size] first "
        "dimention must be 3, got dim[", kImageSizeDimValue, "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (WithValue(kBoundingBoxesDim2Value, 4, bounding_boxes_unused_dim2, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call WithValue function, input[bounding_boxes] third "
        "dimention must be 4, got dim[", kBoundingBoxesDim2Value, "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc begin_desc = op.GetOutputDesc("begin");
  begin_desc.SetShape(Shape({3}));
  begin_desc.SetDataType(op.GetInputDesc("image_size").GetDataType());
  if (op.UpdateOutputDesc("begin", begin_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("fail to update output[begin] desc."));
    return GRAPH_FAILED;
  }

  TensorDesc size_desc = op.GetOutputDesc("size");
  size_desc.SetShape(Shape({3}));
  size_desc.SetDataType(op.GetInputDesc("image_size").GetDataType());
  if (op.UpdateOutputDesc("size", size_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("fail to update output[size] desc."));
    return GRAPH_FAILED;
  }

  TensorDesc bboxes_desc = op.GetOutputDesc("bboxes");
  bboxes_desc.SetShape(Shape({1, 1, 4}));
  bboxes_desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("bboxes", bboxes_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("fail to update output[bboxes] desc."));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SampleDistortedBoundingBox, SampleDistortedBoundingBoxInfer);

IMPLEMT_INFERFUNC(SampleDistortedBoundingBoxExt2, SampleDistortedBoundingBoxExt2Infer) {
  bool judge = false;

  Shape image_size;
  judge = (WithRank(op.get_input_desc_image_size(), 1, image_size, op.GetName().c_str()) != GRAPH_SUCCESS);
  if (judge) {
    std::string err_msg = ConcatString(
        "failed to call WithRank function, input[image_size] rank must be 1 ,"
        "got rank[", op.get_input_desc_image_size().GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  Shape bounding_boxes;
  judge = (WithRank(op.get_input_desc_bounding_boxes(), 3, bounding_boxes, op.GetName().c_str()) != GRAPH_SUCCESS);
  if (judge) {
    std::string err_msg = ConcatString(
        "failed to call WithRank function, input[bounding_boxes] rank must be 3 ,"
        "got rank[", op.get_input_desc_image_size().GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  Shape min_object_covered;
  judge =
      (WithRank(op.get_input_desc_min_object_covered(), 0, min_object_covered, op.GetName().c_str()) != GRAPH_SUCCESS);
  if (judge) {
    std::string err_msg = ConcatString(
        "failed to call WithRank function, input[min_object_covered] rank must "
        "be scalar, got rank[",
        op.get_input_desc_image_size().GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  const int64_t image_size_dim_value = op.get_input_desc_image_size().GetShape().GetDim(0);
  const int64_t bounding_boxes_dim2_value = op.get_input_desc_bounding_boxes().GetShape().GetDim(2);
  if ((image_size_dim_value != 3) || (bounding_boxes_dim2_value != 4)) {
    std::string err_msg = ConcatString(
        "0th dim of input[image_size] must be 3, got[", image_size_dim_value,
        "] and 2nd dim of input[bounding_boxes] must be 4, got[",
        bounding_boxes_dim2_value, "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc begin_desc = op.GetOutputDesc("begin");
  begin_desc.SetShape(Shape({3}));
  begin_desc.SetDataType(op.GetInputDesc("image_size").GetDataType());
  if (op.UpdateOutputDesc("begin", begin_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("fail to update output[begin] desc."));
    return GRAPH_FAILED;
  }

  TensorDesc size_desc = op.GetOutputDesc("size");
  size_desc.SetShape(Shape({3}));
  size_desc.SetDataType(op.GetInputDesc("image_size").GetDataType());
  if (op.UpdateOutputDesc("size", size_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("fail to update output[size] desc."));
    return GRAPH_FAILED;
  }

  TensorDesc bboxes_desc = op.GetOutputDesc("bboxes");
  bboxes_desc.SetShape(Shape({1, 1, 4}));
  bboxes_desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("bboxes", bboxes_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("fail to update output[bboxes] desc."));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SampleDistortedBoundingBoxExt2, SampleDistortedBoundingBoxExt2Infer);

IMPLEMT_INFERFUNC(DrawBoundingBoxes, DrawBoundingBoxesInfer) {
  Shape images;

  if (WithRank(op.GetInputDesc(0), 4, images, op.GetName().c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
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
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
          ConcatString("invalid 3th dim[", depth, "] of input[images], should be 1, 3 or 4"));
      return GRAPH_FAILED;
    }
  }

  Shape boxes;
  if (WithRank(op.GetInputDesc(1), 3, boxes, op.GetName().c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
        ConcatString("call WithRank function failed, ",
            GetShapeErrMsg(1, DebugString(op.GetInputDesc(1).GetShape().GetDims()), "3D")));
    return GRAPH_FAILED;
  }
  if (boxes.GetDim(2) != 4) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        ConcatString("invalid 2th dim[", boxes.GetDim(2),
            "] of input[boxes], should 4"));
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
  const char* op_name = op.GetName().c_str();
  auto boxes_desc = op_desc->MutableInputDesc(0);
  if (WithRank(boxes_desc, 2, boxes_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(boxes_desc->GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  GeShape scores_shape;
  auto scores_desc = op_desc->MutableInputDesc(1);
  if (WithRank(scores_desc, 1, scores_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(scores_desc->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  GeShape max_output_size_shape;
  auto max_output_size_desc = op_desc->MutableInputDesc(2);
  if (WithRank(max_output_size_desc, 0, max_output_size_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2,
        DebugString(max_output_size_desc->GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t unused_dim;
  if (Merge(boxes_shape.GetDim(0), scores_shape.GetDim(0), unused_dim) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call Merge function, 0th dim[",
        boxes_shape.GetDim(0), "] of input[boxes] not equal 0th dim[",
        scores_shape.GetDim(0), "] of input[scores]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (boxes_shape.GetDim(1) != 4 && boxes_shape.GetDim(1) != UNKNOWN_DIM) {
    std::string err_msg = ConcatString(
        "0th dim[", boxes_shape.GetDim(1), "] of input[boxes] not equal 4");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  auto selected_indices_desc = op_desc->MutableOutputDesc("selected_indices");
  selected_indices_desc->SetShape(GeShape({UNKNOWN_DIM}));
  selected_indices_desc->SetShapeRange({std::pair<int64_t, int64_t>(1, -1)});
  selected_indices_desc->SetDataType(DT_INT32);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(NonMaxSuppression, NonMaxSuppressionInfer);

IMPLEMT_INFERFUNC(NonMaxSuppressionV2, NonMaxSuppressionV2Infer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  GeShape boxes_shape;
  auto boxes_desc = op_desc->MutableInputDesc(0);
  const char* op_name = op.GetName().c_str();
  if (WithRank(boxes_desc, 2, boxes_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(boxes_desc->GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  GeShape scores_shape;
  auto scores_desc = op_desc->MutableInputDesc(1);
  if (WithRank(scores_desc, 1, scores_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(scores_desc->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  GeShape max_output_size_shape;
  auto max_output_size_desc = op_desc->MutableInputDesc(2);
  if (WithRank(max_output_size_desc, 0, max_output_size_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2,
        DebugString(max_output_size_desc->GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  GeShape iou_threshold_shape;
  auto iou_threshold_desc = op_desc->MutableInputDesc(3);
  if (WithRank(iou_threshold_desc, 0, iou_threshold_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(3,
        DebugString(iou_threshold_desc->GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t unused_dim;
  if (Merge(boxes_shape.GetDim(0), scores_shape.GetDim(0), unused_dim) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call Merge function, 0th dim[",
        boxes_shape.GetDim(0), "] of input[boxes] not equal 0th dim[",
        scores_shape.GetDim(0), "] of input[scores]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (boxes_shape.GetDim(1) != 4 && boxes_shape.GetDim(1) != UNKNOWN_DIM) {
    std::string err_msg = ConcatString("1th dim[", boxes_shape.GetDim(1), "] of input[boxes] not equal 4.");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  auto selected_indices_desc = op_desc->MutableOutputDesc("selected_indices");
  selected_indices_desc->SetShape(GeShape({UNKNOWN_DIM}));
  selected_indices_desc->SetShapeRange({std::pair<int64_t, int64_t>(1, -1)});
  selected_indices_desc->SetDataType(DT_INT32);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(NonMaxSuppressionV2, NonMaxSuppressionV2Infer);

IMPLEMT_INFERFUNC(NonMaxSuppressionV3, NonMaxSuppressionV3Infer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  GeShape boxes_shape;
  auto boxes_desc = op_desc->MutableInputDesc(0);
  const char* op_name = op.GetName().c_str();
  if (WithRank(boxes_desc, 2, boxes_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(boxes_desc->GetShape().GetDims()), "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  GeShape scores_shape;
  auto scores_desc = op_desc->MutableInputDesc(1);
  if (WithRank(scores_desc, 1, scores_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
        DebugString(scores_desc->GetShape().GetDims()), "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
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
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
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
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  GeShape score_threshold_shape;
  auto score_threshold_desc = op_desc->MutableInputDesc(4);
  if (WithRank(score_threshold_desc, 0, score_threshold_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(4,
        DebugString(score_threshold_desc->GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t unused_dim;
  if (Merge(boxes_shape.GetDim(0), scores_shape.GetDim(0), unused_dim) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call Merge function, 0th dim[",
        boxes_shape.GetDim(0), "] of input[boxes] not equal 0th dim[",
        scores_shape.GetDim(0), "] of input[scores]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (boxes_shape.GetDim(1) != 4 && boxes_shape.GetDim(1) != UNKNOWN_DIM) {
    std::string err_msg = ConcatString(
        "1th dim[", boxes_shape.GetDim(1), "] of input[boxes] not equal 4.");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  auto selected_indices_desc = op_desc->MutableOutputDesc(0);
  selected_indices_desc->SetDataType(DT_INT32);
  selected_indices_desc->SetShape(GeShape({UNKNOWN_DIM}));
  selected_indices_desc->SetShapeRange({std::pair<int64_t, int64_t>(1, -1)});

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(NonMaxSuppressionV3, NonMaxSuppressionV3Infer);

IMPLEMT_INFERFUNC(NonMaxSuppressionV4, NonMaxSuppressionV4Infer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  const char *op_name = op.GetName().c_str();
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
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }


  GeShape scores_shape;
  auto scores_desc = op_desc->MutableInputDesc(1);
  if (WithRank(scores_desc, 1, scores_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1,
                                         DebugString(scores_desc->GetShape().GetDims()), 
                                         "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  GeShape max_output_size_shape;
  auto max_output_size_desc = op_desc->MutableInputDesc(2);
  if (WithRank(max_output_size_desc, 0, max_output_size_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2, 
                                         DebugString(max_output_size_desc->GetShape().GetDims()),
                                         "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  GeShape iou_threshold_shape;
  auto iou_threshold_desc = op_desc->MutableInputDesc(3);
  if (WithRank(iou_threshold_desc, 0, iou_threshold_shape, op_name) != GRAPH_SUCCESS) {
     std::string err_msg = GetShapeErrMsg(3,
                                          DebugString(iou_threshold_desc->GetShape().GetDims()), 
                                          "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  GeShape score_threshold_shape;
  auto score_threshold_desc = op_desc->MutableInputDesc(4);
  if (WithRank(score_threshold_desc, 0, score_threshold_shape, op_name) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(4, 
                                         DebugString(score_threshold_desc->GetShape().GetDims()), 
                                         "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t unused_dim;
  if (Merge(boxes_shape.GetDim(0), scores_shape.GetDim(0), unused_dim) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call Merge function, 0th dim[",
                                       boxes_shape.GetDim(0), "] of input[boxes] not equal 0th dim[",
                                       scores_shape.GetDim(0), "] of input[scores]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (WithValue(boxes_shape.GetDim(1), 4, unused_dim, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call WithValue function, 1th dim[",
                                       boxes_shape.GetDim(1), "] of input[boxes] not equal 4");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> selected_indices_dims{UNKNOWN_DIM};
  bool pad_to_max = false;
  if (op.GetAttr("pad_to_max_output_size", pad_to_max) != ge::GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
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
        AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
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
  if (WithRank(op.GetInputDesc("overlaps"), 2, overlaps_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call WithRank function, ",
      "input[overlaps] rank must be 2, but got rank[",
      op.GetInputDesc("overlaps").GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc("scores"), 1, scores_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call WithRank function, ",
      "input[scores] rank must be 1, but got rank[",
      op.GetInputDesc("scores").GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc("max_output_size"), 0, max_output_size_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call WithRank function, ",
      "input[max_output_size] rank must be 0, but got rank[",
      op.GetInputDesc("max_output_size").GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc("overlap_threshold"), 0, overlap_threshold_shape, op.GetName().c_str()) !=
      GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call WithRank function, ",
      "input[overlap_threshold] rank must be 0, but got rank[",
      op.GetInputDesc("overlap_threshold").GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc("score_threshold"), 0, score_threshold_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call WithRank function, ",
      "input[score_threshold] rank must be 0, but got rank[",
      op.GetInputDesc("score_threshold").GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  int64_t unused_dim = 0;
  if (Merge(overlaps_shape.GetDim(0), scores_shape.GetDim(0), unused_dim) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call Merge function to merge the input[overlaps] 0th dim",
        "[" , overlaps_shape.GetDim(0), "] and the input[scores]'s 0th dim [", 
        scores_shape.GetDim(0), "]");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (Merge(overlaps_shape.GetDim(0), overlaps_shape.GetDim(1), unused_dim) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString(
        "failed to call Merge function to merge the input[overlaps] 0th dim",
        "[" , overlaps_shape.GetDim(0), "] and the input[overlaps]'s 1th dim [", 
        overlaps_shape.GetDim(1), "]");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc selected_indices_desc = op.GetOutputDesc("selected_indices");
  Shape selecte_indices_shape;
  Vector(ge::UNKNOWN_DIM, selecte_indices_shape);
  selected_indices_desc.SetDataType(DT_INT32);
  selected_indices_desc.SetShape(selecte_indices_shape);
  if (op.UpdateOutputDesc("selected_indices", selected_indices_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
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
        op.GetName(), string("get op desc failed, op desc is nullptr."));
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
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t channels;
  if (op.GetAttr("channels", channels) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
                                       string("get attr[channels] failed."));
    return GRAPH_FAILED;
  }
  if (channels != 0 && channels != 1 && channels != 3 && channels != 4) {
    err_msg =
        ConcatString("attr[channels] must be 0, 1, 3, 4, got [", channels, "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (channels == 0) {
    channels = UNKNOWN_DIM;
    OP_LOGI(op.GetName().c_str(), "attr[channels] is 0, use unknowdim");
  }
  image.SetDataType(DT_UINT8);
  std::vector<int64_t> image_shape({UNKNOWN_DIM, UNKNOWN_DIM, channels});
  image.SetShape(Shape(image_shape));
  if (op.UpdateOutputDesc("image", image) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
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
        op.GetName(), string("get op desc failed, op desc is nullptr."));
    return GRAPH_FAILED;
  }

  // unknown shape support
  op_desc->SetOpInferDepends({"crop_window"});

  GeShape contents_shape;
  auto contents_desc = op_desc->MutableInputDesc(0);
  if (contents_desc == nullptr) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("get input[contents] desc failed, input[contents] "
                             "desc is nullptr."));
    return GRAPH_FAILED;
  }
  std::string err_msg;
  if (WithRank(contents_desc, 0, contents_shape, op.GetName().c_str())
      != GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[contents] rank must be 0, got "
        "rank[",
        contents_desc->GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_PARAM_INVALID;
  }

  int64_t channels_dim = UNKNOWN_DIM;
  int64_t height = UNKNOWN_DIM;
  int64_t width = UNKNOWN_DIM;

  int32_t channels;
  if (op.GetAttr("channels", channels) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
                                       string("failed to get attr[channels]."));
    return GRAPH_PARAM_INVALID;
  }
  if (channels != 0) {
    if (channels < 0) {
      err_msg = ConcatString("attr[channels] must be non-negative, got[",
                             channels, "]");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_PARAM_INVALID;
    }
    channels_dim = channels;
  }

  GeShape crop_window_shape;
  auto crop_window_desc = op_desc->MutableInputDesc(1);
  if (crop_window_desc == nullptr) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        op.GetName(), string("get input[crop_window] desc failed, "
                             "input[crop_window] desc is nullptr."));
    return GRAPH_FAILED;
  }
  if (WithRank(crop_window_desc, 1, crop_window_shape, op.GetName().c_str())
      != GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithRank function, input[crop_window] rank must be 1, "
        "got rank[",
        crop_window_desc->GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_PARAM_INVALID;
  }
  int64_t unused_dim;
  if (WithValue(crop_window_shape.GetDim(0), 4, unused_dim,
                op_desc->GetName().c_str()) != GRAPH_SUCCESS) {
    err_msg = ConcatString(
        "failed to call WithValue function, dim[0] of input[crop_window] must "
        "be 4, got[",
        crop_window_shape.GetDim(0), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
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

static void GetResizeConstValue(const Operator& op, const GeTensorPtr& const_tensor,
                                const DataType& dtype, std::vector<int64_t>& const_data) {
  size_t size = const_tensor->GetData().GetSize();
  void* data_ptr = (void*)const_tensor->GetData().GetData();
  if (data_ptr == nullptr) {
    return;
  }

  if (dtype == ge::DT_INT32){
    int32_t* const_data_ptr = reinterpret_cast<int32_t*>(data_ptr);
    size = size / sizeof(int32_t);
    for (size_t i=0; i < size; i++) {
      const_data.push_back((int64_t)((int32_t) ((*(const_data_ptr + i)))));
    }
  } else if (dtype == ge::DT_INT64) {
    int64_t* const_data_ptr = reinterpret_cast<int64_t*>(data_ptr);
    size = size / sizeof(int64_t);
    for (size_t i=0; i < size; i++) {
      const_data.push_back((int64_t)((int64_t) ((*(const_data_ptr + i)))));
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "resize const not support the type");
  }
}

bool ResizeConstInferShape(const Operator& op, const string& image_name,
                           const string& size_name, const string& output_name) {
  auto node = NodeUtils::GetNodeFromOperator(op);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_desc == nullptr, OP_LOGE(op.GetName().c_str(), "op desc is null."), return false);

  auto input_desc_x = op_desc->MutableInputDesc(image_name);
  auto input_desc_size = op_desc->MutableInputDesc(size_name);
  auto output_desc_y = op_desc->MutableOutputDesc(output_name);
  auto image_shape = input_desc_x->MutableShape().GetDims();
  auto input_format = input_desc_x->GetFormat();
  auto input_size_dtype = input_desc_size->GetDataType();

  std::vector<std::pair<int64_t, int64_t>> x_range;
  // check whether is -2 case
  bool is_unkown_rank = image_shape == UNKNOWN_RANK ? true : false;
  if (is_unkown_rank) {
    OP_LOGW(op.GetName().c_str(), "the input os unkown rank, will set the input -1, -1, -1 , -1");
    image_shape = {-1, -1, -1, -1};
  } else {
    input_desc_x->GetShapeRange(x_range);
  }
  MakeUpShapeRange(image_shape, x_range);

  GeTensorPtr size_tensor = nullptr;
  vector<int64_t> size_out;
  std::vector<std::pair<int64_t, int64_t>> output_range;
  if (NodeUtils::GetInputConstData(node, size_name, size_tensor) != GRAPH_SUCCESS) {
    OP_LOGW(op.GetName().c_str(), "get sise const value failed, will set output h w = [-1, -1]");
    size_out.push_back(-1);
    size_out.push_back(-1);
    output_range.push_back(std::pair<int64_t, int64_t>{1, -1});
    output_range.push_back(std::pair<int64_t, int64_t>{1, -1});
  } else {
    GetResizeConstValue(op, size_tensor, input_size_dtype, size_out);
    output_range.push_back(std::pair<int64_t, int64_t>{size_out[0], size_out[0]});
    output_range.push_back(std::pair<int64_t, int64_t>{size_out[1], size_out[1]});
  }

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
    OP_LOGE(op.GetName().c_str(), "Not supported this format %d", input_format);
    return false;
  }

  output_desc_y->SetShape(GeShape(y_shape));
  output_desc_y->SetOriginShape(GeShape(y_shape));
  auto input_dtype = input_desc_x->GetDataType();
  output_desc_y->SetDataType(input_dtype);
  output_desc_y->SetShapeRange(result_range);

  return true;
}

IMPLEMT_COMMON_INFERFUNC(ResizeInferShape) {
  vector<int64_t> images_shape = op.GetInputDesc("x").GetShape().GetDims();
  vector<int64_t> size_out;
  if (op.GetAttr("size", size_out) == ge::GRAPH_FAILED) {
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue size failed!");
    return GRAPH_FAILED;
  }

  if (size_out.size() != DIM_SIZE2) {
    OP_LOGE(op.GetName().c_str(), "length of size_out must be equal to 2");
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
    OP_LOGE(op.GetName().c_str(), "Not supported this format");
  }
  td.SetShape(ge::Shape(y_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

// ---------------ResizeBilinearV2 Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(ResizeBilinearV2InferShape) {
  const vector<string> depend_names = {"size"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  if (!ResizeConstInferShape(op, "x", "size", "y")) {
    return GRAPH_FAILED;
  }

  auto op_desc_info = OpDescUtils::GetOpDescFromOperator(op);
  auto output_desc_y = op_desc_info->MutableOutputDesc("y");
  output_desc_y->SetDataType(DT_FLOAT);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ResizeBilinearV2, ResizeBilinearV2InferShape);
// ---------------ResizeBilinearV2 Op End-------------------

// ---------------ResizeBilinearV2D Op Start-------------------
IMPLEMT_COMMON_INFERFUNC(ResizeBilinearV2DInferShape) {

  vector<int64_t> images_shape = op.GetInputDesc("x").GetShape().GetDims();
  vector<int64_t> size_out;
  if (op.GetAttr("size", size_out) == ge::GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "size");
    std::string err_msg = GetInputInvalidErrMsg("size");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (size_out.size() != DIM_SIZE2) {
    std::string err_msg = GetAttrSizeErrMsg("size_out", ConcatString(size_out.size()), ConcatString(DIM_SIZE2));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
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
    OP_LOGE(op.GetName().c_str(), "get attr min_dimension failed");
    return GRAPH_FAILED;
  }
  std::int64_t maxDims = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("max_dimension", maxDims)) {
    OP_LOGE(op.GetName().c_str(), "get attr max_dimension failed");
    return GRAPH_FAILED;
  }
  CHECK(minDims == 0 || maxDims == 0, OP_LOGE(op.GetName().c_str(), "min_dimension and max_dimension should not be 0."),
        return GRAPH_FAILED);
  float minDimsFloat = static_cast<float>(minDims);
  float maxDimsFloat = static_cast<float>(maxDims);
  std::int64_t batchDIms = 0;
  std::int64_t heightDIms = 0;
  std::int64_t widthDims = 0;
  std::int64_t channelDIms = 0;
  auto inputImagesShape = op.GetInputDesc("images").GetShape().GetDims();

  if (inputImagesShape.size() != DIM_SIZE4) {
    OP_LOGE(op.GetName().c_str(), "length of size_out must be equal to 4");
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
    OpsInputFormatErrReport(op.GetName(), "images", expectedFormatList, ConcatString(inputFormat));
    OP_LOGE(op.GetName().c_str(), "Not supported this format");
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
  const vector<string> depend_names = {"size"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  if (!ResizeConstInferShape(op, "x", "size", "y")) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(ResizeNearestNeighborV2, ResizeNearestNeighborV2InferShape);
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
    OP_LOGW(op.GetName().c_str(), "the input os unkown rank, will set the input -1, -1, -1 , -1");
    grads_shape = {-1, -1, -1, -1};
  } else {
    input_desc_grad->GetShapeRange(grads_range);
  }
  if (is_unkown_rank_images) {
    OP_LOGW(op.GetName().c_str(), "the input os unkown rank, will set the input -1, -1, -1 , -1");
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
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
  }
  output_desc_y->SetShape(GeShape(y_shape));
  output_desc_y->SetOriginShape(GeShape(y_shape));
  output_desc_y->SetShapeRange(y_range);
  output_desc_y->SetDataType(input_dtype);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ResizeBilinearV2Grad, ResizeBilinearV2GradInfer);
// ---------------ResizeBilinearV2Grad Op End-------------------

IMPLEMT_INFERFUNC(EncodeJpeg, EncodeJpegInfer) {
  return EncodeImageShapeFn(op);
}

INFER_FUNC_REG(EncodeJpeg, EncodeJpegInfer);

IMPLEMT_INFERFUNC(ExtractJpegShape, ExtractJpegShapeInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc(0), 0, unused_shape, op.GetName().c_str())
      != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0,
        DebugString(op.GetInputDesc(0).GetShape().GetDims()), "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  DataType output_type;
  if (op.GetAttr("output_type", output_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
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
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
                                      string("update output[image_shape] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ExtractJpegShape, ExtractJpegShapeInfer);

IMPLEMT_INFERFUNC(DrawBoundingBoxesV2, DrawBoundingBoxesV2Infer) {
  auto imagesTensor = op.get_input_desc_images();

  Shape images;
  if (WithRankAtLeast(imagesTensor, 3, images, op.GetName().c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
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

  if (WithRank(op.GetInputDesc(0), 2, boxes, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(0, 
                                         DebugString(op.GetInputDesc(0).GetShape().GetDims()),
                                         "2D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(1), 1, scores, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(1, 
                                         DebugString(op.GetInputDesc(1).GetShape().GetDims()),
                                         "1D");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(2), 0, max_output_size, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(2, 
                                         DebugString(op.GetInputDesc(2).GetShape().GetDims()), 
                                         "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(3), 0, iouThreshold, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(3, 
                                         DebugString(op.GetInputDesc(3).GetShape().GetDims()),
                                         "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(4), 0, scoreThreshold, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(4, 
                                         DebugString(op.GetInputDesc(4).GetShape().GetDims()), 
                                         "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(5), 0, softNmsSigma, op.GetName().c_str()) != GRAPH_SUCCESS) {
    std::string err_msg = GetShapeErrMsg(5, 
                                         DebugString(op.GetInputDesc(5).GetShape().GetDims()), 
                                         "scalar");
    err_msg = string("failed to call WithRank, ") + err_msg;
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  int64_t un_used;

  if (Merge(boxes.GetDim(0), scores.GetDim(0), un_used) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call Merge function, 0th dim[",
                                       boxes.GetDim(0), "] of input[boxes] not equal 0th dim[",
                                       scores.GetDim(0), "] of input[scores]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  if (boxes.GetDim(1) != 4) {
    if (boxes.GetDim(1) != UNKNOWN_DIM) {
      std::string err_msg = ConcatString("1th dim[", 
                                         boxes.GetDim(1), 
                                         "] of input[boxes] not equal 4 or -1.");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
  }

  bool pad_to_max;
  if (ge::GRAPH_SUCCESS != op.GetAttr("pad_to_max_output_size", pad_to_max)) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), string("get attr[pad_to_max_output_size] failed"));
    return GRAPH_FAILED;
  }

  TensorDesc out_desc = op.GetOutputDesc("selected_indices");
  TensorDesc out_desc_scores = op.GetOutputDesc("selected_scores");
  out_desc.SetDataType(DT_INT32);
  DataType type;
  if (op.GetAttr("T", type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), string("get attr[T] failed"));
    return GRAPH_FAILED;
  }
  out_desc_scores.SetDataType(type);

  if (!pad_to_max) {
    out_desc.SetShape(Shape({ge::UNKNOWN_DIM}));
    out_desc_scores.SetShape(Shape({ge::UNKNOWN_DIM}));
    if (op.UpdateOutputDesc("selected_indices", out_desc) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), string("update description for output[selected_indices] failed"));
      return GRAPH_FAILED;
    }
    if (op.UpdateOutputDesc("selected_scores", out_desc_scores) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), string("update description for output[selected_scores] failed"));
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
        AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(), err_msg);
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
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
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
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    if (org_images_shape.size() < 2) {
      std::string err_msg = ConcatString(
        "the 1th input[original_images]'s rank should not be less than 2, ",
        "current rank is ", org_images_shape.size());
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
      return GRAPH_FAILED;
    }
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(org_images_shape[1]);
    y_shape.push_back(org_images_shape[2]);
    y_shape.push_back(grads_shape[3]);
    OP_LOGI(op.GetName().c_str(), "Real format is %d", input_format);
  }

  desc.SetShape(ge::Shape(y_shape));
  desc.SetDataType(DT_FLOAT);
  return op.UpdateOutputDesc("y", desc);
}

INFER_FUNC_REG(ScaleAndTranslateGrad, ScaleAndTranslateGradInfer);

// ---------------IMGWarp Op start-------------------
IMPLEMT_COMMON_INFERFUNC(IMGWarpInferShape) {
  OP_LOGI(op.GetName().c_str(), "start to infershape for IMGWarp.");
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_info == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), OtherErrMsg("invalid OpDesc.")), return GRAPH_FAILED);
  auto image_desc = op_info->MutableInputDesc("img");
  auto offset_desc = op_info->MutableInputDesc("warp_offset");
  auto image_dtype = image_desc->GetDataType();
  vector<int64_t> image_shape = image_desc->MutableShape().GetDims();
  vector<int64_t> offset_shape = offset_desc->MutableShape().GetDims();

  // check image_shape//offset_shape must be 4dims
  if (image_shape.size() != DIM_SIZE4) {
    std::string err_msg = GetAttrSizeErrMsg("img", ConcatString(image_shape.size()), ConcatString(DIM_SIZE4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (offset_shape.size() != DIM_SIZE4) {
    std::string err_msg = GetAttrSizeErrMsg("warp_offset", ConcatString(offset_shape.size()), ConcatString(DIM_SIZE4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  vector<int64_t> output_shape = image_shape;
  output_shape[2] = offset_shape[2];
  output_shape[3] = offset_shape[3];
  auto output_desc = op_info->MutableOutputDesc("warp_img");
  output_desc->SetShape(GeShape(output_shape));
  output_desc->SetOriginShape(GeShape(output_shape));
  output_desc->SetDataType(image_dtype);
  OP_LOGI(op.GetName().c_str(), "end to infershape for IMGWarp.");
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(IMGWarp, IMGWarpInferShape);
// ----------------IMGWarp END---------------------

// ---------------Remap Op start-------------------
IMPLEMT_COMMON_INFERFUNC(RemapInferShape) {
  OP_LOGI(op.GetName().c_str(), "start to infershape for Remap.");
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_info == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), OtherErrMsg("invalid OpDesc.")), return GRAPH_FAILED);
  auto image_desc = op_info->MutableInputDesc("img");
  auto offset_desc = op_info->MutableInputDesc("map_offset");
  auto image_dtype = image_desc->GetDataType();
  vector<int64_t> image_shape = image_desc->MutableShape().GetDims();
  vector<int64_t> offset_shape = offset_desc->MutableShape().GetDims();

  // check image_shape//offset_shape must be 4dims
  if (image_shape.size() != DIM_SIZE4) {
    std::string err_msg = GetAttrSizeErrMsg("img", ConcatString(image_shape.size()), ConcatString(DIM_SIZE4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }
  if (offset_shape.size() != DIM_SIZE4) {
    std::string err_msg = GetAttrSizeErrMsg("map_offset", ConcatString(offset_shape.size()), ConcatString(DIM_SIZE4));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(), err_msg);
    return GRAPH_FAILED;
  }

  vector<int64_t> output_shape = image_shape;
  output_shape[1] = offset_shape[1];
  output_shape[2] = offset_shape[2];
  auto output_desc = op_info->MutableOutputDesc("map_img");
  output_desc->SetShape(GeShape(output_shape));
  output_desc->SetOriginShape(GeShape(output_shape));
  output_desc->SetDataType(image_dtype);
  OP_LOGI(op.GetName().c_str(), "end to infershape for Remap.");
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(Remap, RemapInferShape);
// ----------------Remap END---------------------

IMPLEMT_INFERFUNC(CombinedNonMaxSuppression, CombinedNonMaxSuppressionInfer) {
  Shape boxes;
  Shape scores;
  Shape max_output_size_per_class;
  Shape max_total_size;
  Shape unused_shape;

  if (WithRank(op.GetInputDesc(0), 4, boxes, op.GetName().c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
        GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "4D"));
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 3, scores, op.GetName().c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
        GetShapeErrMsg(1, DebugString(op.GetInputDesc(1).GetShape().GetDims()), "3D"));
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(2), 0, max_output_size_per_class, op.GetName().c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
        GetShapeErrMsg(2, DebugString(op.GetInputDesc(2).GetShape().GetDims()), "scalar"));
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(3), 0, max_total_size, op.GetName().c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
        GetShapeErrMsg(3, DebugString(op.GetInputDesc(3).GetShape().GetDims()), "scalar"));
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(4), 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
        GetShapeErrMsg(4, DebugString(op.GetInputDesc(4).GetShape().GetDims()), "scalar"));
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(5), 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
        GetShapeErrMsg(5, DebugString(op.GetInputDesc(5).GetShape().GetDims()), "scalar"));
    return GRAPH_FAILED;
  }

  int64_t unused = 0;
  int64_t dim1 = boxes.GetDim(0);
  int64_t dim2 = scores.GetDim(0);
  if (Merge(dim1, dim2, unused) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
        ConcatString("call Merge function failed to merge 0th dim of input[boxes]"
        " and input[scores], ", dim1, " and ", dim2));
    return GRAPH_FAILED;
  }
  int64_t dim3 = boxes.GetDim(1);
  int64_t dim4 = scores.GetDim(1);
  if (Merge(dim3, dim4, unused) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
        ConcatString("call Merge function failed to merge 1th dim of input[boxes]"
        " and input[scores], ", dim3, " and ", dim4));
    return GRAPH_FAILED;
  }

  if (boxes.GetDim(3) != 4) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        ConcatString("invalid 3th dim value[", boxes.GetDim(3), "], it should be 4"));
    return GRAPH_FAILED;
  }

  Shape boxes_shape = op.GetInputDesc(0).GetShape();
  Shape scores_shape = op.GetInputDesc(1).GetShape();
  if (ValueKnown(boxes_shape, 2) && ValueKnown(scores_shape, 2)) {
    if (boxes_shape.GetDim(2) != 1 && boxes_shape.GetDim(2) != scores_shape.GetDim(2)) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
          ConcatString("2th dim of input[boxes] and input[scores] are not equal, ",
              boxes_shape.GetDim(2), " and ", scores_shape.GetDim(2)));
      return GRAPH_FAILED;
    }
  }

  Tensor maxTotalSizeTensor;
  if (op.GetInputConstData("max_total_size", maxTotalSizeTensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        std::string("get const data from input[max_total_size] failed"));
    return GRAPH_FAILED;
  }
  int64_t maxTotalSize;
  if (MakeDimForScalarInput(maxTotalSizeTensor, maxTotalSize, op.GetName().c_str()) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
        ConcatString("call MakeDimForScalarInput failed to get value from input[max_total_size] tensor"));
    return GRAPH_FAILED;
  }
  if (maxTotalSize <= 0) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        ConcatString("invalid value[", maxTotalSize, "] of input[max_total_size], should be > 0"));
    return GRAPH_FAILED;
  }

  Tensor maxOutputSizePerClassTensor;
  if (op.GetInputConstData("max_output_size_per_class", maxOutputSizePerClassTensor) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
        std::string("get const data from input[max_output_size_per_class] failed"));
    return GRAPH_FAILED;
  }
  int64_t maxOutputSizePerClass;
  if (MakeDimForScalarInput(maxOutputSizePerClassTensor, maxOutputSizePerClass, op.GetName().c_str()) !=
      GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(op.GetName(),
        ConcatString("call MakeDimForScalarInput failed to get value from input[max_output_size_per_class] tensor"));
    return GRAPH_FAILED;
  }

  int64_t output_size;
  bool pad_per_class;
  if (op.GetAttr("pad_per_class", pad_per_class) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName().c_str(),
        std::string("get attr[pad_per_class] failed"));
    return GRAPH_FAILED;
  }
  if (!pad_per_class) {
    output_size = maxTotalSize;
  } else {
    if (maxOutputSizePerClass <= 0) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName(),
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
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName().c_str(),
        std::string("update output[nmsed_boxes] desc failed"));
    return GRAPH_FAILED;
  }
  TensorDesc desc2 = op.GetOutputDesc("nmsed_scores");
  desc2.SetShape(shape2);
  desc2.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("nmsed_scores", desc2) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName().c_str(),
        std::string("update output[nmsed_scores] desc failed"));
    return GRAPH_FAILED;
  }
  TensorDesc desc3 = op.GetOutputDesc("nmsed_classes");
  desc3.SetShape(shape3);
  desc3.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("nmsed_classes", desc3) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName().c_str(),
        std::string("update output[nmsed_classes] desc failed"));
    return GRAPH_FAILED;
  }
  TensorDesc desc4 = op.GetOutputDesc("valid_detections");
  desc4.SetShape(shape4);
  desc4.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("valid_detections", desc4) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(op.GetName().c_str(),
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
  CHECK(output_size.size() == 1,  OP_LOGE(op.GetName().c_str(), "invalid output size in attr."), return GRAPH_FAILED);
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

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SpatialTransformerD, SpatialTransformerDInferShape);

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
      OP_LOGD(op.GetName().c_str(), "const data int32 fusion pass ====== %d",
              (float_t)(*(const_data_ptr + i)));
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "not support this type");
    return false;
  }
  return true;
}

static bool CalculateSizeOut(const Operator& op,
                             const std::vector<int64_t>& image_shape,
                             std::vector<float_t>& scale_out,
                             const ge::Format& input_format,
                             std::vector<int64_t>& size_out) {
  int64_t size_out_h;
  int64_t size_out_w;
  if (scale_out.size() == DIM_SIZE4) {
    if (input_format == FORMAT_NHWC) {
      scale_out.erase(scale_out.begin() + 3);  // 3 is index
      scale_out.erase(scale_out.begin() + 0);  // 0 is index
    } else if (input_format == FORMAT_NCHW) {
      scale_out.erase(scale_out.begin() + 1);  // 1 is index
      scale_out.erase(scale_out.begin() + 0);  // 0 is index
    } else {
      OP_LOGE(op.GetName().c_str(), "Not supported this format%d",
              input_format);
    }
  }
  if (scale_out.size() != DIM_SIZE2) {
    OP_LOGE(op.GetName().c_str(),
            "length of scale_out after erase must be equal to 2");
    return false;
  }
  if (input_format == FORMAT_NHWC && image_shape.size() > 2) {
    size_out_h = image_shape[1] * scale_out[0];  // 0 and 1 is index
    size_out_w = image_shape[2] * scale_out[1];  // 2 and 1 is index
  } else if (input_format == FORMAT_NCHW && image_shape.size() > 3) {
    size_out_h = image_shape[2] * scale_out[0];  // 0 and 2 is index
    size_out_w = image_shape[3] * scale_out[1];  // 3 and 1 is index
  } else {
    OP_LOGE(op.GetName().c_str(),
            "Not supported this format%d, output tensor will be wrong",
            input_format);
    return false;
  }
  size_out.push_back(size_out_h);
  size_out.push_back(size_out_w);
  return true;
}

static graphStatus HadleSizeOut(const Operator& op,
                                const ge::Format& input_format,
                                std::vector<int64_t>& size_out) {
  if (size_out.size() == DIM_SIZE4) {
    if (input_format == FORMAT_NHWC) {
      size_out.erase(size_out.begin() + 3);  // 3 is index
      size_out.erase(size_out.begin() + 0);  // 0 is index
    } else if (input_format == FORMAT_NCHW) {
      size_out.erase(size_out.begin() + 1);  // 1 is index
      size_out.erase(size_out.begin() + 0);  // 0 is index
    } else {
      OP_LOGE(op.GetName().c_str(), "Not supported this format%d",
              input_format);
    }
  }
  if (size_out.size() != DIM_SIZE2) {
    OP_LOGE(op.GetName().c_str(),
            "length of size_out after erase must be equal to 2");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// ---------------ResizeNearest Op start-------------------
IMPLEMT_INFERFUNC(Resize, ResizeNearestInferShape) {
  vector<int64_t> images_shape = op.GetInputDesc("x").GetShape().GetDims();
  DataType input_dtype = op.GetInputDesc("x").GetDataType();
  Format input_format = op.GetInputDesc("x").GetFormat();
  int64_t inputs_size = op.GetInputsSize();
  DataType inputs_dtype_scales = op.GetInputDesc("scales").GetDataType();
  TensorDesc td = op.GetOutputDesc("y");
  Tensor scales_tensor;
  Tensor sizes_tensor;
  vector<int64_t> size_out;
  vector<float_t> scale_out;

  if (op.GetInputConstData("scales", scales_tensor) != GRAPH_SUCCESS) {
    OP_LOGW(op.GetName().c_str(), "Get constValue failed of [scales]");
  }

  if (inputs_size == 4) {  // 4 is number of inputs
    if (op.GetInputConstData("sizes", sizes_tensor) != GRAPH_SUCCESS) {
      OP_LOGW(op.GetName().c_str(), "Get constValue failed of [sizes]");
    }
    DataType input_dtype_sizes = op.GetInputDesc("sizes").GetDataType();
    GetConstValue(op, sizes_tensor, input_dtype_sizes, size_out);
  }
  if (size_out.size() == 0) {
    GetConstValueFloat(op, scales_tensor, inputs_dtype_scales, scale_out);
    if (!CalculateSizeOut(op, images_shape, scale_out, input_format,
                          size_out)) {
      OP_LOGE(op.GetName().c_str(), "calculate size out failed.");
      return GRAPH_FAILED;
    }
  }
  HadleSizeOut(op, input_format, size_out);

  vector<int64_t> y_shape;
  if (input_format == FORMAT_NHWC && images_shape.size() > 3) {
    y_shape.push_back(images_shape[0]);  // 0 is index
    y_shape.push_back(size_out[0]);      // 0 is index
    y_shape.push_back(size_out[1]);      // 1 is index
    y_shape.push_back(images_shape[3]);  // 3 is index
  } else if (input_format == FORMAT_NCHW && images_shape.size() > 1) {
    y_shape.push_back(images_shape[0]);  // 0 is index
    y_shape.push_back(images_shape[1]);  // 1 is index
    y_shape.push_back(size_out[0]);      // 0 is index
    y_shape.push_back(size_out[1]);      // 1 is index
  } else {
    OP_LOGE(op.GetName().c_str(), "Not supported this format%d", input_format);
  }
  td.SetShape(ge::Shape(y_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
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
    OP_LOGE(op.GetName().c_str(), "Fail to get input_image range");
    return GRAPH_FAILED;
  }
  y_desc.SetShapeRange(image_range);

  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Fail to update output y_desc");
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
    OP_LOGE(op.GetName().c_str(), "Input image should be NHWC or NCHW format, actual is [%s]",
            TypeUtils::FormatToSerialString(image_format).c_str());
    return GRAPH_FAILED;
  }

  if (image_shape.size() != 4 || flow_shape.size() != 4) {
    OP_LOGE(op.GetName().c_str(),
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
    OP_LOGE(op.GetName().c_str(),
            "Input flow channel should be 2, actual is %d", flow_shape[3]);
    return GRAPH_FAILED;
  }

  if (flow_shape[0] != image_shape[0] || flow_shape[pos_h] != image_shape[pos_h] ||
      flow_shape[pos_w] != image_shape[pos_w]) {
    OP_LOGE(op.GetName().c_str(),
            "Input flow batch, height and width should be same as image, actually flow:[%d, %d, %d], image:[%d, %d, %d]",
            flow_shape[0], flow_shape[pos_h], flow_shape[pos_w],
            image_shape[0], image_shape[pos_h], image_shape[pos_w]);
    return GRAPH_FAILED;
  }

  if (image_shape[pos_h] < 2 || image_shape[pos_w] < 2) {
    OP_LOGE(op.GetName().c_str(),
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
    OP_LOGE(op.GetName().c_str(), "Fail to get input_image or input_flow range");
    return GRAPH_FAILED;
  }
  grad_image_desc.SetShapeRange(image_range);
  grad_flow_desc.SetShapeRange(flow_range);

  if (op.UpdateOutputDesc("grad_image", grad_image_desc) != GRAPH_SUCCESS ||
      op.UpdateOutputDesc("grad_flow", grad_flow_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Fail to update output desc.");
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
    OP_LOGE(op.GetName().c_str(), "Grad format should be same as image format, actually grad: [%s], image: [%s]",
            TypeUtils::FormatToSerialString(grad_format).c_str(), TypeUtils::FormatToSerialString(image_format).c_str());
    return GRAPH_FAILED;
  }

  if (grad_shape.size() != 4 || image_shape.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "Grad shape and image shape should both be 4d, acutally grad: [%zu], image: [%zu]",
            grad_shape.size(), image_shape.size());
    return GRAPH_FAILED;
  }

  if (grad_shape != image_shape) {
    OP_LOGE(op.GetName().c_str(),
            "The shape of grad and image should be the same, acutally grad:[%d, %d, %d, %d], image[%d, %d, %d, %d]",
            grad_shape[0], grad_shape[1], grad_shape[2], grad_shape[3],
            image_shape[0], image_shape[1], image_shape[2], image_shape[3]);
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
        OP_LOGW(op.GetName().c_str(), "Expected dim of x and grid should be 4. x dim is %d. grid dim is %d.",
                x_shape.size(), grid_shape.size());
        return GRAPH_FAILED;
    }

    x_shape[2] = grid_shape[1];
    x_shape[3] = grid_shape[2];
    TensorDesc output_desc = op.GetOutputDesc("y");
    output_desc.SetShape(ge::Shape(x_shape));
    output_desc.SetDataType(x_dtype);
    output_desc.SetFormat(x_format);
    (void)op.UpdateOutputDesc("y", output_desc);
    return GRAPH_SUCCESS;
}
INFER_FUNC_REG(GridSampler2D, GridSampler2DInferShape);
// ----------------GridSampler2D END---------------------

// ---------------GridUnnormal Op start-------------------
IMPLEMT_INFERFUNC(GridUnnormal, GridUnnormalInferShape) {
    vector<int64_t> grid_shape = op.GetInputDesc("grid").GetShape().GetDims();
    vector<int64_t> x_shape = op.GetInputDesc("assist").GetShape().GetDims();
    DataType grid_dtype = op.GetInputDesc("grid").GetDataType();
    Format grid_format = op.GetInputDesc("grid").GetFormat();

    if (x_shape.size() != 4 || grid_shape.size() != 4) {
        OP_LOGW(op.GetName().c_str(), "Expected dim of assist and grid should be 4. assist dim is %d. grid dim is %d.",
                x_shape.size(), grid_shape.size());
        return GRAPH_FAILED;
    }

    if (grid_shape[3] != 2) {
        OP_LOGW(op.GetName().c_str(), "Expected last dim of grid should be 2. last dim of grid is %d.", grid_shape[3]);
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
        OP_LOGW(op.GetName().c_str(), "Expected dim of x and position should be 4. x dim is %d. position dim is %d.",
                x_shape.size(), pos_shape.size());
        return GRAPH_FAILED;
    }

    vector<int64_t> output_shape = x_shape;
    output_shape[2] = pos_shape[1];
    output_shape[3] = pos_shape[2];
    TensorDesc output_desc = op.GetOutputDesc("y");
    output_desc.SetShape(ge::Shape(output_shape));
    output_desc.SetDataType(x_dtype);
    output_desc.SetFormat(x_format);
    (void)op.UpdateOutputDesc("y", output_desc);
    return GRAPH_SUCCESS;
}
INFER_FUNC_REG(ImageUnfold, ImageUnfoldInferShape);
// ----------------ImageUnfold END---------------------

// ---------------IMGWarpOffsets Op start-------------------
IMPLEMT_INFERFUNC(IMGWarpOffsets, IMGWarpOffsetsInferShape) {
  std::string op_name = op.GetName();
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
    OP_LOGE(op.GetName().c_str(), "Expected dim of x should be 5. x dim is %d.", x_shape.size());
    return GRAPH_FAILED;
  }

  if (grid_shape.size() != 5) {
    OP_LOGE(op.GetName().c_str(), "Expected dim of grid should be 5. grid dim is %d.", grid_shape.size());
    return GRAPH_FAILED;
  }

  if (grid_shape[4] != 3) {
    OP_LOGE(op.GetName().c_str(), "Expected dim of last axis of grid should be 3. real value is %d.", grid_shape[4]);
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

// ---------------GridSampler3DGrid Op start-------------------
IMPLEMT_INFERFUNC(GridSampler3DGrad, GridSampler3DGradInferShape) {
  vector<int64_t> grad_shape = op.GetInputDescByName("grad").GetShape().GetDims();  // NCDHW
  TensorDesc x_desc = op.GetInputDescByName("x");
  TensorDesc grid_desc = op.GetInputDescByName("grid");
  vector<int64_t> grid_shape = grid_desc.GetShape().GetDims();
  vector<int64_t> x_shape = x_desc.GetShape().GetDims();

  if (x_shape.size() != 5) {
    OP_LOGE(op.GetName().c_str(), "Expected dim of x should be 5. real value is %d.", x_shape.size());
    return GRAPH_FAILED;
  }

  if (grid_shape.size() != 5) {
    OP_LOGE(op.GetName().c_str(), "Expected dim of grid should be 5. real value is %d.", grid_shape.size());
    return GRAPH_FAILED;
  }

  if (grad_shape.size() != 5) {
    OP_LOGE(op.GetName().c_str(), "Expected dim of grad should be 5. real value is %d.", grad_shape.size());
    return GRAPH_FAILED;
  }

  (void)op.UpdateOutputDesc("dx", x_desc);
  (void)op.UpdateOutputDesc("dgrid", grid_desc);
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(GridSampler3DGrad, GridSampler3DGradInferShape);
// ----------------GridSampler3DGrid END---------------------

}  // namespace ge
