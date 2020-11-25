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

namespace ge {
IMPLEMT_INFERFUNC(AdjustHue, AdjustHueInfer) {
  auto tensor = op.get_input_desc_images();
  Shape out;
  if (WithRankAtLeast(tensor, 3, out, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input images at least 3-D.");
    return GRAPH_FAILED;
  }

  DataType type = op.GetInputDesc("images").GetDataType();

  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(Shape(out));
  y_desc.SetDataType(type);
  op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AdjustHue, AdjustHueInfer);

IMPLEMT_INFERFUNC(AdjustSaturation, AdjustSaturationInfer) {
  auto tensor = op.get_input_desc_images();
  Shape out;
  if (WithRankAtLeast(tensor, 3, out, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input images at least 3-D.");
    return GRAPH_FAILED;
  }

  DataType type = op.GetInputDesc("images").GetDataType();

  TensorDesc y_desc = op.GetOutputDesc("y");
  y_desc.SetShape(Shape(out));
  y_desc.SetDataType(type);
  op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(AdjustSaturation, AdjustSaturationInfer);

IMPLEMT_INFERFUNC(AdjustContrast, AdjustContrastInfer) {
  Shape shape;
  if (WithRank(op.GetInputDesc(1), 0, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input contrast_factor rank must be 0");
    return GRAPH_PARAM_INVALID;
  }
  if (WithRankAtLeast(op.GetInputDesc(0), 3, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input images at least 3-D");
    return GRAPH_PARAM_INVALID;
  }

  TensorDesc desc = op.GetOutputDesc("y");
  desc.SetShape(Shape(shape));
  auto data_type = op.GetInputDesc("images").GetDataType();
  desc.SetDataType(data_type);
  return op.UpdateOutputDesc("y", desc);
}

INFER_FUNC_REG(AdjustContrast, AdjustContrastInfer);

IMPLEMT_INFERFUNC(CropAndResize, CropAndResizeInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  // unknown shape support
  op_desc->SetOpInferDepends({"crop_size"});

  auto x_desc = op_desc->MutableInputDesc(0);
  GeShape x_shape;
  if (WithRank(x_desc, 4, x_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input x must be 4-D, real rank is %lld", x_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  auto boxes_desc = op_desc->MutableInputDesc(1);
  GeShape boxes_shape;
  if (WithRank(boxes_desc, 2, boxes_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input boxes must be 2-D, real rank is %lld", boxes_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  auto box_index_desc = op_desc->MutableInputDesc(2);
  GeShape box_index_shape;
  if (WithRank(box_index_desc, 1, box_index_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input box_index must be 1-D, real rank is %lld",
            box_index_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  auto crop_size_desc = op_desc->MutableInputDesc(3);
  GeShape crop_size_shape;
  if (WithRank(crop_size_desc, 1, crop_size_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input crop_size must be 1-D, real rank is %lld",
            crop_size_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  auto x_dims = x_shape.GetDims();
  auto boxes_dims = boxes_shape.GetDims();
  auto box_index_dims = box_index_shape.GetDims();
  auto crop_size_dims = crop_size_shape.GetDims();

  if (boxes_dims[0] != UNKNOWN_DIM &&
      box_index_dims[0] != UNKNOWN_DIM &&
      boxes_dims[0] != box_index_dims[0]) {
    OP_LOGE(op.GetName().c_str(),
            "the 0th dimension of boxes and box_index must be equal"
            "the real dims are: boxes[0]=%lld, box_index[0]=%lld",
            boxes_dims[0], box_index_dims[0]);
    return GRAPH_FAILED;
  }

  if (crop_size_dims[0] != 2 && crop_size_dims[0] != UNKNOWN_DIM) {
    OP_LOGE(op.GetName().c_str(),
            "crop_size must be a 1-D tensor containing 2 elements, real dim is %lld",
            crop_size_dims[0]);
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
  if (input_format == FORMAT_NHWC) {
    y_dims.push_back(boxes_dims[0]);
    y_dims.push_back(crop_height);
    y_dims.push_back(crop_width);
    y_dims.push_back(x_dims[3]);
  } else if (input_format == FORMAT_NCHW) {
    y_dims.push_back(boxes_dims[0]);
    y_dims.push_back(x_dims[1]);
    y_dims.push_back(crop_height);
    y_dims.push_back(crop_width);
  } else {
    OP_LOGE(op.GetName().c_str(), "Not supported this format");
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
    OP_LOGE(op.GetName().c_str(), "input grads must be 4-D");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 4, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input iamges must be 4-D");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(2), 2, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input boxes must be 2-D");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(3), 1, shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input box_index must be 1-D");
    return GRAPH_FAILED;
  }

  auto grads_shape = op.GetInputDesc(0).GetShape().GetDims();
  auto boxes_shape = op.GetInputDesc(2).GetShape().GetDims();
  auto box_index_shape = op.GetInputDesc(3).GetShape().GetDims();

  if (grads_shape[0] != boxes_shape[0] && boxes_shape[0] != box_index_shape[0]) {
    OP_LOGE(op.GetName().c_str(), "the 0th dimension of boxes, grads and box_index must be equal");
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
  if (WithRank(grads_desc, 4, grads_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input grads must be 4-D, real rank is %lld", grads_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  auto boxes_desc = op_desc->MutableInputDesc(1);
  GeShape boxes_shape;
  if (WithRank(boxes_desc, 2, boxes_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input boxes must be 2-D, real rank is %lld", boxes_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  auto box_index_desc = op_desc->MutableInputDesc(2);
  GeShape box_index_shape;
  if (WithRank(box_index_desc, 1, box_index_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input box_index must be 1-D, real rank is %lld",
            box_index_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  auto image_size_desc = op_desc->MutableInputDesc(3);
  GeShape image_size_shape;
  if (WithRank(image_size_desc, 1, image_size_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input image_size must be 1-D, real rank is %lld",
            image_size_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  auto grads_dims = grads_shape.GetDims();
  auto boxes_dims = boxes_shape.GetDims();
  auto box_index_dims = box_index_shape.GetDims();

  if (!DimsAllEqualOrUnknown({grads_dims[0], boxes_dims[0], box_index_dims[0]})) {
    OP_LOGE(op.GetName().c_str(),
            "the 0th dimension of grads, boxes and box_index must be equal"
            "real dims are: grads[0]=%lld, boxes[0]=%lld, box_index[0]=%lld",
            grads_dims[0], boxes_dims[0], box_index_dims[0]);
    return GRAPH_FAILED;
  }

  auto image_size_dims = image_size_shape.GetDims();
  if (image_size_dims[0] != 4 && image_size_dims[0] != UNKNOWN_DIM) {
    OP_LOGE(op.GetName().c_str(), "image_size must be a 1-D tensor with 4 elements, real dim size is %lld",
            image_size_dims[0]);
    return GRAPH_FAILED;
  }

  DataType type;
  if (op.GetAttr("T", type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Op get attr T failed");
    return GRAPH_FAILED;
  }

  int64_t batch = UNKNOWN_DIM;
  int64_t image_height = UNKNOWN_DIM;
  int64_t image_width = UNKNOWN_DIM;
  int64_t depth = UNKNOWN_DIM;
  Tensor image_size_tensor;
  if (op.GetInputConstData("image_size", image_size_tensor) == GRAPH_SUCCESS) {
    const int32_t* size_data = reinterpret_cast<const int32_t*>(image_size_tensor.GetData());
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
    OP_LOGE(op.GetName().c_str(), "Not supported this format");
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
  if (input_format == FORMAT_NHWC) {
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(org_images_shape[1]);
    y_shape.push_back(org_images_shape[2]);
    y_shape.push_back(grads_shape[3]);
  } else if (input_format == FORMAT_NCHW) {
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(grads_shape[1]);
    y_shape.push_back(org_images_shape[2]);
    y_shape.push_back(org_images_shape[3]);
  } else {
    OP_LOGE(op.GetName().c_str(), "Not supported this format %d", input_format);
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
  if (op.GetInputDesc("grads").GetShape().GetShapeSize() == UNKNOWN_DIM || 
      op.GetInputDesc("size").GetShape().GetShapeSize() == UNKNOWN_DIM) {
    y_desc->SetShape(GeShape({UNKNOWN_DIM}));
    y_desc->SetDataType(grads_desc->GetDataType());
    return GRAPH_SUCCESS;
  }
  // unknown shape support
  std::vector<std::string> input_infer_depends = {"size"};
  op_desc->SetOpInferDepends(input_infer_depends);

  GeShape grads_shape;
  if (WithRank(grads_desc, 4, grads_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input grads must be 4-D, real rank is %lld", grads_desc->GetShape().GetDimNum());
    return GRAPH_PARAM_INVALID;
  }

  GeShape size_shape;
  if (WithRank(size_desc, 1, size_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input size must be 1-D, real rank is %lld", size_desc->GetShape().GetDimNum());
    return GRAPH_PARAM_INVALID;
  }

  auto size_dims = size_shape.GetDims();
  if (size_dims[0] != 2 && size_dims[0] != UNKNOWN_DIM) {
    OP_LOGE(op.GetName().c_str(), "input size must be 1-D of 2 elements, real dim size is %lld", size_dims[0]);
    return GRAPH_PARAM_INVALID;
  }

  auto size_height = UNKNOWN_DIM;
  auto size_width = UNKNOWN_DIM;
  Tensor size_tensor;
  if (op.GetInputConstData("size", size_tensor) == GRAPH_SUCCESS) {
    auto size_data = reinterpret_cast<const int32_t*>(size_tensor.GetData());
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
    OP_LOGE(op.GetName().c_str(), "Not supported this format: %d", input_format);
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
    OpsGetAttrErrReport(op.GetName(), "size");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue size failed!");
    return GRAPH_FAILED;
  }

  if (size_out.size() != DIM_SIZE2) {
    OpsAttrValueErrReport(op.GetName(), "size", ConcatString(DIM_SIZE2), ConcatString(size_out.size()));
    OP_LOGE(op.GetName().c_str(), "length of size_out must be equal to 2");
    return GRAPH_FAILED;
  }
  Format input_format = op.GetInputDesc("grads").GetFormat();
  TensorDesc td = op.GetOutputDesc("y");
  vector<int64_t> y_shape;
  if (input_format == FORMAT_NHWC) {
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(size_out[0]);
    y_shape.push_back(size_out[1]);
    y_shape.push_back(grads_shape[3]);
  } else if (input_format == FORMAT_NCHW) {
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(grads_shape[1]);
    y_shape.push_back(size_out[0]);
    y_shape.push_back(size_out[1]);
  } else {
    string expected_format_list = ConcatString("FORMAT_NHWC, FORMAT_NHWC");
    OpsInputFormatErrReport(op.GetName(), "grads", expected_format_list, ConcatString(input_format));
    OP_LOGE(op.GetName().c_str(), "Not supported this format%d", input_format);
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
    return GRAPH_FAILED;
  }
  return ColorspaceShapeFn(op, "y");
}

INFER_FUNC_REG(RGBToHSV, RGBToHSVInfer);

IMPLEMT_INFERFUNC(SampleDistortedBoundingBoxExt2, SampleDistortedBoundingBoxExt2Infer) {
  bool judge = false;

  Shape image_size;
  judge = (WithRank(op.get_input_desc_image_size(), 1, image_size, op.GetName().c_str()) != GRAPH_SUCCESS);
  if (judge) {
    OP_LOGE(op.GetName().c_str(), "input image_size must be 1-D");
    return GRAPH_FAILED;
  }

  Shape bounding_boxes;
  judge = (WithRank(op.get_input_desc_bounding_boxes(), 3, bounding_boxes, op.GetName().c_str()) != GRAPH_SUCCESS);
  if (judge) {
    OP_LOGE(op.GetName().c_str(), "input bounding_boxes must be 3-D");
    return GRAPH_FAILED;
  }

  Shape min_object_covered;
  judge =
      (WithRank(op.get_input_desc_min_object_covered(), 0, min_object_covered, op.GetName().c_str()) != GRAPH_SUCCESS);
  if (judge) {
    OP_LOGE(op.GetName().c_str(), "input min_object_covered must be a scalar");
    return GRAPH_FAILED;
  }

  const int64_t image_size_dim_value = op.get_input_desc_image_size().GetShape().GetDim(0);
  const int64_t bounding_boxes_dim2_value = op.get_input_desc_bounding_boxes().GetShape().GetDim(2);
  if ((image_size_dim_value != 3) || (bounding_boxes_dim2_value != 4)) {
    OP_LOGE(op.GetName().c_str(),
            "DimValue0 of input image_size must be 3 and DimValue2 of "
            "bounding_boxes must be 4");
    return GRAPH_FAILED;
  }

  TensorDesc begin_desc = op.GetOutputDesc("begin");
  begin_desc.SetShape(Shape({3}));
  begin_desc.SetDataType(op.GetInputDesc("image_size").GetDataType());
  if (op.UpdateOutputDesc("begin", begin_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "fail to update output begin.");
    return GRAPH_FAILED;
  }

  TensorDesc size_desc = op.GetOutputDesc("size");
  size_desc.SetShape(Shape({3}));
  size_desc.SetDataType(op.GetInputDesc("image_size").GetDataType());
  if (op.UpdateOutputDesc("size", size_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "fail to update output size.");
    return GRAPH_FAILED;
  }

  TensorDesc bboxes_desc = op.GetOutputDesc("bboxes");
  bboxes_desc.SetShape(Shape({1, 1, 4}));
  bboxes_desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("bboxes", bboxes_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "fail to update output bboxes.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(SampleDistortedBoundingBoxExt2, SampleDistortedBoundingBoxExt2Infer);

IMPLEMT_INFERFUNC(DrawBoundingBoxes, DrawBoundingBoxesInfer) {
  Shape images;

  if (WithRank(op.GetInputDesc(0), 4, images, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of images must be 4");
    return GRAPH_FAILED;
  }

  Format input_format = op.GetInputDesc(0).GetFormat();
  int64_t depth = images.GetDim(3);
  if (input_format == FORMAT_NCHW) {
    depth = images.GetDim(1);
  }
  if (depth != ge::UNKNOWN_DIM) {
    if (!(depth == 1 || depth == 3 || depth == 4)) {
      OP_LOGE(op.GetName().c_str(), "The value of dim-3 must be 1, 3 or 4");
      return GRAPH_FAILED;
    }
  }

  Shape boxes;
  if (WithRank(op.GetInputDesc(1), 3, boxes, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of boxes must be 3");
    return GRAPH_FAILED;
  }
  if (boxes.GetDim(2) != 4) {
    OP_LOGE(op.GetName().c_str(), "The value of dim-2 for boxes must be  4");
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
  auto boxes_desc = op_desc->MutableInputDesc(0);
  if (WithRank(boxes_desc, 2, boxes_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of boxes must be 2, real rank is %lld",
            boxes_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  GeShape scores_shape;
  auto scores_desc = op_desc->MutableInputDesc(1);
  if (WithRank(scores_desc, 1, scores_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of scores must be 1, real rank is %lld",
            scores_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  GeShape max_output_size_shape;
  auto max_output_size_desc = op_desc->MutableInputDesc(2);
  if (WithRank(max_output_size_desc, 0, max_output_size_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of max_output_size must be 0, real rank is %lld",
            max_output_size_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  int64_t unused_dim;
  if (Merge(boxes_shape.GetDim(0), scores_shape.GetDim(0), unused_dim) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to merge dim of boxes[0] and scores[0].");
    return GRAPH_FAILED;
  }

  if (boxes_shape.GetDim(1) != 4 && boxes_shape.GetDim(1) != UNKNOWN_DIM) {
    OP_LOGE(op.GetName().c_str(), "The dim of boxes[1] is not 4 but %lld", boxes_shape.GetDim(1));
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
  if (WithRank(boxes_desc, 2, boxes_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of boxes must be 2, real rank is %lld",
            boxes_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  GeShape scores_shape;
  auto scores_desc = op_desc->MutableInputDesc(1);
  if (WithRank(scores_desc, 1, scores_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of scores must be 1, real rank is %lld",
            scores_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  GeShape max_output_size_shape;
  auto max_output_size_desc = op_desc->MutableInputDesc(2);
  if (WithRank(max_output_size_desc, 0, max_output_size_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of max_output_size must be 0, real rank is %lld",
            max_output_size_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  GeShape iou_threshold_shape;
  auto iou_threshold_desc = op_desc->MutableInputDesc(3);
  if (WithRank(iou_threshold_desc, 0, iou_threshold_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of iou_threshold must be 0, real rank is %lld",
            iou_threshold_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  int64_t unused_dim;
  if (Merge(boxes_shape.GetDim(0), scores_shape.GetDim(0), unused_dim) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to merge dim of boxes[0] and scores[0].");
    return GRAPH_FAILED;
  }

  if (boxes_shape.GetDim(1) != 4 && boxes_shape.GetDim(1) != UNKNOWN_DIM) {
    OP_LOGE(op.GetName().c_str(), "The dim of boxes[1] is not 4 but %lld", boxes_shape.GetDim(1));
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
  if (WithRank(boxes_desc, 2, boxes_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of boxes must be 2, real rank is %lld",
            boxes_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  GeShape scores_shape;
  auto scores_desc = op_desc->MutableInputDesc(1);
  if (WithRank(scores_desc, 1, scores_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of scores must be 1, real rank is %lld",
            scores_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  GeShape max_output_size_shape;
  auto max_output_size_desc = op_desc->MutableInputDesc(2);
  if (WithRank(max_output_size_desc, 0, max_output_size_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of max_output_size must be 0, real rank is %lld",
            max_output_size_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  GeShape iou_threshold_shape;
  auto iou_threshold_desc = op_desc->MutableInputDesc(3);
  if (WithRank(iou_threshold_desc, 0, iou_threshold_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of iou_threshold must be 0, real rank is %lld",
            iou_threshold_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  GeShape score_threshold_shape;
  auto score_threshold_desc = op_desc->MutableInputDesc(4);
  if (WithRank(score_threshold_desc, 0, score_threshold_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of score_threshold must be 0, real rank is %lld",
            score_threshold_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  int64_t unused_dim;
  if (Merge(boxes_shape.GetDim(0), scores_shape.GetDim(0), unused_dim) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to merge boxes[0]=%lld and scores[0]=%lld",
            boxes_shape.GetDim(0), scores_shape.GetDim(0));
    return GRAPH_FAILED;
  }

  if (boxes_shape.GetDim(1) != 4 && boxes_shape.GetDim(1) != UNKNOWN_DIM) {
    OP_LOGE(op.GetName().c_str(), "The dim of boxes[1] is not 4 but %lld", boxes_shape.GetDim(1));
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

  // unknwon shape support
  std::vector<std::string> input_infer_depends = {"max_output_size"};
  op_desc->SetOpInferDepends(input_infer_depends);

  GeShape boxes_shape;
  auto boxes_desc = op_desc->MutableInputDesc(0);
  if (WithRank(boxes_desc, 2, boxes_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of boxes must be 2, real rank is %lld",
            boxes_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  GeShape scores_shape;
  auto scores_desc = op_desc->MutableInputDesc(1);
  if (WithRank(scores_desc, 1, scores_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of scores must be 1, real rank is %lld",
            scores_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  GeShape max_output_size_shape;
  auto max_output_size_desc = op_desc->MutableInputDesc(2);
  if (WithRank(max_output_size_desc, 0, max_output_size_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of max_output_size must be 0, real rank is %lld",
            max_output_size_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  GeShape iou_threshold_shape;
  auto iou_threshold_desc = op_desc->MutableInputDesc(3);
  if (WithRank(iou_threshold_desc, 0, iou_threshold_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of iou_threshold must be 0, real rank is %lld",
            iou_threshold_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  GeShape score_threshold_shape;
  auto score_threshold_desc = op_desc->MutableInputDesc(4);
  if (WithRank(score_threshold_desc, 0, score_threshold_shape) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of score_threshold must be 0, real rank is %lld",
            score_threshold_desc->GetShape().GetDimNum());
    return GRAPH_FAILED;
  }

  int64_t unused_dim;
  if (Merge(boxes_shape.GetDim(0), scores_shape.GetDim(0), unused_dim) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to merge dim of boxes[0]=%lld and scores[0]=%lld.",
            boxes_shape.GetDim(0), scores_shape.GetDim(0));
    return GRAPH_FAILED;
  }
  if (WithValue(boxes_shape.GetDim(1), 4, unused_dim, op.GetName().c_str()) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "The dim of boxes[1] is not 4, real dim is %lld",
              boxes_shape.GetDim(1));
      return GRAPH_FAILED;
  }

  std::vector<int64_t> selected_indices_dims{UNKNOWN_DIM};
  bool pad_to_max = false;
  if (op.GetAttr("pad_to_max_output_size", pad_to_max) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The tile op GetOpAttr ConstValue failed!");
    return GRAPH_FAILED;
  }
  if (pad_to_max) {
    Tensor selected_indices_tensor;
    if (op.GetInputConstData("max_output_size", selected_indices_tensor) == GRAPH_SUCCESS) {
      const int32_t *selected_indices_data =
          reinterpret_cast<const int32_t*>(selected_indices_tensor.GetData());
      int32_t selected_indices_data_0 = *selected_indices_data;
      if (selected_indices_data_0 < 0) {
        OP_LOGE(op.GetName().c_str(),
                "The tensor value of max_output_size must be non-negative, real value is %d",
                selected_indices_data_0);
        return GRAPH_FAILED;
      }
      selected_indices_dims[0] = selected_indices_data_0;
    }
  }
  auto selected_indices_desc = op_desc->MutableOutputDesc("selected_indices");
  (void)FillOpDesc(selected_indices_desc, GeShape(selected_indices_dims), DT_INT32);

  auto valid_outputs_desc = op_desc->MutableOutputDesc("valid_outputs");
  (void)FillOpDesc(valid_outputs_desc, GeShape(), DT_INT32);

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
    OP_LOGE(op.GetName().c_str(), "The rank of overlaps should be equal to 2.");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc("scores"), 1, scores_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of scores should be equal to 1.");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc("max_output_size"), 0, max_output_size_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of max_output_size should be equal to 0.");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc("overlap_threshold"), 0, overlap_threshold_shape, op.GetName().c_str()) !=
      GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of overlap_threshold should be equal to 0.");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc("score_threshold"), 0, score_threshold_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of score_threshold should be equal to 0.");
    return GRAPH_FAILED;
  }
  int64_t unused_dim = 0;
  if (Merge(overlaps_shape.GetDim(0), scores_shape.GetDim(0), unused_dim) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "dims[0] of overlaps should be equal to dims[0] of scores.");
    return GRAPH_FAILED;
  }
  if (Merge(overlaps_shape.GetDim(0), overlaps_shape.GetDim(1), unused_dim) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "dims[0] of overlaps should be equal to dims[1] of overlaps.");
    return GRAPH_FAILED;
  }

  TensorDesc selected_indices_desc = op.GetOutputDesc("selected_indices");
  Shape selecte_indices_shape;
  Vector(ge::UNKNOWN_DIM, selecte_indices_shape);
  selected_indices_desc.SetDataType(DT_INT32);
  selected_indices_desc.SetShape(selecte_indices_shape);
  if (op.UpdateOutputDesc("selected_indices", selected_indices_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to update selected_indices desc.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(NonMaxSuppressionWithOverlaps, NonMaxSuppressionWithOverlapsInfer);

IMPLEMT_INFERFUNC(EncodePng, EncodePngInfer) {
  return EncodeImageShapeFn(op);
}

INFER_FUNC_REG(EncodePng, EncodePngInfer);

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
  if (input_format == FORMAT_NHWC) {
    y_shape.push_back(image_shape[0]);
    y_shape.push_back(size_out[0]);
    y_shape.push_back(size_out[1]);
    y_shape.push_back(image_shape[3]);
    result_range.push_back(x_range[0]);
    result_range.push_back(output_range[0]);
    result_range.push_back(output_range[1]);
    result_range.push_back(x_range[3]);
  } else if (input_format == FORMAT_NCHW) {
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
  if (input_format == FORMAT_NHWC) {
    y_shape.push_back(images_shape[0]);
    y_shape.push_back(size_out[0]);
    y_shape.push_back(size_out[1]);
    y_shape.push_back(images_shape[3]);
  } else if (input_format == FORMAT_NCHW) {
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
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue size failed!");
    return GRAPH_FAILED;
  }

  if (size_out.size() != DIM_SIZE2) {
    OpsAttrValueErrReport(op.GetName(), "size's length", "2", ConcatString(size_out.size()));
    OP_LOGE(op.GetName().c_str(), "length of size_out must be equal to 2");
    return GRAPH_FAILED;
  }
  Format input_format = op.GetInputDesc("x").GetFormat();
  TensorDesc td = op.GetOutputDesc("y");
  vector<int64_t> y_shape;
  if (input_format == FORMAT_NHWC) {
    y_shape.push_back(images_shape[0]);
    y_shape.push_back(size_out[0]);
    y_shape.push_back(size_out[1]);
    y_shape.push_back(images_shape[3]);
  } else if (input_format == FORMAT_NCHW) {
    y_shape.push_back(images_shape[0]);
    y_shape.push_back(images_shape[1]);
    y_shape.push_back(size_out[0]);
    y_shape.push_back(size_out[1]);
  } else {
    string expected_format_list = ConcatString("FORMAT_NHWC, FORMAT_NCHW");
    OpsInputFormatErrReport(op.GetName(), "x", expected_format_list, ConcatString(input_format));
    OP_LOGE(op.GetName().c_str(), "Not supported this format");
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
  PREPARE_DYNAMIC_SHAPE_WITH_NO_DEPENDS();
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
  if (input_format == FORMAT_NHWC) {
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(images_shape[1]);
    y_shape.push_back(images_shape[2]);
    y_shape.push_back(grads_shape[3]);
    y_range.push_back(grads_range[0]);
    y_range.push_back(image_range[1]);
    y_range.push_back(image_range[2]);
    y_range.push_back(grads_range[3]);
  } else if (input_format == FORMAT_NCHW) {
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
    OpsInputFormatErrReport(op.GetName(), "grads", expected_format_list, ConcatString(input_format));
    OP_LOGE(op.GetName().c_str(), "Not supported this format%d", input_format);
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
  if (WithRank(op.GetInputDesc(0), 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "the first input must be 0-D .");
    return GRAPH_FAILED;
  }
  DataType output_type;
  if (op.GetAttr("output_type", output_type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr output_type failed");
    return GRAPH_FAILED;
  }
  Shape output_shape;
  Vector(3, output_shape);
  TensorDesc image_shape_desc = op.GetOutputDesc("image_shape");
  image_shape_desc.SetShape(output_shape);
  image_shape_desc.SetDataType(output_type);
  image_shape_desc.SetFormat(FORMAT_NHWC);
  if (op.UpdateOutputDesc("image_shape", image_shape_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update image_shape desc failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ExtractJpegShape, ExtractJpegShapeInfer);

IMPLEMT_INFERFUNC(DrawBoundingBoxesV2, DrawBoundingBoxesV2Infer) {
  auto imagesTensor = op.get_input_desc_images();

  Shape images;
  if (WithRankAtLeast(imagesTensor, 3, images, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input images rank must be at least 3.");
    return GRAPH_FAILED;
  }

  DataType type = op.GetInputDesc("images").GetDataType();
  TensorDesc outputDesc = op.GetOutputDesc("y");
  outputDesc.SetDataType(type);
  outputDesc.SetShape(images);
  if (op.UpdateOutputDesc("y", outputDesc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "fail to update output y.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
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
    OP_LOGE(op.GetName().c_str(), "The rank of boxes must be 2");
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(1), 1, scores, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of scores must be 1");
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(2), 0, max_output_size, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of max_output_size must be 0");
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(3), 0, iouThreshold, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of iou_threshold must be 0");
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(4), 0, scoreThreshold, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of score_threshold must be 0");
    return GRAPH_FAILED;
  }

  if (WithRank(op.GetInputDesc(5), 0, softNmsSigma, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of soft_nms_sigma must be 0");
    return GRAPH_FAILED;
  }

  int64_t un_used;

  if (Merge(boxes.GetDim(0), scores.GetDim(0), un_used) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Failed to merge dim of boxes[0] and scores[0].");
    return GRAPH_FAILED;
  }

  if (boxes.GetDim(1) != 4) {
    if (boxes.GetDim(1) != UNKNOWN_DIM) {
      OP_LOGE(op.GetName().c_str(), "The dim of boxes[1] is not 4, real value is %ld.", boxes.GetDim(1));
      return GRAPH_FAILED;
    }
  }

  bool pad_to_max;
  if (ge::GRAPH_SUCCESS != op.GetAttr("pad_to_max_output_size", pad_to_max)) {
    OP_LOGE(op.GetName().c_str(), "The tile op GetOpAttr ConstValue failed!");
    return GRAPH_FAILED;
  }

  TensorDesc out_desc = op.GetOutputDesc("selected_indices");
  TensorDesc out_desc_scores = op.GetOutputDesc("selected_scores");
  out_desc.SetDataType(DT_INT32);
  DataType type;
  if (op.GetAttr("T", type) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr T error.");
    return GRAPH_FAILED;
  }
  out_desc_scores.SetDataType(type);

  if (!pad_to_max) {
    out_desc.SetShape(Shape({UNKNOWN_DIM}));
    out_desc_scores.SetShape(Shape({UNKNOWN_DIM}));
    if (op.UpdateOutputDesc("selected_indices", out_desc) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "fail to update output selected_indices.");
      return GRAPH_FAILED;
    }
    if (op.UpdateOutputDesc("selected_scores", out_desc_scores) != GRAPH_SUCCESS) {
      OP_LOGE(op.GetName().c_str(), "fail to update output selected_scores.");
      return GRAPH_FAILED;
    }
  } else {
    Tensor in_tensor;
    if (op.GetInputConstData("max_output_size", in_tensor) != GRAPH_SUCCESS) {
      out_desc.SetShape(Shape({UNKNOWN_DIM}));
      out_desc_scores.SetShape(Shape({UNKNOWN_DIM}));
      if (op.UpdateOutputDesc("selected_indices", out_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "fail to update output selected_indices.");
        return GRAPH_FAILED;
      }
      if (op.UpdateOutputDesc("selected_scores", out_desc_scores) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "fail to update output selected_scores.");
        return GRAPH_FAILED;
      }
    } else {
      const int32_t* size_data = reinterpret_cast<const int32_t*>(in_tensor.GetData());
      if (*size_data < 0) {
        OP_LOGE(op.GetName().c_str(), "The dim size of max_output_size must be non-negative");
        return GRAPH_FAILED;
      }
      out_desc.SetShape(Shape({*size_data}));
      if (op.UpdateOutputDesc("selected_indices", out_desc) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "fail to update output selected_indices.");
        return GRAPH_FAILED;
      }
      out_desc_scores.SetShape(Shape({*size_data}));
      if (op.UpdateOutputDesc("selected_scores", out_desc_scores) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "fail to update output selected_scores.");
        return GRAPH_FAILED;
      }
    }
  }

  TensorDesc out_desc1 = op.GetOutputDesc("valid_outputs");
  out_desc1.SetShape(Shape());
  out_desc1.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("valid_outputs", out_desc1) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "fail to update output valid_outputs.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(NonMaxSuppressionV5, NonMaxSuppressionV5Infer);

IMPLEMT_INFERFUNC(ScaleAndTranslate, ScaleAndTranslateInfer) {
  TensorDesc desc = op.GetOutputDesc("y");
  desc.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "fail to update output y.");
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
  if (input_format == FORMAT_NHWC) {
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(org_images_shape[1]);
    y_shape.push_back(org_images_shape[2]);
    y_shape.push_back(grads_shape[3]);
  } else if (input_format == FORMAT_NCHW) {
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(grads_shape[1]);
    y_shape.push_back(org_images_shape[2]);
    y_shape.push_back(org_images_shape[3]);
  } else {
    y_shape.push_back(grads_shape[0]);
    y_shape.push_back(org_images_shape[1]);
    y_shape.push_back(org_images_shape[2]);
    y_shape.push_back(grads_shape[3]);
    OP_LOGE(op.GetName().c_str(), "Real format is %d", input_format);
  }

  desc.SetShape(ge::Shape(y_shape));
  desc.SetDataType(DT_FLOAT);
  return op.UpdateOutputDesc("y", desc);
}

INFER_FUNC_REG(ScaleAndTranslateGrad, ScaleAndTranslateGradInfer);

IMPLEMT_INFERFUNC(CombinedNonMaxSuppression, CombinedNonMaxSuppressionInfer) {
  Shape boxes;
  Shape scores;
  Shape max_output_size_per_class;
  Shape max_total_size;
  Shape unused_shape;

  if (WithRank(op.GetInputDesc(0), 4, boxes, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of boxes must be 4");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 3, scores, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of scores must be 3");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(2), 0, max_output_size_per_class, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of max_output_size_per_class must be 0");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(3), 0, max_total_size, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of max_total_size must be 0");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(4), 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of unused_shape1 must be 0");
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(5), 0, unused_shape, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "The rank of unused_shape2 must be 0");
    return GRAPH_FAILED;
  }

  int64_t unused = 0;
  int64_t dim1 = boxes.GetDim(0);
  int64_t dim2 = scores.GetDim(0);
  if (Merge(dim1, dim2, unused) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Merge boxes and scores dim 0 failed.");
    return GRAPH_FAILED;
  }
  int64_t dim3 = boxes.GetDim(1);
  int64_t dim4 = scores.GetDim(1);
  if (Merge(dim3, dim4, unused) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Merge boxes and scores dim 1 failed.");
    return GRAPH_FAILED;
  }

  if (boxes.GetDim(3) != 4) {
    OP_LOGE(op.GetName().c_str(), "The value of dim-3 for boxes must be 4, real value is %ld.", boxes.GetDim(3));
    return GRAPH_FAILED;
  }

  Shape boxes_shape = op.GetInputDesc(0).GetShape();
  Shape scores_shape = op.GetInputDesc(1).GetShape();
  if (ValueKnown(boxes_shape, 2) && ValueKnown(scores_shape, 2)) {
    if (boxes_shape.GetDim(2) != 1 && boxes_shape.GetDim(2) != scores_shape.GetDim(2)) {
      OP_LOGE(op.GetName().c_str(), "boxes_shape and scores_shape do not match.");
      return GRAPH_FAILED;
    }
  }

  Tensor maxTotalSizeTensor;
  if (op.GetInputConstData("max_total_size", maxTotalSizeTensor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get maxTotalSizeTensor error.");
    return GRAPH_FAILED;
  }
  int64_t maxTotalSize;
  if (MakeDimForScalarInput(maxTotalSizeTensor, maxTotalSize, op.GetName().c_str()) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input maxTotalSize MakeDimForScalarInput error, real value is %ld.", maxTotalSize);
    return GRAPH_FAILED;
  }
  if (maxTotalSize <= 0) {
    OP_LOGE(op.GetName().c_str(), "max_total_size should be > 0, real value is %ld.", maxTotalSize);
    return GRAPH_FAILED;
  }

  Tensor maxOutputSizePerClassTensor;
  if (op.GetInputConstData("max_output_size_per_class", maxOutputSizePerClassTensor) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get maxOutputSizePerClassTensor error.");
    return GRAPH_FAILED;
  }
  int64_t maxOutputSizePerClass;
  if (MakeDimForScalarInput(maxOutputSizePerClassTensor, maxOutputSizePerClass, op.GetName().c_str()) !=
      GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "input maxOutputSizePerClass MakeDimForScalarInput error.");
    return GRAPH_FAILED;
  }

  int64_t output_size;
  bool pad_per_class;
  if (op.GetAttr("pad_per_class", pad_per_class) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr pad_per_class failed");
    return GRAPH_FAILED;
  }
  if (!pad_per_class) {
    output_size = maxTotalSize;
  } else {
    if (maxOutputSizePerClass <= 0) {
      OP_LOGE(op.GetName().c_str(), "max_output_size_per_class should be > 0, real value is %ld.",
              maxOutputSizePerClass);
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
    OP_LOGE(op.GetName().c_str(), "fail to update output nmsed_boxes.");
    return GRAPH_FAILED;
  }
  TensorDesc desc2 = op.GetOutputDesc("nmsed_scores");
  desc2.SetShape(shape2);
  desc2.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("nmsed_scores", desc2) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "fail to update output nmsed_scores.");
    return GRAPH_FAILED;
  }
  TensorDesc desc3 = op.GetOutputDesc("nmsed_classes");
  desc3.SetShape(shape3);
  desc3.SetDataType(DT_FLOAT);
  if (op.UpdateOutputDesc("nmsed_classes", desc3) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "fail to update output nmsed_classes.");
    return GRAPH_FAILED;
  }
  TensorDesc desc4 = op.GetOutputDesc("valid_detections");
  desc4.SetShape(shape4);
  desc4.SetDataType(DT_INT32);
  if (op.UpdateOutputDesc("valid_detections", desc4) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "fail to update output valid_detections.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(CombinedNonMaxSuppression, CombinedNonMaxSuppressionInfer);

IMPLEMT_INFERFUNC(SpatialTransformerD, SpatialTransformerDInferShape) {
  auto x_shape = op.get_input_desc_x().GetShape();
  auto x_dtype = op.get_input_desc_x().GetDataType();

  std::vector<int64_t> output_size = op.get_attr_output_size();
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

}  // namespace ge
