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
 * \file nn_other_ops.cpp
 * \brief
 */
#include "inc/nn_detect_ops.h"

#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#include "util/util.h"
#include "util/error_util.h"
#include "op_log.h"

namespace ge {
// ------------------CheckValid-----------------------
IMPLEMT_COMMON_INFERFUNC(CheckValidInferShape) {
  auto tensordesc_bbox = op.GetInputDesc("bbox_tensor");
  auto shape_bbox = tensordesc_bbox.GetShape();

  std::vector<int64_t> dims_tmp;
  dims_tmp.push_back(shape_bbox.GetDim(0));

  Shape valid_shape(dims_tmp);
  TensorDesc td = op.GetOutputDesc("valid_tensor");
  td.SetShape(valid_shape);
  (void)op.UpdateOutputDesc("valid_tensor", td);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(CheckValid, CheckValidInferShape);
// -------------------CheckValid END------------------------

// ---------------ROIAlignGrad--------------------
IMPLEMT_INFERFUNC(ROIAlignGrad, ROIAlignGradInfer) {
  std::vector<int64_t> xdiff_shape;
  if (GRAPH_SUCCESS != op.GetAttr("xdiff_shape", xdiff_shape)) {
    OpsGetAttrErrReport(op.GetName(), "xdiff_shape");
    OP_LOGE(op.GetName().c_str(), "GetOpAttr ConstValue xdiff_shape failed!");
    return GRAPH_FAILED;
  }
  auto inputType = op.GetInputDesc("ydiff").GetDataType();

  Shape valid_shape(xdiff_shape);
  TensorDesc td = op.GetOutputDesc("xdiff");
  td.SetShape(ge::Shape(valid_shape));
  td.SetDataType(inputType);
  (void)op.UpdateOutputDesc("xdiff", td);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ROIAlignGrad, ROIAlignGradInfer);
// -------------ROIAlignGrad END----------------

// --------------ROIAlign------------------
IMPLEMT_INFERFUNC(ROIAlign, ROIAlignInfer) {
  auto inputDtype = op.GetInputDesc("features").GetDataType();
  auto input_shape = op.GetInputDesc("features").GetShape();
  auto input2_shape = op.GetInputDesc("rois").GetShape();
  int64_t pool_h_shape;
  int64_t pool_w_shape;
  if (op.GetAttr("pooled_height", pool_h_shape) == ge::GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "pooled_height");
    OP_LOGI(op.GetName().c_str(), "GetOpAttr ConstValue pooled_height failed. Use unknown shape.");
    pool_h_shape = UNKNOWN_DIM;
  }
  if (op.GetAttr("pooled_width", pool_w_shape) == ge::GRAPH_FAILED) {
    OpsGetAttrErrReport(op.GetName(), "pooled_width");
    OP_LOGI(op.GetName().c_str(), "GetOpAttr ConstValue pooled_width failed. Use unknown shape.");
    pool_w_shape = UNKNOWN_DIM;
  }

  std::vector<int64_t> dimsTmp;
  dimsTmp.push_back(input2_shape.GetDim(0));
  dimsTmp.push_back(input_shape.GetDim(1));
  dimsTmp.push_back(pool_h_shape);
  dimsTmp.push_back(pool_w_shape);
  Shape validShape(dimsTmp);

  auto td = op.GetOutputDesc("y");
  td.SetShape(validShape);
  td.SetDataType(inputDtype);
  (void)op.UpdateOutputDesc("y", td);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ROIAlign, ROIAlignInfer);
// ------------ROIAlign END-------------

// ----------------Iou-------------------
IMPLEMT_COMMON_INFERFUNC(IouInferShape) {
  auto shapeBbox = op.GetInputDesc("bboxes").GetShape();
  auto shapeGbox = op.GetInputDesc("gtboxes").GetShape();
  vector<int64_t> shapeOut;
  shapeOut.push_back(shapeGbox.GetDim(0));
  shapeOut.push_back(shapeBbox.GetDim(0));

  Shape outputShape(shapeOut);
  DataType inputType = op.GetInputDesc("bboxes").GetDataType();

  TensorDesc td = op.GetOutputDesc("overlap");
  td.SetShape(outputShape);
  td.SetDataType(inputType);
  (void)op.UpdateOutputDesc("overlap", td);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Iou, IouInferShape);
// ----------------Iou-------------------

// ----------------BoundingBoxDecode-------------------
IMPLEMT_COMMON_INFERFUNC(BoundingBoxDecodeInferShape) {
  if (InferShapeAndTypeTwoInOneOutBroadcast(op, "rois", "deltas", "bboxes")) {
    return GRAPH_SUCCESS;
  }
  OP_LOGE(op.GetName().c_str(), "the BoundingBoxDecode InferShape Failed.");
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(BoundingBoxDecode, BoundingBoxDecodeInferShape);
// ----------------BoundingBoxDecode END-------------------

// ----------------BoundingBoxEncode-------------------
IMPLEMT_COMMON_INFERFUNC(BoundingBoxEncodeInferShape) {
  if (InferShapeAndTypeTwoInOneOutBroadcast(op, "anchor_box", "ground_truth_box", "delats")) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(BoundingBoxEncode, BoundingBoxEncodeInferShape);
// ----------------BoundingBoxEncode END-------------------

// ----------------PriorBox Op Start-------------------
IMPLEMT_INFERFUNC(PriorBox, PriorBoxInfer) {
  auto featureDesc = op.GetInputDesc("x");

  std::vector<float> aspect_ratio;
  op.GetAttr("aspect_ratio", aspect_ratio);
  int64_t ar_size = aspect_ratio.size();

  std::vector<float> min_size;
  op.GetAttr("min_size", min_size);
  int64_t min_size_size = min_size.size();

  std::vector<float> max_size;
  op.GetAttr("max_size", max_size);
  int64_t max_size_size = max_size.size();

  bool flip;
  op.GetAttr("flip", flip);

  auto input_dType = featureDesc.GetDataType();
  auto output_dType = input_dType;

  auto xShape = featureDesc.GetShape().GetDims();
  if (xShape.size() < 4) {
    OP_LOGE(op.GetName().c_str(), "input x dim is illegal, expected: > 3, actual: %zu.", xShape.size());
    return GRAPH_FAILED;
  }
  int64_t inputH, inputW;
  inputH = xShape[2];
  inputW = xShape[3];

  vector<float> aspectratios_new;
  for (int i = 0; i < ar_size; i++) {
    float ar = aspect_ratio[i];
    bool already_exist = false;
    if (fabsf(ar - 1.0) < 1e-6) {
      already_exist = true;
    } else {
      for (uint16_t j = 0; j < aspectratios_new.size(); j++) {
        if (fabsf(ar - aspectratios_new[j]) < 1e-6) {
          already_exist = true;
          break;
        }
      }
    }
    if (!already_exist) {
      aspectratios_new.push_back(ar);
      if (flip) {
        if (ar <= 0) {
          OpsAttrValueErrReport(op.GetName(), "aspect_ratio", "greater than 0", ConcatString(ar));
          OP_LOGE(op.GetName().c_str(), "aspect_ratio need greater than 0");
          return GRAPH_FAILED;
        }
        aspectratios_new.push_back(1.0 / ar);
      }
    }
  }
  int64_t ar_new_size = aspectratios_new.size();

  int64_t priorNum;
  if (ar_size == 1 && (fabsf(aspect_ratio[0] - 1.0) < 1e-6)) {
    priorNum = min_size_size * ar_size + max_size_size;
  } else {
    priorNum = min_size_size + min_size_size * ar_new_size + max_size_size;
  }

  vector<int64_t> yShape({1, 2, inputH * inputW * priorNum * 4, 1});

  auto outdesc = op.GetOutputDesc("y");
  outdesc.SetShape(Shape(yShape));
  outdesc.SetDataType(output_dType);
  outdesc.SetFormat(featureDesc.GetFormat());
  (void)op.update_output_desc_y(outdesc);

  return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(PriorBox, PriorBoxVerify) {
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(PriorBox, PriorBoxInfer);
VERIFY_FUNC_REG(PriorBox, PriorBoxVerify);
// ----------------PriorBox Op End-------------------

// ----------------PriorBoxD Op Start-------------------
IMPLEMT_INFERFUNC(PriorBoxD, PriorBoxDInfer) {
  auto featureDesc = op.GetInputDesc("x");
  auto boxDesc = op.GetInputDesc("box_height");
  auto input_dType = featureDesc.GetDataType();
  auto output_dType = input_dType;

  auto xShape = featureDesc.GetShape().GetDims();
  if (xShape.size() < 4) {
    OP_LOGE(op.GetName().c_str(), "input x dim is illegal, expected: > 3, actual: %zu.", xShape.size());
    return GRAPH_FAILED;
  }
  int64_t inputH, inputW;
  inputH = xShape[2];
  inputW = xShape[3];
  auto boxLen = boxDesc.GetShape().GetDims();
  if (boxLen.size() == 0) {
    OP_LOGE(op.GetName().c_str(), "input box dim is illegal, expected: > 0, actual: %zu.", boxLen.size());
    return GRAPH_FAILED;
  }
  int64_t priorNum;
  priorNum = boxLen[0];
  vector<int64_t> yShape({1, 2, inputH * inputW * priorNum, 4});

  auto outdesc = op.GetOutputDesc("y");
  outdesc.SetShape(Shape(yShape));
  outdesc.SetDataType(output_dType);
  outdesc.SetFormat(featureDesc.GetFormat());
  (void)op.update_output_desc_y(outdesc);

  return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(PriorBoxD, PriorBoxDVerify) {
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(PriorBoxD, PriorBoxDInfer);
VERIFY_FUNC_REG(PriorBoxD, PriorBoxDVerify);
// ----------------PriorBoxD Op End-------------------

// ---------------PriorBoxDV2 Op Start------------------
IMPLEMT_INFERFUNC(PriorBoxDV2, PriorBoxDV2Infer) {
  auto featureDesc = op.GetInputDesc("x");
  auto boxDesc = op.GetInputDesc("boxes");
  auto input_dType = featureDesc.GetDataType();
  auto output_dType = input_dType;

  auto xShape = featureDesc.GetShape().GetDims();
  if (xShape.size() < 4) {
    OP_LOGE(op.GetName().c_str(), "input x dim is illegal, expected: > 3, actual: %zu.", xShape.size());
    return GRAPH_FAILED;
  }
  int64_t inputH, inputW;
  inputH = xShape[2];
  inputW = xShape[3];
  auto boxLen = boxDesc.GetShape().GetDims();
  if (boxLen.size() == 0) {
    OP_LOGE(op.GetName().c_str(), "input box dim is illegal, expected: > 0, actual: %zu.", boxLen.size());
    return GRAPH_FAILED;
  }
  int64_t priorNum;
  priorNum = boxLen[0];
  vector<int64_t> yShape({1, 2, inputH * inputW * priorNum, 4});

  auto outdesc = op.GetOutputDesc("y");
  outdesc.SetShape(Shape(yShape));
  outdesc.SetDataType(output_dType);
  outdesc.SetFormat(featureDesc.GetFormat());
  (void)op.update_output_desc_y(outdesc);

  return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(PriorBoxDV2, PriorBoxDV2Verify) {
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(PriorBoxDV2, PriorBoxDV2Infer);
VERIFY_FUNC_REG(PriorBoxDV2, PriorBoxDV2Verify);
// ---------------PriorBoxDV2 Op End------------------
}  // namespace ge
