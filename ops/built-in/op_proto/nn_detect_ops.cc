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
 * \file nn_detect_ops.cpp
 * \brief
 */
#include "inc/nn_detect_ops.h"
#include <cmath>
#include <string>
#include <vector>

#include "common/util/error_manager/error_manager.h"

#include "util/util.h"
#include "util/error_util.h"
#include "op_log.h"

namespace ge {
// ----------------Yolo-------------------
int64_t CeilX(int64_t size, int64_t alignSize) {
  return (size + alignSize - 1) / alignSize * alignSize;
}
IMPLEMT_COMMON_INFERFUNC(YoloInferShape) {
  // get input depth
  OP_LOGI("yolo", "infer shape begin---");
  auto inputShape = op.GetInputDesc("x").GetShape().GetDims();
  int64_t batchNum = (int64_t)inputShape[0];
  int64_t hwSize = (int64_t)(inputShape[2] * inputShape[3]);
  DataType inputType = op.GetInputDesc("x").GetDataType();
  std::int64_t boxNum = 0;
  std::int64_t coordNum = 0;
  std::int64_t classNum = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("boxes", boxNum)) {
    boxNum = 3;
  }
  if (ge::GRAPH_SUCCESS != op.GetAttr("coords", coordNum)) {
    coordNum = 4;
  }
  if (ge::GRAPH_SUCCESS != op.GetAttr("classes", classNum)) {
    classNum = 80;
  }
  std::vector<int64_t> coord_dim_vector;
  coord_dim_vector.push_back(batchNum);
  coord_dim_vector.push_back(boxNum * coordNum);
  coord_dim_vector.push_back(CeilX(hwSize * 2 + 32, 32) / 2);
  Shape coordsOutShape(coord_dim_vector);
  TensorDesc coordsDesc = op.GetOutputDesc("coord_data");
  coordsDesc.SetShape(coordsOutShape);
  coordsDesc.SetDataType(inputType);
  std::vector<int64_t> obj_dim_vector;
  obj_dim_vector.push_back(batchNum);
  obj_dim_vector.push_back(CeilX(boxNum * hwSize * 2 + 32, 32) / 2);
  Shape objOutShape(obj_dim_vector);
  TensorDesc objDesc = op.GetOutputDesc("obj_prob");
  objDesc.SetShape(objOutShape);
  objDesc.SetDataType(inputType);
  std::vector<int64_t> class_dim_vector;
  class_dim_vector.push_back(batchNum);
  class_dim_vector.push_back(classNum);
  class_dim_vector.push_back(CeilX(boxNum * hwSize * 2 + 32, 32) / 2);
  Shape classesOutShape(class_dim_vector);
  TensorDesc classesDesc = op.GetOutputDesc("classes_prob");
  classesDesc.SetShape(classesOutShape);
  classesDesc.SetDataType(inputType);
  (void)op.UpdateOutputDesc("coord_data", coordsDesc);
  (void)op.UpdateOutputDesc("obj_prob", objDesc);
  (void)op.UpdateOutputDesc("classes_prob", classesDesc);
  OP_LOGI("yolo", "infer shape end---");
  return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(Yolo, YoloVerify) {
  int64_t boxNum = 0;
  int64_t coordNum = 0;
  int64_t classNum = 0;
  auto inputShape = op.GetInputDesc("x").GetShape().GetDims();
  int64_t channelNum = (int64_t)inputShape[1];

  if (ge::GRAPH_SUCCESS != op.GetAttr("boxes", boxNum)) {
    boxNum = 3;
  }
  if (ge::GRAPH_SUCCESS != op.GetAttr("coords", coordNum)) {
    coordNum = 4;
  }
  if (ge::GRAPH_SUCCESS != op.GetAttr("classes", classNum)) {
    classNum = 80;
  }
  if (boxNum < 1) {
    OP_LOGE(op.GetName().c_str(), "box num %d should be greater than 0", boxNum);
    string realvalue = ConcatString(boxNum);
    OpsAttrValueErrReport(op.GetName().c_str(), "boxNum", "greater than 0", realvalue);
    return GRAPH_FAILED;
  }
  if (coordNum != 4) {
    OP_LOGE(op.GetName().c_str(), "coord num %d should be eq 4", coordNum);
    string realvalue = ConcatString(coordNum);
    OpsAttrValueErrReport(op.GetName().c_str(), "coordNum", "4", realvalue);
    return GRAPH_FAILED;
  }
  if (classNum < 1 || classNum > 1024) {
    OP_LOGE(op.GetName().c_str(), "class num %d should be in range [1, 1024]", classNum);
    string realvalue = ConcatString(classNum);
    OpsAttrValueErrReport(op.GetName().c_str(), "classNum", "range [1, 1024]", realvalue);
    return GRAPH_FAILED;
  }

  if (boxNum * (1 + coordNum + classNum) != channelNum) {
    OP_LOGE(op.GetName().c_str(), "channels(%d) should be (boxes(%d) * (1 + coords(%d) + classes(%d)))", channelNum,
            boxNum, coordNum, classNum);
    string realvalue = ConcatString(channelNum);
    OpsAttrValueErrReport(op.GetName().c_str(), "channels", "must equal with boxes*(1+coords+classes)", realvalue);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Yolo, YoloInferShape);

VERIFY_FUNC_REG(Yolo, YoloVerify);
// ----------------Yolo-------------------

// ----------------YoloV2DetectionOutput-------------------
IMPLEMT_VERIFIER(YoloV2DetectionOutput, YoloV2DetectionOutputVerify) {
  return GRAPH_SUCCESS;
}
IMPLEMT_COMMON_INFERFUNC(YoloV2DetectionOutputInferShape) {
  OP_LOGI(op.GetName().c_str(), "infer shape begin---");
  auto coord_shape = op.GetInputDesc("coord_data").GetShape().GetDims();
  int64_t batch = coord_shape[0];
  DataType input_dtype = op.GetInputDesc("coord_data").GetDataType();
  std::int64_t maxNum = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("post_nms_topn", maxNum)) {
    OP_LOGE(op.GetName().c_str(), "get attr failed");
  }
  std::vector<int64_t> dim_vector;
  dim_vector.push_back(batch);
  dim_vector.push_back(6 * maxNum);
  Shape out_shape_bbox(dim_vector);
  TensorDesc bbox_desc = op.GetOutputDesc("box_out");
  bbox_desc.SetShape(out_shape_bbox);
  bbox_desc.SetDataType(input_dtype);
  Shape out_shape_bbox_out_num({batch, 8});
  TensorDesc num_desc = op.GetOutputDesc("box_out_num");
  num_desc.SetShape(out_shape_bbox_out_num);
  num_desc.SetDataType(ge::DT_INT32);
  (void)op.UpdateOutputDesc("box_out", bbox_desc);
  (void)op.UpdateOutputDesc("box_out_num", num_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(YoloV2DetectionOutput, YoloV2DetectionOutputInferShape);
VERIFY_FUNC_REG(YoloV2DetectionOutput, YoloV2DetectionOutputVerify);
// ----------------YoloV2DetectionOutput-------------------

// ----------------YoloV2DetectionOutputD-------------------
IMPLEMT_VERIFIER(YoloV2DetectionOutputD, YoloV2DetectionOutputDVerify) {
  return GRAPH_SUCCESS;
}
IMPLEMT_COMMON_INFERFUNC(YoloV2DetectionOutputDInferShape) {
  OP_LOGI(op.GetName().c_str(), "infer shape begin---");
  auto coord_shape = op.GetInputDesc("coord_data").GetShape().GetDims();
  int64_t batch = coord_shape[0];
  DataType input_dtype = op.GetInputDesc("coord_data").GetDataType();
  std::int64_t maxNum = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("post_nms_topn", maxNum)) {
    OP_LOGE(op.GetName().c_str(), "get attr failed");
  }
  std::vector<int64_t> dim_vector;
  dim_vector.push_back(batch);
  dim_vector.push_back(6 * maxNum);
  Shape out_shape_bbox(dim_vector);
  TensorDesc bbox_desc = op.GetOutputDesc("box_out");
  bbox_desc.SetShape(out_shape_bbox);
  bbox_desc.SetDataType(input_dtype);
  Shape out_shape_bbox_out_num({batch, 8});
  TensorDesc num_desc = op.GetOutputDesc("box_out_num");
  num_desc.SetShape(out_shape_bbox_out_num);
  num_desc.SetDataType(ge::DT_INT32);
  (void)op.UpdateOutputDesc("box_out", bbox_desc);
  (void)op.UpdateOutputDesc("box_out_num", num_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(YoloV2DetectionOutputD, YoloV2DetectionOutputDInferShape);
VERIFY_FUNC_REG(YoloV2DetectionOutputD, YoloV2DetectionOutputDVerify);
// ----------------YoloV2DetectionOutputD-------------------

// ----------------YoloV3DetectionOutput-------------------
IMPLEMT_VERIFIER(YoloV3DetectionOutput, YoloV3DetectionOutputVerify) {
  return GRAPH_SUCCESS;
}
IMPLEMT_COMMON_INFERFUNC(YoloV3DetectionOutputInferShape) {
  OP_LOGI("yolov3 detection output", "infer shape begin---");
  auto coord_shape = op.GetInputDesc("coord_data_low").GetShape().GetDims();
  int64_t batch = coord_shape[0];
  DataType input_dtype = op.GetInputDesc("coord_data_low").GetDataType();
  std::int64_t maxNum = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("post_nms_topn", maxNum)) {
    OP_LOGE(op.GetName().c_str(), "get attr failed");
  }
  std::vector<int64_t> dim_vector;
  dim_vector.push_back(batch);
  dim_vector.push_back(6 * maxNum);
  Shape out_shape_bbox(dim_vector);
  TensorDesc bbox_desc = op.GetOutputDesc("box_out");
  bbox_desc.SetShape(out_shape_bbox);
  bbox_desc.SetDataType(input_dtype);
  Shape out_shape_bbox_out_num({batch, 8});
  TensorDesc num_desc = op.GetOutputDesc("box_out_num");
  num_desc.SetShape(out_shape_bbox_out_num);
  num_desc.SetDataType(ge::DT_INT32);
  (void)op.UpdateOutputDesc("box_out", bbox_desc);
  (void)op.UpdateOutputDesc("box_out_num", num_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(YoloV3DetectionOutput, YoloV3DetectionOutputInferShape);
VERIFY_FUNC_REG(YoloV3DetectionOutput, YoloV3DetectionOutputVerify);
// ----------------YoloV3DetectionOutput-------------------

// ----------------YoloV3DetectionOutputD-------------------
IMPLEMT_VERIFIER(YoloV3DetectionOutputD, YoloV3DetectionOutputDVerify) {
  return GRAPH_SUCCESS;
}
IMPLEMT_COMMON_INFERFUNC(YoloV3DetectionOutputDInferShape) {
  OP_LOGI(op.GetName().c_str(), "infer shape begin---");
  auto coord_shape = op.GetInputDesc("coord_data_low").GetShape().GetDims();
  int64_t batch = coord_shape[0];
  DataType input_dtype = op.GetInputDesc("coord_data_low").GetDataType();
  std::int64_t maxNum = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("post_nms_topn", maxNum)) {
    OP_LOGE(op.GetName().c_str(), "get attr failed");
  }
  std::vector<int64_t> dim_vector;
  dim_vector.push_back(batch);
  dim_vector.push_back(6 * maxNum);
  Shape out_shape_bbox(dim_vector);
  TensorDesc bbox_desc = op.GetOutputDesc("box_out");
  bbox_desc.SetShape(out_shape_bbox);
  bbox_desc.SetDataType(input_dtype);
  Shape out_shape_bbox_out_num({batch, 8});
  TensorDesc num_desc = op.GetOutputDesc("box_out_num");
  num_desc.SetShape(out_shape_bbox_out_num);
  num_desc.SetDataType(ge::DT_INT32);
  (void)op.UpdateOutputDesc("box_out", bbox_desc);
  (void)op.UpdateOutputDesc("box_out_num", num_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(YoloV3DetectionOutputD, YoloV3DetectionOutputDInferShape);
VERIFY_FUNC_REG(YoloV3DetectionOutputD, YoloV3DetectionOutputDVerify);
// ----------------YoloV3DetectionOutputD-------------------

// ----------------YoloV3DetectionOutputV2------------------
IMPLEMT_VERIFIER(YoloV3DetectionOutputV2, YoloV3DetectionOutputV2Verify) {
  return GRAPH_SUCCESS;
}
IMPLEMT_COMMON_INFERFUNC(YoloV3DetectionOutputV2InferShape) {
  OP_LOGI(op.GetName().c_str(), "infer shape begin---");
  auto coord_shape = op.GetDynamicInputDesc("x", 0).GetShape().GetDims();
  if (coord_shape.empty()) {
    InferShapeOtherErrReport(op.GetName(), "input shape is NULL!");
    OP_LOGE(op.GetName().c_str(), "input shape is NULL!");
    return GRAPH_FAILED;
  }
  int64_t batch = coord_shape[0];
  DataType input_dtype = op.GetDynamicInputDesc("x", 0).GetDataType();
  std::int64_t maxNum = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("post_nms_topn", maxNum)) {
    OP_LOGE(op.GetName().c_str(), "get attr failed");
  }
  std::vector<int64_t> dim_vector;
  dim_vector.push_back(batch);
  std::int64_t outBoxDim = 3;
  if (ge::GRAPH_SUCCESS != op.GetAttr("out_box_dim", outBoxDim)) {
    OP_LOGE(op.GetName().c_str(), "get attr failed");
  }
  if (outBoxDim == 2) {
    dim_vector.push_back(6 * maxNum);
  } else if (outBoxDim == 3) {
    dim_vector.push_back(6);
    dim_vector.push_back(maxNum);
  }
  Shape out_shape_bbox(dim_vector);
  TensorDesc bbox_desc = op.GetOutputDesc("box_out");
  bbox_desc.SetShape(out_shape_bbox);
  bbox_desc.SetDataType(input_dtype);
  Shape out_shape_bbox_out_num({batch, 8});
  TensorDesc num_desc = op.GetOutputDesc("box_out_num");
  num_desc.SetShape(out_shape_bbox_out_num);
  num_desc.SetDataType(ge::DT_INT32);
  (void)op.UpdateOutputDesc("box_out", bbox_desc);
  (void)op.UpdateOutputDesc("box_out_num", num_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(YoloV3DetectionOutputV2, YoloV3DetectionOutputV2InferShape);
VERIFY_FUNC_REG(YoloV3DetectionOutputV2, YoloV3DetectionOutputV2Verify);
// ----------------YoloV3DetectionOutputV2------------------

// ----------------YoloV3DetectionOutputV2D------------------
IMPLEMT_VERIFIER(YoloV3DetectionOutputV2D, YoloV3DetectionOutputV2DVerify) {
  return GRAPH_SUCCESS;
}
IMPLEMT_COMMON_INFERFUNC(YoloV3DetectionOutputV2DInferShape) {
  OP_LOGI(op.GetName().c_str(), "infer shape begin---");
  auto coord_shape = op.GetDynamicInputDesc("x", 0).GetShape().GetDims();
  if (coord_shape.empty()) {
    InferShapeOtherErrReport(op.GetName(), "input shape is NULL!");
    OP_LOGE(op.GetName().c_str(), "input shape is NULL!");
    return GRAPH_FAILED;
  }
  int64_t batch = coord_shape[0];
  DataType input_dtype = op.GetDynamicInputDesc("x", 0).GetDataType();
  std::int64_t maxNum = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("post_nms_topn", maxNum)) {
    OP_LOGE(op.GetName().c_str(), "get attr failed");
  }
  std::vector<int64_t> dim_vector;
  dim_vector.push_back(batch);
  std::int64_t outBoxDim = 3;
  if (ge::GRAPH_SUCCESS != op.GetAttr("out_box_dim", outBoxDim)) {
    OP_LOGE(op.GetName().c_str(), "get attr failed");
  }
  if (outBoxDim == 2) {
    dim_vector.push_back(6 * maxNum);
  } else if (outBoxDim == 3) {
    dim_vector.push_back(6);
    dim_vector.push_back(maxNum);
  }
  Shape out_shape_bbox(dim_vector);
  TensorDesc bbox_desc = op.GetOutputDesc("box_out");
  bbox_desc.SetShape(out_shape_bbox);
  bbox_desc.SetDataType(input_dtype);
  Shape out_shape_bbox_out_num({batch, 8});
  TensorDesc num_desc = op.GetOutputDesc("box_out_num");
  num_desc.SetShape(out_shape_bbox_out_num);
  num_desc.SetDataType(ge::DT_INT32);
  (void)op.UpdateOutputDesc("box_out", bbox_desc);
  (void)op.UpdateOutputDesc("box_out_num", num_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(YoloV3DetectionOutputV2D, YoloV3DetectionOutputV2DInferShape);
VERIFY_FUNC_REG(YoloV3DetectionOutputV2D, YoloV3DetectionOutputV2DVerify);
// ----------------YoloV3DetectionOutputV2D------------------

// ----------------SPP-------------------
int64_t GetLogN(int64_t base) {
  int64_t n = 1;
  int64_t powVal = 1;
  while (powVal < base) {
    powVal *= 2;
    n += 1;
  }
  return n;
}
IMPLEMT_INFERFUNC(SPP, SPPInferShape) {
  OP_LOGI("SPP", "Enter SPP proto inferfunction!");
  vector<int64_t> xShapeDims = op.get_input_desc_x().GetShape().GetDims();
  auto xDtype = op.get_input_desc_x().GetDataType();
  int64_t pyramidHeight = 1;
  if (xShapeDims.empty()) {
    OP_LOGE("SPP", "input shape is NULL!");
    InferShapeOtherErrReport(op.GetName(), "input shape is NULL!");
    return GRAPH_FAILED;
  }
  if (GRAPH_SUCCESS != op.GetAttr("pyramid_height", pyramidHeight)) {
    OP_LOGE("SPP", "GetOpAttr pyramid_height failed!");
    OpsGetAttrErrReport(op.GetName().c_str(), "pyramid_height");
    return GRAPH_FAILED;
  }
  vector<int64_t> yShapeDims;
  yShapeDims.push_back(xShapeDims[0]);
  int64_t dims = 0;
  for (int64_t i = 0; i < pyramidHeight; i++) {
    int64_t hw = pow(2, i);
    dims += hw * hw;
  }
  yShapeDims.push_back(xShapeDims[1] * dims);
  yShapeDims.push_back(1);
  yShapeDims.push_back(1);
  auto outdesc = op.GetOutputDesc("y");
  outdesc.SetShape(Shape(yShapeDims));
  outdesc.SetDataType(xDtype);
  (void)op.update_output_desc_y(outdesc);
  return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(SPP, SPPVerify) {
  int64_t pyramidHeight = 1;
  vector<int64_t> xShapeDims = op.get_input_desc_x().GetShape().GetDims();
  if (xShapeDims.empty()) {
    OP_LOGE("SPP", "input shape is NULL!");
    return GRAPH_FAILED;
  }
  int64_t bottomH = xShapeDims[2];
  int64_t bottomW = xShapeDims[3];
  if (GRAPH_SUCCESS != op.GetAttr("pyramid_height", pyramidHeight)) {
    OP_LOGE("SPP", "GetOpAttr pyramid_height failed!");
    OpsGetAttrErrReport(op.GetName().c_str(), "pyramid_height");
    return GRAPH_FAILED;
  }
  if (pyramidHeight < 1 || pyramidHeight >= 7) {
    OP_LOGE("SPP", "pyramid_height value should be in range[1:7)!");
    string realvalue = ConcatString(pyramidHeight);
    OpsAttrValueErrReport(op.GetName().c_str(), "pyramidHeight", "in range[1:7)", realvalue);
    return GRAPH_FAILED;
  }
  if (pyramidHeight > GetLogN(bottomH) || pyramidHeight > GetLogN(bottomH)) {
    OP_LOGE("SPP", "pyramid_height value should be in range[1:log(H or W)]");
    string realvalue = ConcatString(pyramidHeight);
    OpsAttrValueErrReport(op.GetName().c_str(), "pyramidHeight", "in range[1:log(H or W)]", realvalue);
    return GRAPH_FAILED;
  }
  if (pyramidHeight > 1 && (bottomH > 510 || bottomW > 510)) {
    OP_LOGE("SPP", "feature map size should be in range[1:510]");
    string realvalue = ConcatString(bottomH);
    OpsAttrValueErrReport(op.GetName().c_str(), "feature map", "[1:510]", realvalue);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(SPP, SPPInferShape);
VERIFY_FUNC_REG(SPP, SPPVerify);
// ----------------SPP-------------------

// ----------------DecodeBbox-------------------
IMPLEMT_VERIFIER(DecodeBbox, DecodeBboxVerify) {
  // check format
  Format box_predictions_format = op.GetInputDesc("box_predictions").GetFormat();
  if (box_predictions_format != FORMAT_ND && box_predictions_format != FORMAT_NCHW &&
      box_predictions_format != FORMAT_NHWC) {
    OP_LOGE(op.GetName().c_str(), "format of box_predictions should be ND or NCHW or NHWC");
    return GRAPH_FAILED;
  }
  Format anchors_format = op.GetInputDesc("anchors").GetFormat();
  if (anchors_format != box_predictions_format) {
    OP_LOGE(op.GetName().c_str(), "format of inputs should be equle");
    return GRAPH_FAILED;
  }

  // check attr decode_clip
  float decode_clip = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("decode_clip", decode_clip)) {
    OP_LOGE(op.GetName().c_str(), "get attr decode_clip failed");
    return GRAPH_FAILED;
  }
  if (decode_clip > 10 || decode_clip < 0) {
    OP_LOGE(op.GetName().c_str(), "decode_clip should in [0, 10]");
    return GRAPH_FAILED;
  }

  // check shape
  auto box_predictions_shape = op.GetInputDesc("box_predictions").GetShape().GetDims();
  if (box_predictions_shape.empty()) {
    OP_LOGE(op.GetName().c_str(), "can not get box_predictions shape.");
    return GRAPH_FAILED;
  }
  auto anchors_shape = op.GetInputDesc("anchors").GetShape().GetDims();
  if (anchors_shape.empty()) {
    OP_LOGE(op.GetName().c_str(), "can not get anchors shape.");
    return GRAPH_FAILED;
  }
  // check shape
  int64_t box_predictions_shape_n = 1;
  for (uint32_t i = 0; i < box_predictions_shape.size(); i++) {
    box_predictions_shape_n = box_predictions_shape_n * box_predictions_shape[i];
  }
  int64_t anchors_shape_n = 1;
  for (uint32_t i = 0; i < anchors_shape.size(); i++) {
    anchors_shape_n = anchors_shape_n * anchors_shape[i];
  }
  if (box_predictions_shape_n != anchors_shape_n) {
    OP_LOGE(op.GetName().c_str(), "first dimension of inputs should be equle");
    return GRAPH_FAILED;
  }
  int64_t box_predictions_shape_dimension = box_predictions_shape.size();
  int64_t anchors_shap_dimension = anchors_shape.size();
  int64_t box_predictions_shape_D = box_predictions_shape[box_predictions_shape.size() - 1];
  int64_t box_predictions_shape_N = box_predictions_shape[0];
  int64_t anchors_shape_D = anchors_shape[anchors_shape.size() - 1];
  int64_t anchors_shape_N = anchors_shape[0];
  if (box_predictions_shape_dimension == 3) {
    if (anchors_shap_dimension != 3) {
      OP_LOGE(op.GetName().c_str(), "The input shape not in {(4,C,H,W), (H,W,4)}");
      return GRAPH_FAILED;
    }
    if (box_predictions_shape_D != 4 || anchors_shape_D != 4) {
      OP_LOGE(op.GetName().c_str(), "last dimension of box_predictions and anchors should be FOUR");
      return GRAPH_FAILED;
    }
  }
  if (box_predictions_shape_dimension == 4) {
    if (anchors_shap_dimension != 4) {
      OP_LOGE(op.GetName().c_str(), "The input shape not in {(4,C,H,W), (H,W,4)}");
      return GRAPH_FAILED;
    }
    if (box_predictions_shape_N != 4 || anchors_shape_N != 4) {
      OP_LOGE(op.GetName().c_str(), "first dimension of box_predictions and anchors should be FOUR");
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DecodeBboxInferShapeCommon) {
  auto boxPredictionsShape = op.GetInputDesc("box_predictions").GetShape().GetDims();
  auto anchorsShape = op.GetInputDesc("anchors").GetShape().GetDims();
  int64_t shape_n = 1;
  for (uint32_t i = 0; i < boxPredictionsShape.size(); i++) {
    shape_n = shape_n * boxPredictionsShape[i];
  }
  std::vector<int64_t> dim_vector1;
  if ((boxPredictionsShape.size() == 4) && (boxPredictionsShape[0] == 4)) {
    dim_vector1.push_back(4);
    dim_vector1.push_back(shape_n / 4);
  }
  if ((boxPredictionsShape.size() == 3) && (boxPredictionsShape[2] == 4)) {
    dim_vector1.push_back(shape_n / 4);
    dim_vector1.push_back(4);
  }
  Shape out_shape_decoded_boxes(dim_vector1);
  TensorDesc decoded_boxes_desc = op.GetOutputDesc("decoded_boxes");
  decoded_boxes_desc.SetShape(out_shape_decoded_boxes);
  (void)op.UpdateOutputDesc("decoded_boxes", decoded_boxes_desc);
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(DecodeBbox, DecodeBboxInferShape) {
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DecodeBbox, DecodeBboxInferShapeCommon);

INFER_FUNC_REG(DecodeBbox, DecodeBboxInferShape);

// Registered verify function
VERIFY_FUNC_REG(DecodeBbox, DecodeBboxVerify);
// ----------------DecodeBbox-------------------

// ----------------ClipBoxes-------------------
IMPLEMT_VERIFIER(ClipBoxes, ClipBoxesVerify) {
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ClipBoxesInferShapeCommon) {
  auto boxes_input_shape = op.GetInputDesc("boxes_input").GetShape().GetDims();
  TensorDesc td = op.GetOutputDesc("boxes_output");
  td.SetShape(ge::Shape(boxes_input_shape));
  (void)op.UpdateOutputDesc("boxes_output", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(ClipBoxes, ClipBoxesInferShape) {
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(ClipBoxes, ClipBoxesInferShapeCommon);

INFER_FUNC_REG(ClipBoxes, ClipBoxesInferShape);

// Registered verify function
VERIFY_FUNC_REG(ClipBoxes, ClipBoxesVerify);

IMPLEMT_VERIFIER(ClipBoxesD, ClipBoxesDVerify) {
  // check format
  Format boxes_input_format = op.GetInputDesc("boxes_input").GetFormat();
  if (boxes_input_format != FORMAT_ND) {
    OP_LOGE(op.GetName().c_str(), "format of boxes_input should be ND");
    return GRAPH_FAILED;
  }
  // check shape
  auto boxes_input_shape = op.GetInputDesc("boxes_input").GetShape().GetDims();
  if (boxes_input_shape.empty()) {
    OP_LOGE(op.GetName().c_str(), "can not get boxes_input shape.");
    return GRAPH_FAILED;
  }
  int64_t boxes_input_dimension = boxes_input_shape.size();
  if (boxes_input_dimension != 2) {
    OP_LOGE(op.GetName().c_str(), "The input shape should be two dimension only!");
    return GRAPH_FAILED;
  }
  // check N
  int64_t num = boxes_input_shape[0];
  if (num <= 0 && num > 65500) {
    OP_LOGE(op.GetName().c_str(), "N dimension of inputs should be in [1, 65500]");
    return GRAPH_FAILED;
  }
  // check D
  int64_t boxes_input_D = boxes_input_shape[1];
  if (boxes_input_D != 4) {
    OP_LOGE(op.GetName().c_str(), "second dimension of boxes_input should be FOUR");
    return GRAPH_FAILED;
  }
  // check attr img_size
  std::vector<int64_t> img_size;
  if (op.GetAttr("img_size", img_size) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr img_size failed");
    return GRAPH_FAILED;
  }
  if ((int64_t)img_size.size() != 2) {
    OP_LOGE(op.GetName().c_str(), "img_size should be [img_h, img_w]!");
    return GRAPH_FAILED;
  }
  int64_t img_h = img_size[0];
  int64_t img_w = img_size[1];
  if ((img_h <= 0) || (img_w <= 0)) {
    OP_LOGE(op.GetName().c_str(), "img_h/img_w should be larger than zero!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(ClipBoxesDInferShapeCommon) {
  auto boxes_input_shape = op.GetInputDesc("boxes_input").GetShape().GetDims();
  TensorDesc td = op.GetOutputDesc("boxes_output");
  td.SetShape(ge::Shape(boxes_input_shape));
  (void)op.UpdateOutputDesc("boxes_output", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(ClipBoxesD, ClipBoxesDInferShape) {
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(ClipBoxesD, ClipBoxesDInferShapeCommon);

INFER_FUNC_REG(ClipBoxesD, ClipBoxesDInferShape);

// Registered verify function
VERIFY_FUNC_REG(ClipBoxesD, ClipBoxesDVerify);
// ----------------ClipBoxes-------------------

// ----------------FastrcnnPredictions-------------------
IMPLEMT_VERIFIER(FastrcnnPredictions, FastrcnnPredictionsVerify) {
  // check format
  Format score_format = op.GetInputDesc("score").GetFormat();
  if (score_format != FORMAT_ND) {
    OpsInputFormatErrReport(op.GetName(), "Score Format", "ND", ConcatString(score_format));
    OP_LOGE(op.GetName().c_str(), "format of score should be ND");
    return GRAPH_FAILED;
  }
  // check attr nms_threshold
  float nms_threshold = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("nms_threshold", nms_threshold)) {
    OpsGetAttrErrReport(op.GetName(), "nms threshold");
    OP_LOGE(op.GetName().c_str(), "get attr nms_threshold failed");
    return GRAPH_FAILED;
  }
  if (nms_threshold > 1 || nms_threshold < 0) {
    OpsAttrValueErrReport(op.GetName(), "nms_threshold", "[0, 1]", ConcatString(nms_threshold));
    OP_LOGE(op.GetName().c_str(), "nms_threshold should in [0, 1]");
    return GRAPH_FAILED;
  }
  // check attr score_threshold
  float score_threshold = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("score_threshold", score_threshold)) {
    OpsGetAttrErrReport(op.GetName(), "score_threshold");
    OP_LOGE(op.GetName().c_str(), "get attr score_threshold failed");
    return GRAPH_FAILED;
  }
  if (score_threshold > 1 || score_threshold < 0) {
    OP_LOGE(op.GetName().c_str(), "score_threshold should in [0, 1]");
    OpsAttrValueErrReport(op.GetName(), "score_threshold", "[0, 1]", ConcatString(nms_threshold));
    return GRAPH_FAILED;
  }
  // check shape
  auto score_shape = op.GetInputDesc("score").GetShape().GetDims();
  if (score_shape.empty()) {
    OP_LOGE(op.GetName().c_str(), "can not get score shape.");
    InferShapeOtherErrReport(op.GetName(), "Score shape is empty, pls check!");
    return GRAPH_FAILED;
  }
  auto rois_shape = op.GetInputDesc("rois").GetShape().GetDims();
  if (rois_shape.empty()) {
    InferShapeOtherErrReport(op.GetName(), "rois shape is empty, pls check!");
    OP_LOGE(op.GetName().c_str(), "can not get rois shape.");
    return GRAPH_FAILED;
  }
  int64_t score_shape_dimension = score_shape.size();
  int64_t rois_shape_dimension = rois_shape.size();
  if (score_shape_dimension != 2 || rois_shape_dimension != 2) {
    OpsTwoInputShapeErrReport(op.GetName(), "score_shape_dimension", "rois_shape_dimension",
                              "score_shape_dimension != 2 or rois_shape_dimension != 2, pls check!");
    OP_LOGE(op.GetName().c_str(), "The input shape should be two dimension only!");
    return GRAPH_FAILED;
  }
  // check N
  int64_t num = score_shape[0];
  if (num != 16 && num != 32 && num != 96) {
    OpsInputShapeErrReport(op.GetName(), "16 or 32 or 96", "first Dim of Score Shape", ConcatString(num));
    OP_LOGE(op.GetName().c_str(), "first dimension of score should be 16 or 32 or 96");
    return GRAPH_FAILED;
  }
  // check classes
  int64_t classes = score_shape[1] - 1;
  if ((classes < 1) || (classes > 32)) {
    OP_LOGE(op.GetName().c_str(), "second dimension of score should in [1, 32]");
    OpsInputShapeErrReport(op.GetName(), "[1, 32]", "Second Dim of Score Shape", ConcatString(classes));
    return GRAPH_FAILED;
  }
  // check D
  int64_t rois_shape_D = rois_shape[1];
  int64_t rois_shape_N = rois_shape[0];
  if (rois_shape_D != 4) {
    OpsInputShapeErrReport(op.GetName(), "4", "Dim of ROIS Shape", ConcatString(rois_shape_D));
    OP_LOGE(op.GetName().c_str(), "second dimension of rois should be FOUR");
    return GRAPH_FAILED;
  }
  if (rois_shape_N != num * classes) {
    OpsTwoInputShapeErrReport(op.GetName(), "Rois Shape", "Score Shape * Classes",
                              "first dimension of rois not equal first dimension mul second dimension of score");
    OP_LOGE(op.GetName().c_str(),
            "first dimension of rois should be consistent to first dimension mul second dimension of score");
    return GRAPH_FAILED;
  }
  // check attr k
  int64_t k = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("k", k)) {
    OP_LOGE(op.GetName().c_str(), "get attr k failed");
    OpsGetAttrErrReport(op.GetName(), "Attribute K");
    return GRAPH_FAILED;
  }
  if (k != num) {
    OP_LOGE(op.GetName().c_str(), "k should be equle to N");
    OpsAttrValueErrReport(op.GetName(), "Attribute K", ConcatString(num), ConcatString(k));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(FastrcnnPredictionsInferShapeCommon) {
  OP_LOGI("fastrcnn_predictions", "infer shape begin---");
  auto score_shape = op.GetInputDesc("score").GetShape().GetDims();
  int64_t num = score_shape[0];
  std::vector<int64_t> dim_vector;
  dim_vector.push_back(num);
  dim_vector.push_back(4);
  Shape out_shape_sorted_rois(dim_vector);
  TensorDesc sorted_rois_desc = op.GetOutputDesc("sorted_rois");
  sorted_rois_desc.SetShape(out_shape_sorted_rois);
  (void)op.UpdateOutputDesc("sorted_rois", sorted_rois_desc);
  std::vector<int64_t> dim_vector1;
  dim_vector1.push_back(num);
  dim_vector1.push_back(1);
  Shape out_shape_sorted_scores(dim_vector1);
  TensorDesc sorted_scores_desc = op.GetOutputDesc("sorted_scores");
  sorted_scores_desc.SetShape(out_shape_sorted_scores);
  (void)op.UpdateOutputDesc("sorted_scores", sorted_scores_desc);
  std::vector<int64_t> dim_vector2;
  dim_vector2.push_back(num);
  dim_vector2.push_back(1);
  Shape out_shape_sorted_classes(dim_vector2);
  TensorDesc sorted_classes_desc = op.GetOutputDesc("sorted_classes");
  sorted_classes_desc.SetShape(out_shape_sorted_classes);
  (void)op.UpdateOutputDesc("sorted_classes", sorted_classes_desc);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(FastrcnnPredictions, FastrcnnPredictionsInferShapeCommon);

// Registered verify function
VERIFY_FUNC_REG(FastrcnnPredictions, FastrcnnPredictionsVerify);
// ----------------FastrcnnPredictions-------------------

// ----------------RpnProposals-------------------
IMPLEMT_VERIFIER(RpnProposals, RpnProposalsVerify) {
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(RpnProposalsInferShapeCommon) {
  OP_LOGI("rpn_proposals", "infer shape begin---");
  DataType input_dtype = op.GetInputDesc("rois").GetDataType();
  int64_t post_nms_num = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("post_nms_num", post_nms_num)) {
    OP_LOGE(op.GetName().c_str(), "get attr post_nms_num failed");
  }
  std::vector<int64_t> dim_vector;
  dim_vector.push_back(post_nms_num);
  dim_vector.push_back(4);
  Shape out_shape_sorted_box(dim_vector);
  TensorDesc sorted_box_desc = op.GetOutputDesc("sorted_box");
  sorted_box_desc.SetShape(out_shape_sorted_box);
  sorted_box_desc.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("sorted_box", sorted_box_desc);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(RpnProposals, RpnProposalsInferShapeCommon);

// Registered verify function
VERIFY_FUNC_REG(RpnProposals, RpnProposalsVerify);
IMPLEMT_VERIFIER(RpnProposalsD, RpnProposalsDVerify) {
  // check format
  Format rois_format = op.GetInputDesc("rois").GetFormat();
  if (rois_format != FORMAT_ND) {
    OP_LOGE(op.GetName().c_str(), "format of rois should be ND");
    return GRAPH_FAILED;
  }
  // check shape
  auto score_shape = op.GetInputDesc("cls_bg_prob").GetShape().GetDims();
  if (score_shape.empty()) {
    OP_LOGE(op.GetName().c_str(), "can not get cls_bg_prob shape.");
    return GRAPH_FAILED;
  }
  auto rois_shape = op.GetInputDesc("rois").GetShape().GetDims();
  if (rois_shape.empty()) {
    OP_LOGE(op.GetName().c_str(), "can not get rois shape.");
    return GRAPH_FAILED;
  }
  int64_t score_shape_dimension = score_shape.size();
  int64_t rois_shape_dimension = rois_shape.size();
  if (score_shape_dimension != 2 || rois_shape_dimension != 2) {
    OP_LOGE(op.GetName().c_str(), "The input shape should be two dimension only!");
    return GRAPH_FAILED;
  }
  // check D
  int64_t rois_shape_D = rois_shape[0];
  if (rois_shape_D != 4) {
    OP_LOGE(op.GetName().c_str(), "second dimension of rois should be FOUR");
    return GRAPH_FAILED;
  }
  // check attr score_threshold
  float score_threshold = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("score_threshold", score_threshold)) {
    OP_LOGE(op.GetName().c_str(), "get attr score_threshold failed");
    return GRAPH_FAILED;
  }
  // check attr nms_threshold
  float nms_threshold = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("nms_threshold", nms_threshold)) {
    OP_LOGE(op.GetName().c_str(), "get attr nms_threshold failed");
    return GRAPH_FAILED;
  }
  if (nms_threshold > 1 || nms_threshold < 0) {
    OP_LOGE(op.GetName().c_str(), "nms_threshold should in [0, 1]");
    return GRAPH_FAILED;
  }
  // check attr k
  int64_t k = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("k", k)) {
    OP_LOGE(op.GetName().c_str(), "get attr k failed");
    return GRAPH_FAILED;
  }
  // check attr min_size
  float min_size = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("min_size", min_size)) {
    OP_LOGE(op.GetName().c_str(), "get attr min_size failed");
    return GRAPH_FAILED;
  }
  if (min_size < 0) {
    OP_LOGE(op.GetName().c_str(), "min_size should greater than 0");
    return GRAPH_FAILED;
  }
  // check attr img_size
  std::vector<int64_t> img_size;
  if (op.GetAttr("img_size", img_size) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "get attr img_size failed");
    return GRAPH_FAILED;
  }
  if ((int64_t)img_size.size() != 2) {
    OP_LOGE(op.GetName().c_str(), "img_size should be [img_h, img_w]!");
    return GRAPH_FAILED;
  }
  int64_t img_h = img_size[0];
  int64_t img_w = img_size[1];
  if ((img_h <= 0) || (img_w <= 0)) {
    OP_LOGE(op.GetName().c_str(), "img_h/img_w should be larger than zero!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(RpnProposalsDInferShapeCommon) {
  OP_LOGI("rpn_proposals_d", "infer shape begin---");
  int64_t post_nms_num = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("post_nms_num", post_nms_num)) {
    OP_LOGE(op.GetName().c_str(), "get attr post_nms_num failed");
  }
  std::vector<int64_t> dim_vector;
  dim_vector.push_back(post_nms_num);
  dim_vector.push_back(4);
  Shape out_shape_sorted_box(dim_vector);
  TensorDesc sorted_box_desc = op.GetOutputDesc("sorted_box");
  sorted_box_desc.SetShape(out_shape_sorted_box);
  (void)op.UpdateOutputDesc("sorted_box", sorted_box_desc);
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(RpnProposalsD, RpnProposalsDInferShapeCommon);

// Registered verify function
VERIFY_FUNC_REG(RpnProposalsD, RpnProposalsDVerify);
// ----------------RpnProposals-------------------

// ----------------DecodeBoundariesTarget-------------------
IMPLEMT_VERIFIER(DecodeBoundariesTarget, DecodeBoundariesTargetVerify) {
  // check format
  Format boundary_predictions_format = op.GetInputDesc("boundary_predictions").GetFormat();
  if (boundary_predictions_format != FORMAT_ND) {
    OP_LOGE(op.GetName().c_str(), "format of boundary_predictions should be ND");
    return GRAPH_FAILED;
  }
  Format anchors_format = op.GetInputDesc("anchors").GetFormat();
  if (anchors_format != boundary_predictions_format) {
    OP_LOGE(op.GetName().c_str(), "format of anchors should be equle");
    return GRAPH_FAILED;
  }
  // check shape
  auto boundary_predictions_shape = op.GetInputDesc("boundary_predictions").GetShape().GetDims();
  if (boundary_predictions_shape.empty()) {
    OP_LOGE(op.GetName().c_str(), "can not get boundary_predictions shape.");
    return GRAPH_FAILED;
  }
  auto anchors_shape = op.GetInputDesc("anchors").GetShape().GetDims();
  if (anchors_shape.empty()) {
    OP_LOGE(op.GetName().c_str(), "can not get anchors shape.");
    return GRAPH_FAILED;
  }
  if (boundary_predictions_shape[0] != anchors_shape[0]) {
    OP_LOGE(op.GetName().c_str(), "the two inputs dim[0] should be equle");
    return GRAPH_FAILED;
  }
  if (boundary_predictions_shape[1] != 1) {
    OP_LOGE(op.GetName().c_str(), "last dimension of boundary_predictions should be ONE");
    return GRAPH_FAILED;
  }
  if (anchors_shape[1] != 4) {
    OP_LOGE(op.GetName().c_str(), "last dimension of anchors should be FOUR");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DecodeBoundariesTargetInferShapeCommon) {
  auto boundary_predictions_shape = op.GetInputDesc("boundary_predictions").GetShape().GetDims();
  auto anchors_shape = op.GetInputDesc("anchors").GetShape().GetDims();
  TensorDesc td = op.GetOutputDesc("boundary_predictions");
  td.SetShape(ge::Shape(boundary_predictions_shape));
  (void)op.UpdateOutputDesc("boundary_encoded", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(DecodeBoundariesTarget, DecodeBoundariesTargetInferShape) {
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DecodeBoundariesTarget, DecodeBoundariesTargetInferShapeCommon);

INFER_FUNC_REG(DecodeBoundariesTarget, DecodeBoundariesTargetInferShape);

// Registered verify function
VERIFY_FUNC_REG(DecodeBoundariesTarget, DecodeBoundariesTargetVerify);
// ----------------DecodeBoundariesTarget-------------------

// ----------------DecodeCornerpointsTargetBG-------------------
IMPLEMT_VERIFIER(DecodeCornerpointsTargetBG, DecodeCornerpointsTargetBGVerify) {
  // check shape
  auto keypoints_predictions_shape = op.GetInputDesc("keypoints_prediction").GetShape().GetDims();
  if (keypoints_predictions_shape.empty()) {
    OP_LOGE(op.GetName().c_str(), "can not get keypoints_predictions shape.");
    return GRAPH_FAILED;
  }
  auto anchors_shape = op.GetInputDesc("anchors").GetShape().GetDims();
  if (anchors_shape.empty()) {
    OP_LOGE(op.GetName().c_str(), "can not get anchors shape.");
    return GRAPH_FAILED;
  }
  int64_t keypoints_predictions_shape_dimension = keypoints_predictions_shape.size();
  int64_t anchors_shape_dimension = anchors_shape.size();
  if (keypoints_predictions_shape_dimension != 2 || anchors_shape_dimension != 2) {
    OP_LOGE(op.GetName().c_str(), "The input shape should be two dimension only!");
    return GRAPH_FAILED;
  }
  // check D
  int64_t anchors_shape_D = anchors_shape[1];
  int64_t keypoints_predictions_shape_D = keypoints_predictions_shape[1];
  if (anchors_shape_D != 4) {
    OP_LOGE(op.GetName().c_str(), "second dimension of anchors should be FOUR");
    return GRAPH_FAILED;
  }
  if (keypoints_predictions_shape_D != 4) {
    OP_LOGE(op.GetName().c_str(), "second dimension of keypoints_predictions should be FOUR");
    return GRAPH_FAILED;
  }
  // check N
  int64_t anchors_shape_N = anchors_shape[0];
  int64_t keypoints_predictions_shape_N = keypoints_predictions_shape[0];
  if (anchors_shape_N != keypoints_predictions_shape_N) {
    OP_LOGE(op.GetName().c_str(), "first dimension of input should be consistent");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DecodeCornerpointsTargetBGInferShapeCommon) {
  auto keypoints_prediction_shape = op.GetInputDesc("keypoints_prediction").GetShape().GetDims();
  auto anchors_shape = op.GetInputDesc("anchors").GetShape().GetDims();
  TensorDesc td = op.GetOutputDesc("keypoints_decoded");
  td.SetShape(ge::Shape(keypoints_prediction_shape));
  (void)op.UpdateOutputDesc("keypoints_decoded", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(DecodeCornerpointsTargetBG, DecodeCornerpointsTargetBGInferShape) {
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DecodeCornerpointsTargetBG, DecodeCornerpointsTargetBGInferShapeCommon);

INFER_FUNC_REG(DecodeCornerpointsTargetBG, DecodeCornerpointsTargetBGInferShape);

// Registered verify function
VERIFY_FUNC_REG(DecodeCornerpointsTargetBG, DecodeCornerpointsTargetBGVerify);
// ----------------DecodeCornerpointsTargetBG-------------------

// ----------------DecodeCornerpointsTargetWrtCenterV1-------------------
IMPLEMT_VERIFIER(DecodeCornerpointsTargetWrtCenterV1, DecodeCornerpointsTargetWrtCenterV1Verify) {
  // check shape
  auto keypoints_predictions_shape = op.GetInputDesc("keypoints_prediction").GetShape().GetDims();
  if (keypoints_predictions_shape.empty()) {
    OP_LOGE(op.GetName().c_str(), "can not get keypoints_predictions shape.");
    return GRAPH_FAILED;
  }
  auto anchors_shape = op.GetInputDesc("anchors").GetShape().GetDims();
  if (anchors_shape.empty()) {
    OP_LOGE(op.GetName().c_str(), "can not get anchors shape.");
    return GRAPH_FAILED;
  }
  int64_t keypoints_predictions_shape_dimension = keypoints_predictions_shape.size();
  int64_t anchors_shape_dimension = anchors_shape.size();
  if (keypoints_predictions_shape_dimension != 2 || anchors_shape_dimension != 2) {
    OP_LOGE(op.GetName().c_str(), "The input shape should be two dimension only!");
    return GRAPH_FAILED;
  }
  // check D
  int64_t anchors_shape_D = anchors_shape[1];
  int64_t keypoints_predictions_shape_D = keypoints_predictions_shape[1];
  if (anchors_shape_D != 4) {
    OP_LOGE(op.GetName().c_str(), "second dimension of anchors should be FOUR");
    return GRAPH_FAILED;
  }
  if (keypoints_predictions_shape_D != 8) {
    OP_LOGE(op.GetName().c_str(), "second dimension of keypoints_predictions should be EIGHT");
    return GRAPH_FAILED;
  }
  // check N
  int64_t anchors_shape_N = anchors_shape[0];
  int64_t keypoints_predictions_shape_N = keypoints_predictions_shape[0];
  if (anchors_shape_N != keypoints_predictions_shape_N) {
    OP_LOGE(op.GetName().c_str(), "first dimension of input should be consistent");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DecodeCornerpointsTargetWrtCenterV1InferShapeCommon) {
  auto keypoints_prediction_shape = op.GetInputDesc("keypoints_prediction").GetShape().GetDims();
  auto anchors_shape = op.GetInputDesc("anchors").GetShape().GetDims();
  TensorDesc td = op.GetOutputDesc("keypoints_decoded");
  td.SetShape(ge::Shape(keypoints_prediction_shape));
  (void)op.UpdateOutputDesc("keypoints_decoded", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(DecodeCornerpointsTargetWrtCenterV1, DecodeCornerpointsTargetWrtCenterV1InferShape) {
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DecodeCornerpointsTargetWrtCenterV1, DecodeCornerpointsTargetWrtCenterV1InferShapeCommon);

INFER_FUNC_REG(DecodeCornerpointsTargetWrtCenterV1, DecodeCornerpointsTargetWrtCenterV1InferShape);

// Registered verify function
VERIFY_FUNC_REG(DecodeCornerpointsTargetWrtCenterV1, DecodeCornerpointsTargetWrtCenterV1Verify);
// ----------------DecodeCornerpointsTargetWrtCenterV1-------------------

// ----------------DecodeWheelsTarget-------------------
IMPLEMT_VERIFIER(DecodeWheelsTarget, DecodeWheelsTargetVerify) {
  // check shape
  auto boundary_predictions_shape = op.GetInputDesc("boundary_predictions").GetShape().GetDims();
  if (boundary_predictions_shape.empty()) {
    OP_LOGE(op.GetName().c_str(), "can not get boundary_predictions shape.");
    return GRAPH_FAILED;
  }
  auto anchors_shape = op.GetInputDesc("anchors").GetShape().GetDims();
  if (anchors_shape.empty()) {
    OP_LOGE(op.GetName().c_str(), "can not get anchors shape.");
    return GRAPH_FAILED;
  }
  int64_t boundary_predictions_shape_dimension = boundary_predictions_shape.size();
  int64_t anchors_shape_dimension = anchors_shape.size();
  if (boundary_predictions_shape_dimension != 2 || anchors_shape_dimension != 2) {
    OP_LOGE(op.GetName().c_str(), "The input shape should be two dimension only!");
    return GRAPH_FAILED;
  }
  // check D
  int64_t anchors_shape_D = anchors_shape[1];
  int64_t boundary_predictions_shape_D = boundary_predictions_shape[1];
  if (anchors_shape_D != 4) {
    OP_LOGE(op.GetName().c_str(), "second dimension of anchors should be FOUR");
    return GRAPH_FAILED;
  }
  if (boundary_predictions_shape_D != 8) {
    OP_LOGE(op.GetName().c_str(), "second dimension of boundary_predictions should be EIGHT");
    return GRAPH_FAILED;
  }
  // check N
  int64_t anchors_shape_N = anchors_shape[0];
  int64_t boundary_predictions_shape_N = boundary_predictions_shape[0];
  if (anchors_shape_N != boundary_predictions_shape_N) {
    OP_LOGE(op.GetName().c_str(), "first dimension of input should be consistent");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DecodeWheelsTargetInferShapeCommon) {
  auto boundary_predictions_shape = op.GetInputDesc("boundary_predictions").GetShape().GetDims();
  auto anchors_shape = op.GetInputDesc("anchors").GetShape().GetDims();
  TensorDesc td = op.GetOutputDesc("boundary_encoded");
  td.SetShape(ge::Shape(boundary_predictions_shape));
  (void)op.UpdateOutputDesc("boundary_encoded", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(DecodeWheelsTarget, DecodeWheelsTargetInferShape) {
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DecodeWheelsTarget, DecodeWheelsTargetInferShapeCommon);

INFER_FUNC_REG(DecodeWheelsTarget, DecodeWheelsTargetInferShape);

// Registered verify function
VERIFY_FUNC_REG(DecodeWheelsTarget, DecodeWheelsTargetVerify);
// ----------------DecodeWheelsTarget-------------------

// ----------------------BatchMultiClassNonMaxSuppression----------------------
IMPLEMT_VERIFIER(BatchMultiClassNonMaxSuppression, BatchMultiClassNonMaxSuppressionVerify) {
  bool transposeBox;
  if (GRAPH_SUCCESS != op.GetAttr("transpose_box", transposeBox)) {
    OP_LOGW(op.GetName().c_str(), "GetAttr of transpose_box failed. set default false");
    transposeBox = false;
  }
  if (transposeBox) {
    // check attr transpose_box
    map<string, string> err_map;
    err_map["param_name"] = "transpose_box";
    err_map["op_name"] = "BatchMultiClassNonMaxSuppression";
    err_map["expected_value"] = "false";
    err_map["input_value"] = "true";
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);

    return GRAPH_FAILED;
  }
  auto inputShape = op.GetInputDesc("boxes").GetShape().GetDims();
  // check input shape
  if (inputShape.empty()) {
    OP_LOGE(op.GetName().c_str(), "can not get input boxes shape.");
    return GRAPH_FAILED;
  }
  if (inputShape[inputShape.size() - 1] != 4) {
    std::stringstream tmpVecToStr;
    std::string wrongString;
    tmpVecToStr.str("");
    tmpVecToStr << "[";
    tmpVecToStr << std::to_string(inputShape[0]);
    for (size_t i = 1; i < inputShape.size(); i++) {
      tmpVecToStr << ",";
      tmpVecToStr << std::to_string(inputShape[i]);
    }
    tmpVecToStr << "]";
    wrongString = tmpVecToStr.str();
    tmpVecToStr.str("");
    tmpVecToStr << "[";
    tmpVecToStr << std::to_string(inputShape[0]);
    std::string correctString;
    for (size_t i = 1; i < inputShape.size() - 1; i++) {
      tmpVecToStr << ",";
      tmpVecToStr << std::to_string(inputShape[i]);
    }
    tmpVecToStr << ",";
    tmpVecToStr << std::to_string(4);
    tmpVecToStr << "]";
    correctString = tmpVecToStr.str();
    tmpVecToStr.str("");
    map<string, string> err_map;
    err_map["index"] = "0";
    err_map["opname"] = "BatchMultiClassNonMaxSuppression";
    err_map["wrong_shape"] = wrongString.c_str();
    err_map["correct_shape"] = correctString.c_str();
    std::string report_error_code = "E35000";
    OP_LOGE(op.GetName().c_str(), "the last dim of boxes shape must be 4, while is %d",
            inputShape[inputShape.size() - 1]);
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);

    return GRAPH_FAILED;
  }

  if (inputShape.size() == 4) {
    // check class num in boxes
    auto inputScoreShape = op.GetInputDesc("scores").GetShape().GetDims();
    // check input shape
    if (inputScoreShape.empty()) {
      OP_LOGE(op.GetName().c_str(), "can not get input score shape.");
      return GRAPH_FAILED;
    }
    auto scoreClassNum = inputScoreShape[inputScoreShape.size() - 1];
    auto boxesClassNum = inputShape[2];
    if ((boxesClassNum != scoreClassNum) && (boxesClassNum != 1)) {
      OP_LOGE(op.GetName().c_str(), "the ClassNum in input score is %d, while ClassNum in boxes is %d", scoreClassNum,
              boxesClassNum);
      map<string, string> err_map;
      err_map["param_name"] = "ClassNum in boxes";
      err_map["op_name"] = "BatchMultiClassNonMaxSuppression";
      err_map["expected_value"] = "1 or ClassNum in scores";
      err_map["input_value"] = std::to_string(boxesClassNum);
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);

      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(BatchMultiClassNonMaxSuppressionInferShape) {
  std::int64_t maxOutNum = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("max_total_size", maxOutNum)) {
    OP_LOGE(op.GetName().c_str(), "get attr failed");
    return false;
  }
  auto inputScoreShape = op.GetInputDesc("scores").GetShape().GetDims();
  DataType inputDtype = op.GetInputDesc("scores").GetDataType();

  // check input shape
  if (inputScoreShape.empty()) {
    OP_LOGE(op.GetName().c_str(), "can not get input scores shape.");
    return GRAPH_FAILED;
  }

  bool transposeBox;
  if (GRAPH_SUCCESS != op.GetAttr("transpose_box", transposeBox)) {
    OP_LOGW(op.GetName().c_str(), "GetAttr of transpose_box failed. set default false");
    transposeBox = false;
  }

  vector<int64_t> nmsedBoxesShape;
  nmsedBoxesShape.push_back(inputScoreShape[0]);
  if (!transposeBox) {
    nmsedBoxesShape.push_back(maxOutNum);
    nmsedBoxesShape.push_back(4);
  } else {
    nmsedBoxesShape.push_back(4);
    nmsedBoxesShape.push_back(maxOutNum);
  }
  TensorDesc tdBoxes = op.GetOutputDesc("nmsed_boxes");
  tdBoxes.SetShape(ge::Shape(nmsedBoxesShape));
  tdBoxes.SetDataType(inputDtype);
  (void)op.UpdateOutputDesc("nmsed_boxes", tdBoxes);

  vector<int64_t> nmsedScoreShape;
  nmsedScoreShape.push_back(inputScoreShape[0]);
  nmsedScoreShape.push_back(maxOutNum);
  TensorDesc tdScore = op.GetOutputDesc("nmsed_scores");
  tdScore.SetShape(ge::Shape(nmsedScoreShape));
  tdScore.SetDataType(inputDtype);
  (void)op.UpdateOutputDesc("nmsed_scores", tdScore);

  TensorDesc tdClass = op.GetOutputDesc("nmsed_classes");
  tdClass.SetShape(ge::Shape(nmsedScoreShape));
  tdClass.SetDataType(inputDtype);
  (void)op.UpdateOutputDesc("nmsed_classes", tdClass);

  vector<int64_t> nmsedValidNum;
  nmsedValidNum.push_back(inputScoreShape[0]);
  TensorDesc tdNum = op.GetOutputDesc("nmsed_num");
  tdNum.SetShape(ge::Shape(nmsedValidNum));
  tdNum.SetDataType(DT_INT32);
  (void)op.UpdateOutputDesc("nmsed_num", tdNum);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BatchMultiClassNonMaxSuppression, BatchMultiClassNonMaxSuppressionInferShape);
VERIFY_FUNC_REG(BatchMultiClassNonMaxSuppression, BatchMultiClassNonMaxSuppressionVerify);
// ----------------------BatchMultiClassNonMaxSuppression END----------------------

// ----------------NormalizeBBox-------------------
COMMON_INFER_FUNC_REG(NormalizeBBox, ELMTWISE_INFER_SHAPEANDTYPE("boxes", "y"));
// ----------------NormalizeBBox-------------------

// ----------------ToAbsoluteBBox-------------------
COMMON_INFER_FUNC_REG(ToAbsoluteBBox, ELMTWISE_INFER_SHAPEANDTYPE("normalized_boxes", "y"));
// ----------------ToAbsoluteBBox-------------------

// ----------------DecodeBboxV2-------------------
IMPLEMT_VERIFIER(DecodeBboxV2, DecodeBboxV2Verify) {
  // check shape
  auto box_predictions_shape = op.GetInputDesc("boxes").GetShape().GetDims();
  if (box_predictions_shape.empty()) {
    OP_LOGE(op.GetName().c_str(), "can not get boxes shape.");
    return GRAPH_FAILED;
  }
  auto anchors_shape = op.GetInputDesc("anchors").GetShape().GetDims();
  if (anchors_shape.empty()) {
    OP_LOGE(op.GetName().c_str(), "can not get anchors shape.");
    return GRAPH_FAILED;
  }

  // check attr decode_clip
  float decode_clip = 0;
  if (ge::GRAPH_SUCCESS != op.GetAttr("decode_clip", decode_clip)) {
    OP_LOGE(op.GetName().c_str(), "get attr decode_clip failed");
    return GRAPH_FAILED;
  }
  if (decode_clip > 10 || decode_clip < 0) {
    OP_LOGE(op.GetName().c_str(), "decode_clip should in [0, 10]");
    return GRAPH_FAILED;
  }

  // check attr reversed_box
  bool reversed_box = false;
  if (ge::GRAPH_SUCCESS != op.GetAttr("reversed_box", reversed_box)) {
    OP_LOGE(op.GetName().c_str(), "get attr reversed_box failed");
    return GRAPH_FAILED;
  }

  // check attr scales
  std::vector<float> scales_list;
  if (ge::GRAPH_SUCCESS != op.GetAttr("scales", scales_list)) {
    OP_LOGE(op.GetName().c_str(), "get attr scales failed");
    return GRAPH_FAILED;
  }
  if (scales_list.size() != 4) {
    OP_LOGE(op.GetName().c_str(), "scales list dimension should be 4");
    return GRAPH_FAILED;
  }
  // check shape
  int64_t box_predictions_shape_n = 1;
  for (uint32_t i = 0; i < box_predictions_shape.size(); i++) {
    box_predictions_shape_n = box_predictions_shape_n * box_predictions_shape[i];
  }
  int64_t anchors_shape_n = 1;
  for (uint32_t i = 0; i < anchors_shape.size(); i++) {
    anchors_shape_n = anchors_shape_n * anchors_shape[i];
  }
  if (box_predictions_shape_n != anchors_shape_n) {
    OP_LOGE(op.GetName().c_str(), "first dimension of inputs should be equle");
    return GRAPH_FAILED;
  }
  int64_t box_predictions_shape_D = box_predictions_shape[box_predictions_shape.size() - 1];
  int64_t box_predictions_shape_N = box_predictions_shape[0];
  int64_t anchors_shape_D = anchors_shape[anchors_shape.size() - 1];
  int64_t anchors_shape_N = anchors_shape[0];
  if (reversed_box == false) {
    if (box_predictions_shape_D != 4 || anchors_shape_D != 4) {
      OP_LOGE(op.GetName().c_str(), "The input shape not in {(N4), (N4)}");
      return GRAPH_FAILED;
    }
  }
  if (reversed_box == true) {
    if (box_predictions_shape_N != 4 || anchors_shape_N != 4) {
      OP_LOGE(op.GetName().c_str(), "The input shape not in {(4N), (4N)}");
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DecodeBboxV2, ELMTWISE_INFER_SHAPEANDTYPE("boxes", "y"));

// Registered verify function
VERIFY_FUNC_REG(DecodeBboxV2, DecodeBboxV2Verify);
// ----------------DecodeBboxV2-------------------
// --------------------sort----------------------------
IMPLEMT_INFERFUNC(Sort, SortInferShape) {
  TensorDesc tensordesc_input = op.GetInputDesc("x");
  Shape input_shape = tensordesc_input.GetShape();
  DataType input_dtype = tensordesc_input.GetDataType();
  std::vector<int64_t> dims_input = input_shape.GetDims();

  TensorDesc tensordesc_output1 = op.GetOutputDesc("y1");
  TensorDesc tensordesc_output2 = op.GetOutputDesc("y2");

  tensordesc_output1.SetShape(ge::Shape(dims_input));
  tensordesc_output2.SetShape(ge::Shape(dims_input));

  tensordesc_output1.SetDataType(input_dtype);
  tensordesc_output2.SetDataType(DT_INT32);

  (void)op.UpdateOutputDesc("y1", tensordesc_output1);
  (void)op.UpdateOutputDesc("y2", tensordesc_output2);

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Sort, SortVerify) {
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Sort, SortInferShape);
VERIFY_FUNC_REG(Sort, SortVerify);
// --------------------sort---------------------------
}  // namespace ge
