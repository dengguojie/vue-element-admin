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
 * \file ocr_ops.cpp
 * \brief
 */
#include "inc/ocr_ops.h"

#include <vector>
#include <string>

#include "op_log.h"
#include "util/util.h"

namespace {
const int64_t kChannelNum = 3;
const int64_t kResizeH = 64;
const int64_t kResizeW = 512;
}  // namespace

namespace ge {
IMPLEMT_COMMON_INFERFUNC(BatchEnqueueInferShape) {
  // main part of shape infer
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr td = op_desc->MutableOutputDesc("enqueue_count");
  std::vector<int64_t> scalar;
  td->SetShape(GeShape(scalar));
  td->SetDataType(DT_INT32);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BatchEnqueue, BatchEnqueueInferShape);

IMPLEMT_COMMON_INFERFUNC(OCRRecognitionPreHandleInferShape) {
  // main part of shape infer
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto input_imgs_data = op_desc->MutableInputDesc("imgs_data");
  auto output_imgs = op_desc->MutableOutputDesc("imgs");
  auto output_imgs_relation = op_desc->MutableOutputDesc("imgs_relation");
  auto output_imgs_lang = op_desc->MutableOutputDesc("imgs_lang");
  std::vector<std::pair<int64_t, int64_t>> imgs_data_range;
  input_imgs_data->GetShapeRange(imgs_data_range);
  auto input_imgs_dims = input_imgs_data->GetShape().GetDims();

  std::string data_format = "NHWC";
  (void)op.GetAttr("data_format", data_format);
  const bool known_shape = !IsUnknown(input_imgs_dims);
  if ((!imgs_data_range.empty()) || known_shape) {
    int64_t range_max = imgs_data_range[0].second;
    if (known_shape) {
      range_max = input_imgs_dims[0];
    }
    if (range_max == 0) {
      std::string reason = "max shape range of imgs_data must be != 0";
      REPORT_INNER_ERROR("E19999",
                         "[Node:%s] Check imgs_data shape range failed, as %s",
                         op.GetName().c_str(), reason.c_str());
      GE_OP_LOGE(op.GetName().c_str(),
                 "[InferShape] Check imgs_data shape range failed, as %s",
                 reason.c_str());
      return GRAPH_PARAM_INVALID;
    }

    int64_t batch_size = 1;
    (void)op.GetAttr("batch_size", batch_size);
    if (batch_size == 0) {
      std::string reason = "attr[batch_size] must be != 0";
      REPORT_INNER_ERROR("E19999", "[Node:%s] Check attr failed, as %s",
                         op.GetName().c_str(), reason.c_str());
      GE_OP_LOGE(op.GetName().c_str(), "[InferShape] Check attr failed, as %s",
                 reason.c_str());
      return GRAPH_PARAM_INVALID;
    }

    GE_OP_LOGD(
        op.GetName().c_str(),
        "[InferShape] Set output shape range, range_max[%ld], batch_size[%ld]",
        range_max, batch_size);
    int64_t max_pic_num = range_max / kChannelNum + 7 * (batch_size - 1);
    std::pair<int64_t, int64_t> imgs_n_range({1, max_pic_num});
    std::pair<int64_t, int64_t> imgs_h_range({kResizeH, kResizeH});
    std::pair<int64_t, int64_t> imgs_w_range({kResizeW, kResizeW});
    std::pair<int64_t, int64_t> imgs_c_range({kChannelNum, kChannelNum});
    std::vector<std::pair<int64_t, int64_t>> imgs_range_vec(4);
    imgs_range_vec[0] = imgs_n_range;
    if (data_format == "NHWC") {
      imgs_range_vec[1] = imgs_h_range;
      imgs_range_vec[2] = imgs_w_range;
      imgs_range_vec[3] = imgs_c_range;
    } else {
      imgs_range_vec[1] = imgs_c_range;
      imgs_range_vec[2] = imgs_h_range;
      imgs_range_vec[3] = imgs_w_range;
    }
    output_imgs->SetShapeRange(imgs_range_vec);

    std::pair<int64_t, int64_t> imgs_relation_range({1, max_pic_num});
    std::vector<std::pair<int64_t, int64_t>> imgs_relation_range_vec;
    imgs_relation_range_vec.push_back(imgs_relation_range);
    output_imgs_relation->SetShapeRange(imgs_relation_range_vec);

    std::pair<int64_t, int64_t> imgs_lang_range({1, max_pic_num / batch_size});
    std::vector<std::pair<int64_t, int64_t>> imgs_lang_range_vec;
    imgs_lang_range_vec.push_back(imgs_lang_range);
    output_imgs_lang->SetShapeRange(imgs_lang_range_vec);
  }

  if (data_format == "NHWC") {
    output_imgs->SetShape(
        ge::GeShape({UNKNOWN_DIM, kResizeH, kResizeW, kChannelNum}));
    output_imgs->SetOriginShape(
        ge::GeShape({UNKNOWN_DIM, kResizeH, kResizeW, kChannelNum}));
  } else {
    output_imgs->SetShape(
        ge::GeShape({UNKNOWN_DIM, kChannelNum, kResizeH, kResizeW}));
    output_imgs->SetOriginShape(
        ge::GeShape({UNKNOWN_DIM, kChannelNum, kResizeH, kResizeW}));
  }
  output_imgs->SetDataType(input_imgs_data->GetDataType());

  output_imgs_relation->SetDataType(DT_INT32);
  output_imgs_relation->SetShape(ge::GeShape(UNKNOWN_SHAPE));
  output_imgs_relation->SetOriginShape(ge::GeShape(UNKNOWN_SHAPE));

  output_imgs_lang->SetDataType(DT_INT32);
  output_imgs_lang->SetShape(ge::GeShape(UNKNOWN_SHAPE));
  output_imgs_lang->SetOriginShape(ge::GeShape(UNKNOWN_SHAPE));
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(OCRRecognitionPreHandle,
                      OCRRecognitionPreHandleInferShape);

bool SetConstImage(Operator &op, const std::vector<int64_t> &images_shape,
                   std::vector<int64_t> &size_out,
                   const std::string &input_format) {
  std::vector<int64_t> y_shape;
  TensorDesc td = op.GetOutputDescByName("resized_img");
  if (input_format == "NCHW") {
    y_shape.push_back(images_shape[0]);
    y_shape.push_back(960);
    y_shape.push_back(960);
  } else {
    y_shape.push_back(960);
    y_shape.push_back(960);
    y_shape.push_back(images_shape[2]);
  }
  td.SetShape(ge::Shape(y_shape));
  td.SetDataType(DT_UINT8);
  (void)op.UpdateOutputDesc("resized_img", td);
  return true;
}

IMPLEMT_COMMON_INFERFUNC(OCRDetectionPreHandleInferShape) {
  TensorDesc img_desc = op.GetInputDescByName("img");
  TensorDesc resized_img_desc = op.GetOutputDescByName("resized_img");
  TensorDesc h_scale_desc = op.GetOutputDescByName("h_scale");
  TensorDesc w_scale_desc = op.GetOutputDescByName("w_scale");
  std::vector<int64_t> scalar;
  h_scale_desc.SetShape(ge::Shape(scalar));
  h_scale_desc.SetDataType(DT_FLOAT);
  (void)op.UpdateOutputDesc("h_scale", h_scale_desc);
  w_scale_desc.SetShape(ge::Shape(scalar));
  w_scale_desc.SetDataType(DT_FLOAT);
  (void)op.UpdateOutputDesc("w_scale", w_scale_desc);

  std::vector<int64_t> image_shape = img_desc.GetShape().GetDims();
  std::vector<int64_t> size_out;
  std::string data_format;
  (void)op.GetAttr("data_format", data_format);

  if (img_desc.GetShape().GetShapeSize() == UNKNOWN_DIM) {
    std::vector<int64_t> y_shape;
    if (data_format == "NCHW") {
      y_shape.push_back(3);
      y_shape.push_back(960);
      y_shape.push_back(960);
    } else {
      y_shape.push_back(960);
      y_shape.push_back(960);
      y_shape.push_back(3);
    }

    resized_img_desc.SetShape(ge::Shape(y_shape));
    resized_img_desc.SetDataType(DT_UINT8);
    (void)op.UpdateOutputDesc("resized_img", resized_img_desc);
    return GRAPH_SUCCESS;
  }

  if (SetConstImage(op, image_shape, size_out, data_format)) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(OCRDetectionPreHandle, OCRDetectionPreHandleInferShape);

IMPLEMT_COMMON_INFERFUNC(OCRIdentifyPreHandleInferShape) {
  TensorDesc imgs_data_desc = op.GetInputDesc("imgs_data");
  TensorDesc imgs_offset_desc = op.GetInputDesc("imgs_offset");
  TensorDesc imgs_size_desc = op.GetInputDesc("imgs_size");
  TensorDesc resized_imgs_desc = op.GetOutputDescByName("resized_imgs");
  std::vector<int64_t> list_out_size;
  (void)op.GetAttr("size", list_out_size);
  if (list_out_size.size() != 2) {
    return GRAPH_FAILED;
  }
  std::string data_format = "NHWC";
  (void)op.GetAttr("data_format", data_format);
  std::vector<std::pair<int64_t, int64_t>> imgs_data_range;
  std::vector<std::pair<int64_t, int64_t>> imgs_offset_range;
  std::vector<std::pair<int64_t, int64_t>> imgs_size_range;
  std::vector<std::pair<int64_t, int64_t>> out_range;
  if (imgs_data_desc.GetShape().GetShapeSize() == UNKNOWN_DIM &&
      imgs_offset_desc.GetShape().GetShapeSize() == UNKNOWN_DIM &&
      imgs_size_desc.GetShape().GetShapeSize() == UNKNOWN_DIM) {
    imgs_size_desc.GetShapeRange(imgs_size_range);
    if (imgs_size_range.size() != 2) {
      GE_OP_LOGE(op.GetName().c_str(), "Img size shape range size must be 2");
      return GRAPH_FAILED;
    }
    if (data_format == "NCHW") {
      resized_imgs_desc.SetShape(ge::Shape(
          {UNKNOWN_DIM, kChannelNum, list_out_size[0], list_out_size[1]}));
      resized_imgs_desc.SetDataType(DT_UINT8);
      out_range.push_back(imgs_size_range[0]);
      out_range.push_back(
          std::pair<int64_t, int64_t>{kChannelNum, kChannelNum});
      out_range.push_back(
          std::pair<int64_t, int64_t>{list_out_size[0], list_out_size[0]});
      out_range.push_back(
          std::pair<int64_t, int64_t>{list_out_size[1], list_out_size[1]});
    } else {
      resized_imgs_desc.SetShape(ge::Shape(
          {UNKNOWN_DIM, list_out_size[0], list_out_size[1], kChannelNum}));
      resized_imgs_desc.SetDataType(DT_UINT8);
      out_range.push_back(std::pair<int64_t, int64_t>{1, 128});
      out_range.push_back(
          std::pair<int64_t, int64_t>{list_out_size[0], list_out_size[0]});
      out_range.push_back(
          std::pair<int64_t, int64_t>{list_out_size[1], list_out_size[1]});
      out_range.push_back(
          std::pair<int64_t, int64_t>{kChannelNum, kChannelNum});
    }
    resized_imgs_desc.SetShapeRange(out_range);
    (void)op.UpdateOutputDesc("resized_imgs", resized_imgs_desc);
    return GRAPH_SUCCESS;
  } else if (imgs_data_desc.GetShape().GetShapeSize() == UNKNOWN_DIM ||
             imgs_offset_desc.GetShape().GetShapeSize() == UNKNOWN_DIM ||
             imgs_size_desc.GetShape().GetShapeSize() == UNKNOWN_DIM) {
    return GRAPH_FAILED;
  } else {
    auto offset_dims = imgs_offset_desc.GetShape().GetDims();
    auto size_dims = imgs_size_desc.GetShape().GetDims();
    if (size_dims.size() != 2) {
      GE_OP_LOGE(op.GetName().c_str(), "Img size shape size must be 2");
      return GRAPH_FAILED;
    }
    std::vector<int64_t> out_shape;
    int32_t h = list_out_size[0];
    int32_t w = list_out_size[1];
    int32_t batch_size = offset_dims[0];
    if (data_format == "NCHW") {
      out_shape = {batch_size, kChannelNum, h, w};
    } else {
      out_shape = {batch_size, h, w, kChannelNum};
    }
    resized_imgs_desc.SetShape(ge::Shape(out_shape));
    resized_imgs_desc.SetOriginShape(ge::Shape(out_shape));
    resized_imgs_desc.SetDataType(DT_UINT8);
    (void)op.UpdateOutputDesc("resized_imgs", resized_imgs_desc);
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(OCRIdentifyPreHandle, OCRIdentifyPreHandleInferShape);
}  // namespace ge
