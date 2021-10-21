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


 //batch_dilate_polys
IMPLEMT_COMMON_INFERFUNC(BatchDilatePolysInferShape) {
  // main part of shape infer
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  std::vector<int64_t> dims={1,ge::UNKNOWN_DIM};
  auto score_shape=op_desc->MutableInputDesc("score")->GetShape();
  auto polys_data_shape=op_desc->MutableInputDesc("polys_data")->GetShape();

  auto dilated_polys_data_index=op_desc->MutableOutputDesc("dilated_polys_data");
  auto polys_data_index=op_desc->MutableInputDesc("polys_data");
  auto score_index=op_desc->MutableInputDesc("score");

  std::vector<std::pair<int64_t,int64_t>> dilated_polys_data_range;
  if(IsUnknown(score_shape.GetDims())){
    std::vector<std::pair<int64_t,int64_t>> score_range;
    score_index->GetShapeRange(score_range);
    int64_t polys_score_max_h=score_range[0].second;
    int64_t polys_score_max_w=score_range[1].second;
    std::pair<int64_t,int64_t> data_range({1,polys_score_max_h*polys_score_max_w});
    dilated_polys_data_range.push_back(data_range);
    OP_LOGI("dilated_polys_data_range=%d",polys_score_max_h*polys_score_max_w);
    dilated_polys_data_index->SetShapeRange(dilated_polys_data_range);
  } else{
    auto polys_data_max_dims=score_index->GetShape().GetDims();
    int64_t polys_score_max_h=polys_data_max_dims[0];
    int64_t polys_score_max_w=polys_data_max_dims[1];
    std::pair<int64_t,int64_t> data_range({1,polys_score_max_h*polys_score_max_w});
    dilated_polys_data_range.push_back(data_range);
    dilated_polys_data_index->SetShapeRange(dilated_polys_data_range);
  }
  dilated_polys_data_index->SetDataType(DT_INT32);
  dilated_polys_data_index->SetShape(ge::GeShape(UNKNOWN_SHAPE));
  dilated_polys_data_index->SetOriginShape(ge::GeShape(UNKNOWN_SHAPE));

  auto polys_offset_shape=op_desc->MutableInputDesc("polys_offset")->GetShape();
  auto dilated_polys_offset_index=op_desc->MutableOutputDesc("dilated_polys_offset");
  auto polys_offset_index=op_desc->MutableInputDesc("polys_offset");
  std::vector<std::pair<int64_t,int64_t>> dilated_polys_offset_range;
    if(polys_offset_shape.GetDims()==UNKNOWN_RANK||polys_offset_shape.GetDims()==UNKNOWN_SHAPE){
    std::vector<std::pair<int64_t,int64_t>> polys_offset_range;
    polys_offset_index->GetShapeRange(polys_offset_range);
    int64_t polys_offset_max=polys_offset_range[0].second;
    std::pair<int64_t,int64_t> offset_range({1,polys_offset_max});
    dilated_polys_offset_range.push_back(offset_range);
    dilated_polys_offset_index->SetShapeRange(dilated_polys_data_range);
  } else{
    auto polys_offset_max_dims=dilated_polys_offset_index->GetShape().GetDims();
    int64_t polys_offset_max=polys_offset_max_dims[0];
    std::pair<int64_t,int64_t> offset_range({1,polys_offset_max});
    dilated_polys_offset_range.push_back(offset_range);
    dilated_polys_offset_index->SetShapeRange(dilated_polys_data_range);
  }
  dilated_polys_offset_index->SetDataType(DT_INT32);
  dilated_polys_offset_index->SetShape(ge::GeShape(UNKNOWN_SHAPE));
  dilated_polys_offset_index->SetOriginShape(ge::GeShape(UNKNOWN_SHAPE));

  auto polys_size_shape=op_desc->MutableInputDesc("polys_size")->GetShape();
  auto dilated_polys_size_index=op_desc->MutableOutputDesc("dilated_polys_size");
  auto polys_size_index=op_desc->MutableInputDesc("polys_size");
  std::vector<std::pair<int64_t,int64_t>> dilated_polys_size_range;
  if(polys_size_shape.GetDims()==UNKNOWN_RANK||polys_size_shape.GetDims()==UNKNOWN_SHAPE){
    std::vector<std::pair<int64_t,int64_t>> polys_size_range;
    polys_size_index->GetShapeRange(polys_size_range);
    int64_t polys_size_max=polys_size_range[0].second;
    std::pair<int64_t,int64_t> size_range({1,polys_size_max});
    dilated_polys_size_range.push_back(size_range);
    dilated_polys_size_index->SetShapeRange(dilated_polys_data_range);
  } else{
    auto polys_size_max_dims=dilated_polys_size_index->GetShape().GetDims();
    int64_t polys_size_max=polys_size_max_dims[0];
    std::pair<int64_t,int64_t> size_range({1,polys_size_max});
    dilated_polys_size_range.push_back(size_range);
    dilated_polys_size_index->SetShapeRange(dilated_polys_data_range);
  }
  dilated_polys_size_index->SetDataType(DT_INT32);
  dilated_polys_size_index->SetShape(ge::GeShape(UNKNOWN_SHAPE));
  dilated_polys_size_index->SetOriginShape(ge::GeShape(UNKNOWN_SHAPE));

  return GRAPH_SUCCESS;
  }

  COMMON_INFER_FUNC_REG(BatchDilatePolys,
                      BatchDilatePolysInferShape);
  

  //  OCRFindContours
IMPLEMT_COMMON_INFERFUNC(OCRFindContoursInfer) {
  // main part of shape infer
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  std::vector<int64_t> dims={1,ge::UNKNOWN_DIM};
  auto img_shape=op_desc->MutableInputDesc("img")->GetShape();
  
  auto polys_data_index=op_desc->MutableOutputDesc("polys_data");
  auto polys_offset_index=op_desc->MutableOutputDesc("polys_offset");
  auto polys_size_index=op_desc->MutableOutputDesc("polys_size");
  auto img_index=op_desc->MutableInputDesc("img");

  std::vector<std::pair<int64_t,int64_t>> polys_data_range;
  if(IsUnknown(img_shape.GetDims())){
    std::vector<std::pair<int64_t,int64_t>> img_range;
    img_index->GetShapeRange(img_range);
    int64_t img_max_h=img_range[0].second;
    int64_t img_max_w=img_range[1].second;
    std::pair<int64_t,int64_t> data_range({1,img_max_h*img_max_w});
    polys_data_range.push_back(data_range);
    polys_data_index->SetShapeRange(polys_data_range);
    polys_offset_index->SetShapeRange(polys_data_range);
    polys_size_index->SetShapeRange(polys_data_range);
  } else{
    auto img_max_dims=img_index->GetShape().GetDims();
    int64_t img_max_h=img_max_dims[0];
    int64_t img_max_w=img_max_dims[1];
    std::pair<int64_t,int64_t> data_range({1,img_max_h*img_max_w});
    polys_data_range.push_back(data_range);
    polys_data_index->SetShapeRange(polys_data_range);
    polys_offset_index->SetShapeRange(polys_data_range);
    polys_size_index->SetShapeRange(polys_data_range);
  }
  polys_data_index->SetDataType(DT_INT32);
  polys_data_index->SetShape(ge::GeShape(UNKNOWN_SHAPE));
  polys_data_index->SetOriginShape(ge::GeShape(UNKNOWN_SHAPE));

  polys_offset_index->SetDataType(DT_INT32);
  polys_offset_index->SetShape(ge::GeShape(UNKNOWN_SHAPE));
  polys_offset_index->SetOriginShape(ge::GeShape(UNKNOWN_SHAPE));

  polys_size_index->SetDataType(DT_INT32);
  polys_size_index->SetShape(ge::GeShape(UNKNOWN_SHAPE));
  polys_size_index->SetOriginShape(ge::GeShape(UNKNOWN_SHAPE));
 
  return GRAPH_SUCCESS;
  }

  COMMON_INFER_FUNC_REG(OCRFindContours,
                      OCRFindContoursInfer);

  IMPLEMT_COMMON_INFERFUNC(DequeueInferShape){
    TensorDesc data_desc = op.GetOutputDescByName("data");
    std::vector<int64_t> output_shape;
    (void)op.GetAttr("output_shape",output_shape);
    DataType dtype;
    (void)op.GetAttr("output_type",dtype);
    data_desc.SetShape(Shape(output_shape));
    data_desc.SetDataType(dtype);
    op.UpdateOutputDesc("data",data_desc);
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Dequeue, DequeueInferShape);

IMPLEMT_COMMON_INFERFUNC(OCRDetectionPostHandleInfer){
    auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
    auto input_img = op_desc->MutableInputDesc("img");
    auto input_polys_offset = op_desc->MutableInputDesc("polys_offset");
    auto output_imgs_data = op_desc->MutableOutputDesc("imgs_data");
    auto output_imgs_offset = op_desc->MutableOutputDesc("imgs_offset");
    auto output_imgs_size = op_desc->MutableOutputDesc("imgs_size");
    auto output_rect_points = op_desc->MutableOutputDesc("rect_points");
    std::vector<std::pair<int64_t, int64_t>> img_range;
    std::vector<std::pair<int64_t, int64_t>> polys_offset_range;
    input_img->GetShapeRange(img_range);
    input_polys_offset->GetShapeRange(polys_offset_range);
    auto input_img_dims = input_img->GetShape().GetDims();
    auto input_polys_offset_dims = input_polys_offset->GetShape().GetDims();
    const bool data_known_shape = !IsUnknown(input_img_dims);
    if ((!img_range.empty()) || data_known_shape) {
        int64_t data_range_max = img_range[0].second *
                                 img_range[1].second *
                                 img_range[2].second;
        if (data_known_shape) {
            data_range_max = input_img_dims[0] * input_img_dims[1] * input_img_dims[2]; 
        }
        if (data_range_max == 0) {
            std::string reason = "max shape range of img must be != 0";
            REPORT_INNER_ERROR("E19999",
                               "[Node:%s] Check img shape range failed, as %s",
                               op.GetName().c_str(), reason.c_str());
            GE_OP_LOGE(op.GetName().c_str(),
                       "[InferShape] Check img shape range failed, as %s",
                       reason.c_str());
            return GRAPH_PARAM_INVALID;
        }

        std::pair<int64_t, int64_t> imgs_data_range({1, data_range_max});
        std::vector<std::pair<int64_t, int64_t>> imgs_data_range_vec;
        imgs_data_range_vec.push_back(imgs_data_range);
        output_imgs_data->SetShapeRange(imgs_data_range_vec);
    }

    const bool size_known_shape = !IsUnknown(input_polys_offset_dims);
    if ((!polys_offset_range.empty()) || size_known_shape) {
        int64_t size_range_max = polys_offset_range[0].second;
        if (size_known_shape) {
            size_range_max = input_polys_offset_dims[0]; 
        }
        if (size_range_max == 0) {
            std::string reason = "max shape range of polys_offset must be != 0";
            REPORT_INNER_ERROR("E19999",
                               "[Node:%s] Check polys_offset shape range failed, as %s",
                               op.GetName().c_str(), reason.c_str());
            GE_OP_LOGE(op.GetName().c_str(),
                       "[InferShape] Check polys_offset shape range failed, as %s",
                       reason.c_str());
            return GRAPH_PARAM_INVALID;
        }

        std::pair<int64_t, int64_t> imgs_offset_range({size_range_max, size_range_max});
        std::vector<std::pair<int64_t, int64_t>> imgs_offset_range_vec;
        imgs_offset_range_vec.push_back(imgs_offset_range);
        output_imgs_offset->SetShapeRange(imgs_offset_range_vec);
        output_imgs_offset->SetShape(ge::GeShape(UNKNOWN_SHAPE));
        output_imgs_offset->SetOriginShape(ge::GeShape(UNKNOWN_SHAPE));

        std::pair<int64_t, int64_t> imgs_size_range({size_range_max, size_range_max});
        const int64_t dims = 3;
        std::pair<int64_t, int64_t> imgs_size_range_dims({dims, dims});
        std::vector<std::pair<int64_t, int64_t>> imgs_size_range_vec;
        imgs_size_range_vec.push_back(imgs_size_range);
        imgs_size_range_vec.push_back(imgs_size_range_dims);
        output_imgs_size->SetShapeRange(imgs_size_range_vec);
        output_imgs_size->SetShape(ge::GeShape({UNKNOWN_DIM, dims}));
        output_imgs_size->SetOriginShape(ge::GeShape({UNKNOWN_DIM, dims}));

        std::pair<int64_t, int64_t> rect_points_range({size_range_max, size_range_max});
        std::pair<int64_t, int64_t> rect_points_range_points_num({4, 4});
        std::pair<int64_t, int64_t> rect_points_range_coor_num({2, 2});
        std::vector<std::pair<int64_t, int64_t>> rect_points_range_vec;
        rect_points_range_vec.push_back(rect_points_range);
        rect_points_range_vec.push_back(rect_points_range_points_num);
        rect_points_range_vec.push_back(rect_points_range_coor_num);
        output_rect_points->SetShapeRange(rect_points_range_vec);
        output_rect_points->SetShape(ge::GeShape({UNKNOWN_DIM, 4, 2}));
        output_rect_points->SetOriginShape(ge::GeShape({UNKNOWN_DIM, 4, 2}));
    }

    output_imgs_data->SetDataType(DT_UINT8);
    output_imgs_data->SetShape(ge::GeShape(UNKNOWN_SHAPE));
    output_imgs_data->SetOriginShape(ge::GeShape(UNKNOWN_SHAPE));

    output_imgs_offset->SetDataType(DT_INT32);

    output_imgs_size->SetDataType(DT_INT32);

    output_rect_points->SetDataType(DT_INT32);
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(OCRDetectionPostHandle, OCRDetectionPostHandleInfer);

IMPLEMT_COMMON_INFERFUNC(ResizeAndClipPolysInfer){
    auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
    auto input_polys_data = op_desc->MutableInputDesc("polys_data");
    auto input_polys_offset = op_desc->MutableInputDesc("polys_offset");
    auto output_clipped_polys_data = op_desc->MutableOutputDesc("clipped_polys_data");
    auto output_clipped_polys_offset = op_desc->MutableOutputDesc("clipped_polys_offset");
    auto output_clipped_polys_size = op_desc->MutableOutputDesc("clipped_polys_size");
    std::vector<std::pair<int64_t, int64_t>> polys_data_range;
    std::vector<std::pair<int64_t, int64_t>> polys_offset_range;
    input_polys_data->GetShapeRange(polys_data_range);
    input_polys_offset->GetShapeRange(polys_offset_range);
    auto input_polys_data_dims = input_polys_data->GetShape().GetDims();
    auto input_polys_offset_dims = input_polys_offset->GetShape().GetDims();
    const bool data_known_shape = !IsUnknown(input_polys_data_dims);
    if ((!polys_data_range.empty()) || data_known_shape) {
        int64_t data_range_max = polys_data_range[0].second;
        if (data_known_shape) {
            data_range_max = input_polys_data_dims[0]; 
        }
        if (data_range_max == 0) {
            std::string reason = "max shape range of polys_data must be != 0";
            REPORT_INNER_ERROR("E19999",
                               "[Node:%s] Check polys_data shape range failed, as %s",
                               op.GetName().c_str(), reason.c_str());
            GE_OP_LOGE(op.GetName().c_str(),
                       "[InferShape] Check polys_data shape range failed, as %s",
                       reason.c_str());
            return GRAPH_PARAM_INVALID;
        }

        std::pair<int64_t, int64_t> clipped_polys_data_range({1, data_range_max});
        std::vector<std::pair<int64_t, int64_t>> clipped_polys_data_range_vec;
        clipped_polys_data_range_vec.push_back(clipped_polys_data_range);
        output_clipped_polys_data->SetShapeRange(clipped_polys_data_range_vec);
    }

    const bool size_known_shape = !IsUnknown(input_polys_offset_dims);
    if ((!polys_offset_range.empty()) || size_known_shape) {
        int64_t size_range_max = polys_offset_range[0].second;
        if (size_known_shape) {
            size_range_max = input_polys_offset_dims[0]; 
        }
        if (size_range_max == 0) {
            std::string reason = "max shape range of polys_offset must be != 0";
            REPORT_INNER_ERROR("E19999",
                               "[Node:%s] Check polys_offset shape range failed, as %s",
                               op.GetName().c_str(), reason.c_str());
            GE_OP_LOGE(op.GetName().c_str(),
                       "[InferShape] Check polys_offset shape range failed, as %s",
                       reason.c_str());
            return GRAPH_PARAM_INVALID;
        }

        std::pair<int64_t, int64_t> clipped_polys_offset_range({1, size_range_max});
        std::vector<std::pair<int64_t, int64_t>> clipped_polys_offset_range_vec;
        clipped_polys_offset_range_vec.push_back(clipped_polys_offset_range);
        output_clipped_polys_offset->SetShapeRange(clipped_polys_offset_range_vec);

        std::pair<int64_t, int64_t> clipped_polys_size_range({1, size_range_max});
        std::vector<std::pair<int64_t, int64_t>> clipped_polys_size_range_vec;
        clipped_polys_size_range_vec.push_back(clipped_polys_size_range);
        output_clipped_polys_size->SetShapeRange(clipped_polys_size_range_vec);
    }

    output_clipped_polys_data->SetDataType(DT_INT32);
    output_clipped_polys_data->SetShape(ge::GeShape(UNKNOWN_SHAPE));
    output_clipped_polys_data->SetOriginShape(ge::GeShape(UNKNOWN_SHAPE));

    output_clipped_polys_offset->SetDataType(DT_INT32);
    output_clipped_polys_offset->SetShape(ge::GeShape(UNKNOWN_SHAPE));
    output_clipped_polys_offset->SetOriginShape(ge::GeShape(UNKNOWN_SHAPE));

    output_clipped_polys_size->SetDataType(DT_INT32);
    output_clipped_polys_size->SetShape(ge::GeShape(UNKNOWN_SHAPE));
    output_clipped_polys_size->SetOriginShape(ge::GeShape(UNKNOWN_SHAPE));
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ResizeAndClipPolys, ResizeAndClipPolysInfer);


}  // namespace ge
