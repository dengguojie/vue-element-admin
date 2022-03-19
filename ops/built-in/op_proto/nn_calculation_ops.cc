/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
 * \file nn_calculation_ops.cpp
 * \brief
 */
#define CHECK_OP_FUNC(cond, post_action_expr, msg, ...)                                                          \
  {                                                                                                              \
    if (cond) {                                                                                                  \
      AscendString local_op_name;                                                                                \
      CHECK(op.GetName(local_op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), post_action_expr); \
      CUBE_INNER_ERR_REPORT(local_op_name.GetString(), msg, ##__VA_ARGS__);                                      \
      post_action_expr;                                                                                          \
    }                                                                                                            \
  }

#define CHECK_SIZE(cond, post_action_expr, msg)                           \
  {                                                     \
    if (cond) {                                         \
      AscendString local_op_name; \
      CHECK(op.GetName(local_op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), post_action_expr); \
      ErrorManager::GetInstance().ATCReportErrMessage("E50058", {"op_name", "description"}, \
                                                      {local_op_name.GetString(), msg}); \
      post_action_expr;                              \
    }                                                   \
  }

#define CHECK_INFO(cond, post_action_expr, msg, ...)                                                             \
  {                                                                                                              \
    if (cond) {                                                                                                  \
      AscendString local_op_name;                                                                                \
      CHECK(op.GetName(local_op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), post_action_expr); \
      OP_LOGE(local_op_name.GetString(), msg, ##__VA_ARGS__);                                                    \
      post_action_expr;                                                                                          \
    }                                                                                                            \
  }

#include "./nn_calculation_ops.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <tuple>

#include "./util/error_util.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "util/util.h"
#include "register/infer_data_slice_registry.h"
#include "graph/common_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "axis_util.h"
#include "util/vector_proto_profiling.h"
#include "util/op_common_util.h"

namespace ge {
namespace {
  // Conv2d
  const int32_t kConv2dDimSizeLimit = 4;
  const size_t kConv2dPadSizeLimit = 4;
  const size_t kConv2dStridesSizeLimit = 4;
  const size_t kConv2dDilationSizeLimit = 4;
  const size_t kConv2dInputSizeLimit = 4;
  const size_t kConv2dPadUpIdx = 0;
  const size_t kConv2dPadDownIdx = 1;
  const size_t kConv2dPadLeftIdx = 2;
  const size_t kConv2dPadRightIdx = 3;
  // Conv3d
  const int32_t kConv3dDimSizeLimit = 5;
  const int32_t kConv3dStridesSizeLimit = 5;
  const int32_t kConv3dInputSizeLimit = 5;
  const int32_t kConv3dPadsSizeLimit = 6;
  const size_t kConv3dPadHeadIdx = 0;
  const size_t kConv3dPadTailIdx = 1;
  const size_t kConv3dPadUpIdx = 2;
  const size_t kConv3dPadDownIdx = 3;
  const size_t kConv3dPadLeftIdx = 4;
  const size_t kConv3dPadRightIdx = 5;
  // NDHWC
  const size_t kNDimIdx = 0;
  const size_t kDDimNDHWCIdx = 1;
  const size_t kHDimNDHWCIdx = 2;
  const size_t kWDimNDHWCIdx = 3;
  const size_t kCDimNDHWCIdx = 4;
  // NCDHW
  const size_t kDDimNCDHWIdx = 2;
  const size_t kHDimNCDHWIdx = 3;
  const size_t kWDimNCDHWIdx = 4;
  const size_t kCDimNCDHWIdx = 1;
  // NHWC
  const size_t kHDimNHWCIdx = 1;
  const size_t kWDimNHWCIdx = 2;
  const size_t kCDimNHWCIdx = 3;
  // NCHW
  const size_t kHDimNCHWIdx = 2;
  const size_t kWDimNCHWIdx = 3;
  const size_t kCDimNCHWIdx = 1;
  const size_t kNDimNCHWIdx = 0;
  // NC1HWC0
  const size_t kC1DimNC1HWC0Idx = 1;
  const size_t kHDimNC1HWC0Idx = 2;
  const size_t kWDimNC1HWC0Idx = 3;
  const size_t kC0DimNC1HWC0Idx = 4;
  // NDC1HWC0
  const size_t kDDimNDC1HWC0Idx = 1;
  const size_t kC1DimNDC1HWC0Idx = 2;
  const size_t kHDimNDC1HWC0Idx = 3;
  const size_t kWDimNDC1HWC0Idx = 4;
  // data slice
  const int32_t kConv3dDataSlice = 6;
  const int32_t kDataSliceLen = 2;
  const int32_t kBlockBaseSize = 16;
  const int32_t kFilterDataSlice = 4;

  const int32_t kDeformDimSizeLimit = 4;
  const int32_t kDeformKsizeLimit = 2;
  const int64_t kDynamicRangeLowerBound = 1;
  const int64_t kDynamicRangeUpperBound = 4096;
  const int32_t MAX_RANGE = std::numeric_limits<int32_t>::max();
  const std::vector<int64_t> DYNAMIC_DIM_ALL = {-2};
  using SliceTuple = std::tuple<int32_t, int32_t, int32_t, int32_t>;
  using OutputTuple = std::tuple<GeTensorDescPtr, bool>;
}

struct AttrDataInfo {
  int32_t stride;
  int32_t pads;
  int64_t kernel;
};
struct AttrInfo {
  AttrDataInfo h_attr;
  AttrDataInfo w_attr;
};
struct PosInfo {
  int32_t n_position;
  int32_t h_position;
  int32_t w_position;
  int32_t c_position;
};
struct ShapeValInfo {
  int64_t n_value;
  int64_t h_value;
  int64_t w_value;
  int64_t c_value;
};
struct OpDetailInfo {
  AscendString op_name;
  OpDescPtr op_desc;
  AscendString op_type;
};
// ----------------LSTM begin-------------------

IMPLEMT_VERIFIER(LSTM, LSTMVerify) {
  return GRAPH_SUCCESS;
}
IMPLEMT_VERIFIER(LSTM, LSTMInferShape) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  int32_t output_size = 0;
  int32_t input_size = op.GetInputsSize();
  bool expose_hidden = false;

  if (ge::GRAPH_SUCCESS != op.GetAttr("expose_hidden", expose_hidden)) {
    OP_LOGE(op_name.GetString(), "GetOpAttr expose_hidden failed!");
  }

  if (input_size == 9) {
    output_size = op.GetInputDesc(3).GetShape().GetDim(2);
  } else if (input_size == 7 && expose_hidden) {
    output_size = op.GetInputDesc(2).GetShape().GetDim(2);
  } else if (input_size == 7) {
    output_size = op.GetInputDesc(6).GetShape().GetDim(1);
  } else {
    output_size = op.GetInputDesc(4).GetShape().GetDim(1);
  }

  ge::TensorDesc inputXTensorDesc = op.GetInputDesc(0);

  vector<int64_t> hDims;

  hDims.push_back(inputXTensorDesc.GetShape().GetDim(0));
  hDims.push_back(inputXTensorDesc.GetShape().GetDim(1));
  hDims.push_back(output_size);

  TensorDesc outputHTensorDesc = op.GetOutputDesc(0);
  outputHTensorDesc.SetShape(ge::Shape(hDims));
  (void)op.UpdateOutputDesc("h", outputHTensorDesc);

  if (expose_hidden) {
    int32_t c_index = 0;
    int32_t h_index = 0;
    if (input_size == 9) {
      h_index = 3;
      c_index = 4;
    } else {
      h_index = 2;
      c_index = 3;
    }
    ge::TensorDesc inputHTensorDesc = op.GetInputDesc(h_index);
    ge::TensorDesc inputCTensorDesc = op.GetInputDesc(c_index);
    ge::Shape shape = inputCTensorDesc.GetShape();
    ge::Shape shapeH = inputHTensorDesc.GetShape();

    TensorDesc outputHtTensorDesc = op.GetOutputDesc(1);
    TensorDesc outputCtTensorDesc = op.GetOutputDesc(2);

    outputCtTensorDesc.SetShape(shape);
    outputHtTensorDesc.SetShape(shapeH);

    (void)op.UpdateOutputDesc("h_t", outputHtTensorDesc);
    (void)op.UpdateOutputDesc("c_t", outputCtTensorDesc);
  }

  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(LSTM, LSTMInferShape);
VERIFY_FUNC_REG(LSTM, LSTMVerify);
// ----------------LSTM end------------------

// cal new_pad_value for conv/depthwise/conv_compress(data slice info)
static void InferHWConv2D(int32_t input, int32_t kernel, int32_t pad, int32_t stride,
                          int32_t dilation, vector<int64_t> output_slice, vector<int64_t>& data_slice,
                          int64_t* start_add_pad, int64_t* end_add_pad) {
  OP_LOGD("InferHWConv2D:", "input:%d kernel:%d pad:%d stride:%d dilation:%d start_add_pad:%ld end_add_pad:%ld\n",
          input, kernel, pad, stride, dilation, *start_add_pad, *end_add_pad);
  if (output_slice.size() >= 2) { // output slice has 2 dim(h_start, h_end)
    // calc start rule: (i_start + pad_h)/stride_h = output_start
    int64_t i_start = output_slice[0] * stride - pad;
    if (i_start < 0) {
      *start_add_pad = -i_start;
      i_start = 0;
    } else {
      *start_add_pad = 0;
    }
    // calc end rule: (iend_start + pad_h)/stride_h = output_end
    // iend_end = iend_start + dilation*(kernel_h-1)
    int64_t i_end = output_slice[1] * stride - pad + dilation * (kernel - 1);
    if (i_end >= input) {
      *end_add_pad = i_end - input + 1;
      i_end = input - 1;
    } else {
      *end_add_pad = 0;
    }
    data_slice = {i_start, i_end};
    OP_LOGD("InferHWConv2D:", "i_start:%ld i_end:%ld output_slice[0]:%ld output_slice[1]:%ld\n",
            i_start, i_end, output_slice[0], output_slice[1]);
  }
}
// ----------------DepthwiseConv2d Op-------------------
/*!
  * @brief provide DepthwiseConv2D operator slice data
  * @param DepthwiseConv2D Operator type.
  * @param DepthwiseConv2D slice data function
  * @return Status The processing flow result.
  */
IMPLEMT_INFER_DATA_SLICE(DepthwiseConv2D, DepthwiseConv2DInferDataSlice) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter DepthwiseConv2D InferDataSlice");
  // get input h/w, filter h/w, pad_h,pad_w, stride_h, stride_w, dilation_h,dilation_w
  auto x_tensor = op.GetInputDesc(0);
  auto w_tensor = op.GetInputDesc(1);

  auto x_shape = x_tensor.GetOriginShape().GetDims();
  auto w_shape = w_tensor.GetOriginShape().GetDims();

  auto x_format = x_tensor.GetOriginFormat();
  auto w_format = w_tensor.GetOriginFormat();

  CHECK(IsUnknownRankShape(x_shape),
        CUBE_INNER_ERR_REPORT(op_name.GetString(), "input x shape [-2] do not support split."),
        return GRAPH_FAILED);

  std::vector<int32_t> stride_list;
  std::vector<int32_t> dilation_list;
  std::vector<int32_t> pad_list;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stride_list) || GRAPH_SUCCESS != op.GetAttr("dilations", dilation_list)
      || GRAPH_SUCCESS != op.GetAttr("pads", pad_list)) {
    return GRAPH_FAILED;
  }
  CHECK(pad_list.size() < 4, CUBE_INNER_ERR_REPORT(op_name.GetString(), "pads size less then 4."),
    return GRAPH_FAILED);

  int32_t ih = 0;
  int32_t iw = 0;
  int32_t kh = 0;
  int32_t kw = 0;
  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;
  int32_t padt = pad_list[0];
  int32_t padl = pad_list[2];

  if (x_format == FORMAT_NCHW) {
    ih = x_shape[2];
    iw = x_shape[3];
    strh = stride_list[2];
    strw = stride_list[3];
    dilh = dilation_list[2];
    dilw = dilation_list[3];
  } else if (x_format == FORMAT_NHWC) {
    ih = x_shape[1];
    iw = x_shape[2];
    strh = stride_list[1];
    strw = stride_list[2];
    dilh = dilation_list[1];
    dilw = dilation_list[2];
  }

  if (w_format == FORMAT_NCHW) {
    kh = w_shape[2];
    kw = w_shape[3];
  } else if (w_format == FORMAT_NHWC) {
    kh = w_shape[1];
    kw = w_shape[2];
  } else if (w_format == FORMAT_HWCN) {
    kh = w_shape[0];
    kw = w_shape[1];
  }

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
  vector<vector<int64_t>> y_data_slice;
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}};
  CHECK_INFO(!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice), return GRAPH_FAILED,
              "no data slice, not need infer input");
  bool have_slice = false;
  vector<int> new_pad_lists = pad_list;
  for (size_t i = 0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() > 0) {
      have_slice = true;
      if (i == 2) {
        CHECK(ih == -1,
              CUBE_INNER_ERR_REPORT(op_name.GetString(), "input x dynamic h do not support split."),
              return GRAPH_FAILED);
        vector<int64_t> ih_slice;
        int64_t top_add_pad = pad_list[0];
        int64_t bom_add_pad = pad_list[1];
        InferHWConv2D(ih, kh, padt, strh, dilh, y_data_slice[i], ih_slice, &top_add_pad, &bom_add_pad);
        OP_LOGD(op_name.GetString(),
                "DepthwiseConv2D h axis slice ori_scope is [%d,%d], calced output scope is [%d,%d]",
                ih_slice[0], ih_slice[1], y_data_slice[i][0], y_data_slice[i][1]);
        new_pad_lists[0] = top_add_pad;
        new_pad_lists[1] = bom_add_pad;
        x_data_slice[i] = ih_slice;
      } else if (i == 3) {
        CHECK(iw == -1,
              CUBE_INNER_ERR_REPORT(op_name.GetString(), "input x dynamic w do not support split."),
              return GRAPH_FAILED);
        vector<int64_t> iw_slice;
        int64_t left_add_pad = pad_list[2];
        int64_t right_add_pad = pad_list[3];
        InferHWConv2D(iw, kw, padl, strw, dilw, y_data_slice[i], iw_slice, &left_add_pad, &right_add_pad);
        OP_LOGD(op_name.GetString(),
                "DepthwiseConv2D w axis slice ori_scope is [%d,%d], calced output scope is [%d,%d]",
                iw_slice[0], iw_slice[1], y_data_slice[i][0], y_data_slice[i][1]);
        new_pad_lists[2] = left_add_pad;
        new_pad_lists[3] = right_add_pad;
        x_data_slice[i] = iw_slice;
      } else {
        bool is_dyn = (i == 0) && (x_shape[0] == -1);
        vector<int64_t> dyn_slice = {-1, -1};
        x_data_slice[i] = is_dyn ? dyn_slice : y_data_slice[i];
      }
    }
  }
  op.SetAttr("pads", new_pad_lists);
  OP_LOGD(op_name.GetString(), "DepthwiseConv2D new pad lists is [%d,%d,%d,%d]", new_pad_lists[0],
          new_pad_lists[1], new_pad_lists[2], new_pad_lists[3]);
  if (!have_slice) {
    return GRAPH_FAILED;
  }
  if (!AttrUtils::SetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
    return GRAPH_FAILED;
  }
  OP_LOGD(op_name.GetString(), "Calc DepthwiseConv2D InferDataSlice end!");
  return GRAPH_SUCCESS;
}

INFER_DATA_SLICE_FUNC_REG(DepthwiseConv2D, DepthwiseConv2DInferDataSlice);

static std::vector<int64_t> GetAttrValue(const ge::Operator& op, const std::string& key_name) {
  std::vector<int64_t> list;
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return list);
  if (ge::GRAPH_SUCCESS != op.GetAttr(key_name.c_str(), list)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "GetOpAttr ConstValue failed!");
  }

  return list;
}

static bool CheckListEmpty(const std::string& opName, const std::vector<int64_t>& list, const std::string& attrName) {
  if (list.empty()) {
    CUBE_INNER_ERR_REPORT(opName.c_str(), "the %s is empty !", attrName.c_str());
    return false;
  }

  return true;
}

static void SetInputConst(const bool is_filter_size_const, const bool is_unknown_shape, const bool unknown_rank,
                          ge::OpDescPtr op_desc) {
  if (is_filter_size_const && (is_unknown_shape || unknown_rank)) {
    vector<bool> is_input_const = {false, true, false};
    op_desc->SetIsInputConst(is_input_const);
  }
}

static bool GetPadDepthwiseConv2D(ge::Operator& op, int64_t inH, int64_t inW, int64_t filterH, int64_t filterW,
                                  int64_t strideH, int64_t strideW, int64_t dilationH, int64_t dilationW,
                                  int64_t& padtop, int64_t& padbottom, int64_t& padleft, int64_t& padright) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);
  std::string padStr;
  std::vector<int64_t> padList;
  if (GRAPH_SUCCESS == op.GetAttr("_padding", padList)) {
    op.SetAttr("pads", padList);
  } else if (GRAPH_SUCCESS == op.GetAttr("padding", padStr)) {
    if (padStr.compare("SAME") == 0) {
      int64_t effective_filter_h = (filterH - 1) * dilationH + 1;
      int64_t effective_filter_w = (filterW - 1) * dilationW + 1;
      CHECK(strideH == 0 || strideW == 0,  CUBE_INNER_ERR_REPORT(op_name.GetString(), "stride is 0."),
        return GRAPH_FAILED);
      int64_t tails_h = inH % strideH;
      int64_t tails_w = inW % strideW;
      int64_t pad_h = std::max((tails_h > 0 ? effective_filter_h - tails_h : effective_filter_h -strideH), (int64_t)0);
      int64_t pad_w = std::max((tails_w > 0 ? effective_filter_w - tails_w : effective_filter_w -strideW), (int64_t)0);
      padList.push_back((pad_h >> 1));
      padList.push_back((pad_h >> 1) + (pad_h & 1));
      padList.push_back((pad_w >> 1));
      padList.push_back((pad_w >> 1) + (pad_w & 1));
    } else if (padStr.compare("VALID") == 0) {
      padList.push_back(0);
      padList.push_back(0);
      padList.push_back(0);
      padList.push_back(0);
    } else {
      CUBE_INNER_ERR_REPORT(op_name.GetString(),
                            "padding should be SAME or VALID."
                            " actual is: %s.",
                            padStr.c_str());
      return false;
    }
    op.SetAttr("pads", padList);
  }
  std::vector<int64_t> padVec;
  CHECK_OP_FUNC(op.GetAttr("pads", padVec) == ge::GRAPH_FAILED, return false, "GetOpAttr ConstValue padding failed!");
  auto pSize = padVec.size();
  CHECK_OP_FUNC(pSize != 4, return false, "pads list should be 4d. actual is: %d.", (int)pSize);
  padtop = padVec[kConv2dPadUpIdx];
  padbottom = padVec[kConv2dPadDownIdx];
  padleft = padVec[kConv2dPadLeftIdx];
  padright = padVec[kConv2dPadRightIdx];
  if (padtop < 0 || padbottom < 0 || padleft < 0 || padright < 0) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(),
                          "pads should be positive, "
                          " actual is [%d,%d,%d,%d].",
                          (int)padtop, (int)padbottom, (int)padleft, (int)padright);
    return false;
  }

  return true;
}

// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(DepthwiseConv2D, DepthwiseConv2DVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  auto xTensor = op.GetInputDesc(0);
  auto wTensor = op.GetInputDesc(1);

  auto xShape = xTensor.GetShape().GetDims();
  auto wShape = wTensor.GetShape().GetDims();

  bool unknown_rank = IsUnknownRankShape(xShape);
  if ((!unknown_rank) && (xShape.size() != 4)) {
    if (xShape.size() == 0) {
      CUBE_INNER_ERR_REPORT(op_name.GetString(), "input x shape is empty.");
    } else {
      CUBE_INNER_ERR_REPORT(op_name.GetString(), "input x shape should be 4d. input x shape size if %d",
        (int)xShape.size());
    }
    return GRAPH_FAILED;
  }

  CHECK_OP_FUNC(wShape.size() != 4, return GRAPH_FAILED,
                "input filter shape should be 4d. input filter shape size is %d", (int)wShape.size());

  auto xDtype = xTensor.GetDataType();
  auto wDtype = wTensor.GetDataType();

  CHECK_OP_FUNC(xDtype != wDtype, return GRAPH_FAILED,
                "input x dtype(%d) is differ from filter dtype(%d).", (int)xDtype, (int)wDtype);

  std::vector<int64_t> dilation;
  dilation = GetAttrValue(op, "dilations");
  CHECK_OP_FUNC(!CheckListEmpty(string(op_name.GetString()), dilation, "dilations"),
                return GRAPH_FAILED, "Get dilations failed!");
  std::vector<int64_t> stride;
  stride = GetAttrValue(op, "strides");
  CHECK_OP_FUNC(!CheckListEmpty(string(op_name.GetString()), stride, "strides"),
                return GRAPH_FAILED, "Get stride failed!");
  CHECK_OP_FUNC(stride.size() != 4, return GRAPH_FAILED,
                "stride dim(%d) must be 4!", (int)stride.size());

  std::vector<int64_t> pads;
  pads = GetAttrValue(op, "pads");
  CHECK_OP_FUNC(!CheckListEmpty(string(op_name.GetString()), pads, "pads"), return GRAPH_FAILED, "Get pads failed!");
  CHECK_OP_FUNC(pads.size() != 4, return GRAPH_FAILED, "attr pads(%d) is too large", (int)pads.size());
  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    CHECK_OP_FUNC(data_format != "NCHW" && data_format != "NHWC", return GRAPH_FAILED,
                  "attr data_format(%s) only support NCHW and NHWC", data_format.c_str());
  }
  // attr offset_x need not check
  return GRAPH_SUCCESS;
}

static map<int, std::string> format2str = {
    {ge::FORMAT_NCHW, "NCHW"},   {ge::FORMAT_NHWC, "NHWC"},   {ge::FORMAT_HWCN, "HWCN"},  {ge::FORMAT_DHWNC, "DHWNC"},
    {ge::FORMAT_DHWCN, "DHWCN"}, {ge::FORMAT_NDHWC, "NDHWC"}, {ge::FORMAT_NCDHW, "NCDHW"}};

static map<int, std::string> dtype2str = {
    {ge::DT_FLOAT, "FLOAT"}, {ge::DT_FLOAT16, "FLOAT16"}, {ge::DT_INT8, "INT8"},
    {ge::DT_INT16, "INT16"}, {ge::DT_UINT16, "UINT16"}, {ge::DT_UINT8, "UINT8"},
    {ge::DT_INT32, "INT32"}, {ge::DT_INT64, "INT64"}, {ge::DT_UINT32, "UINT32"},
    {ge::DT_UINT64, "UINT64"}};

/*!
  * Make sure that output shape is larger than 0
  */
bool CorrectConv2DRangeStart(ge::Operator& op, ge::GeTensorDescPtr& x_tensor,
                             std::vector<std::pair<int64_t, int64_t>>& input_range,
                             int32_t kh_dilate, int32_t kw_dilate) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  auto x_shape = x_tensor->MutableShape().GetDims();
  // only support 4D shape
  auto x_format = x_tensor->GetFormat();
  size_t idx_h = 0;
  size_t idx_w = 0;
  if (x_format == FORMAT_NHWC) {
    idx_h = kHDimNHWCIdx;
    idx_w = kWDimNHWCIdx;
  } else {
    idx_h = kHDimNCHWIdx;
    idx_w = kWDimNCHWIdx;
  }
  std::vector<int32_t> pads_list;
  op.GetAttr("pads", pads_list);
  CHECK_INFO(pads_list.size() != kConv2dDimSizeLimit, return false,
        "size of pads list(%zu) is not 4", pads_list.size());

  int64_t low_h = kh_dilate;
  int64_t low_w = kw_dilate;
  // get the smallest shape value allowed
  if (pads_list[0] == -1) {
    // pad is dynamic, get value from shape or range
    if (x_shape[idx_h] != -1) {
      // shape is static
      low_h = x_shape[idx_h] < low_h ? x_shape[idx_h] : low_h;
    } else {
      // shape is dynamic
      low_h = input_range[idx_h].first < low_h ? input_range[idx_h].first : low_h;
    }
    if (x_shape[idx_w] != -1) {
      low_w = x_shape[idx_w] < low_w ? x_shape[idx_w] : low_w;
    } else {
      low_w = input_range[idx_w].first < low_w ? input_range[idx_w].first : low_w;
    }
  } else {
    // pad is static, get value from kernel
    low_h = low_h - pads_list[kConv2dPadUpIdx] - pads_list[kConv2dPadDownIdx];
    low_w = low_w - pads_list[kConv2dPadLeftIdx] - pads_list[kConv2dPadRightIdx];
  }
  // get larger one for left range
  input_range[idx_h].first = input_range[idx_h].first > low_h ? input_range[idx_h].first : low_h;
  input_range[idx_w].first = input_range[idx_w].first > low_w ? input_range[idx_w].first : low_w;
  return true;
}

static bool reset_range(const ge::Operator& op, const std::string& tensor_name) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return false);
  auto tensor_desc = op_desc->MutableInputDesc(tensor_name);
  CHECK_PTR_NULL(tensor_desc, "tensor", return false);
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  tensor_desc->GetShapeRange(shape_range);
  if (shape_range.empty()) {
    return false;
  }
  std::vector<int64_t> shape_sizes = tensor_desc->MutableShape().GetDims();
  if (shape_range.size() != shape_sizes.size()) {
    return false;
  }
  bool reset_flag = false;
  for (size_t i = 0; i < shape_sizes.size(); i++) {
    if (shape_sizes[i] > 0 && (shape_range[i].first != shape_sizes[i] || shape_range[i].second != shape_sizes[i])) {
      reset_flag = true;
      shape_range[i].first = shape_sizes[i];
      shape_range[i].second = shape_sizes[i];
    }
  }

  if (reset_flag) {
    tensor_desc->SetShapeRange(shape_range);
    OP_LOGW(op_name.GetString(), "%s range does not match the shape value, has been fixed.", tensor_name.c_str());
  }
  return reset_flag;
}

static bool GetDimInFormat(const std::string& opName, const std::string& formatStr, const std::string& dimName,
                           int64_t& dimPosition) {
  dimPosition = formatStr.find(dimName);
  if (dimPosition < 0) {
    CUBE_INNER_ERR_REPORT(opName.c_str(), "Position(%s) is invalid: %d, which format is %s.",
                          dimName.c_str(), (int)dimPosition, formatStr.c_str());
    return false;
  }
  return true;
}

static void GetDepthwiseConv2dOutShapeRange(const std::string& pad_str,
                                            size_t idx,
                                            const vector<int64_t>& attr_params,
                                            const std::vector<std::pair<int64_t, int64_t>>& fm_range,
                                            std::vector<std::pair<int64_t, int64_t>>& out_range) {
  size_t attr_idx = 0;
  int64_t stride = attr_params[attr_idx++];
  int64_t dilation = attr_params[attr_idx++];
  int64_t pad = attr_params[attr_idx++];
  int64_t kernel = attr_params[attr_idx++];
  int64_t low = fm_range[idx].first;
  int64_t high = fm_range[idx].second;
  if (pad_str == "SAME") {
    out_range[idx].first = (low + stride -1) / stride;
    out_range[idx].second = (high + stride -1) / stride;
  } else {
    out_range[idx].first = (low + pad - dilation * (kernel - 1) - 1) / stride + 1;
    out_range[idx].second = (high + pad - dilation * (kernel - 1) - 1) / stride + 1;
  }
  out_range[idx].first = std::max(out_range[idx].first, kDynamicRangeLowerBound);
  out_range[idx].second = std::min(out_range[idx].second, kDynamicRangeUpperBound);
  if (high == -1) {
    out_range[idx].second = high;
  }
}

static bool SetDepthwiseConv2dOutShapeRange(ge::Operator& op,
                                            const vector<int64_t>& attr_params,
                                            ge::GeShape& y_shape,
                                            ge::GeTensorDescPtr& x_tensor,
                                            ge::GeTensorDescPtr& y_tensor) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  auto &x_shape = x_tensor->MutableShape();
  auto x_format = x_tensor->GetFormat();

  size_t idx = 0;
  int64_t strh = attr_params[idx++];
  int64_t strw = attr_params[idx++];
  int64_t dilh = attr_params[idx++];
  int64_t dilw = attr_params[idx++];
  int64_t padh = attr_params[idx++];
  int64_t padw = attr_params[idx++];
  int64_t outc = attr_params[idx++];
  int64_t kh = attr_params[idx++];
  int64_t kw = attr_params[idx++];

  size_t idx_h = 0;
  size_t idx_w = 0;
  size_t idx_c = 0;
  if (x_format == FORMAT_NHWC) {
    idx_h = kHDimNHWCIdx;
    idx_w = kWDimNHWCIdx;
    idx_c = kCDimNHWCIdx;
  } else if (x_format == FORMAT_NCHW) {
    idx_c = kCDimNCHWIdx;
    idx_h = kHDimNCHWIdx;
    idx_w = kWDimNCHWIdx;
  }

  // update pads if padding is SAME
  std::string pad_str;
  if (x_shape.GetDimNum() == kConv2dInputSizeLimit && GRAPH_SUCCESS == op.GetAttr("padding", pad_str) &&
      pad_str == "SAME" && (x_shape.GetDim(idx_h) == -1 || x_shape.GetDim(idx_w) == -1)) {
    op.SetAttr("pads", {-1, -1, -1, -1});
    OP_LOGD(op_name.GetString(), "set pads to {-1, -1, -1, -1} when padding is SAME in dynamic_shape");
  }

  OP_LOGD(op_name.GetString(), "dynamic shape set range");
  std::vector<std::pair<int64_t, int64_t>> fm_range;
  x_tensor->GetShapeRange(fm_range);
  if (x_shape.GetDim(idx_h) == -1) {
    y_shape.SetDim(idx_h, -1);
  }
  if (x_shape.GetDim(idx_w) == -1) {
    y_shape.SetDim(idx_w, -1);
  }
  if (!fm_range.empty()) {
    for (size_t i = 0; i < fm_range.size(); i++) {
      OP_LOGD(op_name.GetString(), "fmap Range[%u] is (%lld, %lld)", i, fm_range[i].first, fm_range[i].second);
    }

    std::vector<std::pair<int64_t, int64_t>> out_range(fm_range);
    out_range[idx_c] = std::make_pair((int64_t)outc, (int64_t)outc);
    out_range[idx_h] = std::make_pair(y_shape.GetDim(idx_h), y_shape.GetDim(idx_h));
    out_range[idx_w] = std::make_pair(y_shape.GetDim(idx_w), y_shape.GetDim(idx_w));
    if (x_shape.GetDim(idx_h) == -1) {
      vector<int64_t> attr_params_h = {strh, dilh, padh, kh};
      GetDepthwiseConv2dOutShapeRange(pad_str, idx_h, attr_params_h, fm_range, out_range);
    }
    if (x_shape.GetDim(idx_w) == -1) {
      vector<int64_t> attr_params_w = {strw, dilw, padw, kw};
      GetDepthwiseConv2dOutShapeRange(pad_str, idx_w, attr_params_w, fm_range, out_range);
    }
    y_tensor->SetShapeRange(out_range);
    for (size_t i = 0; i < out_range.size(); i++) {
      OP_LOGD(op_name.GetString(), "output Range[%u] is (%lld, %lld)", i, out_range[i].first, out_range[i].second);
    }
  }
  return true;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DepthwiseConv2DInferShape) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);
  OP_LOGI(op_name.GetString(), "Enter op_proto inferfunction!");
  PROFILING_PROTO_INIT(op_name.GetString());

  int64_t nPosition = 0;
  int64_t cPosition = 0;
  int64_t hPosition = 0;
  int64_t wPosition = 0;
  int64_t inN = -1;
  int64_t inC = -1;
  int64_t inH = -1;
  int64_t inW = -1;
  int64_t outH = 0;
  int64_t outW = 0;
  int64_t outC = 0;
  int64_t filterN = 0;
  int64_t filterC = 0;
  int64_t filterH = 0;
  int64_t filterW = 0;
  int64_t dilationH = 0;
  int64_t dilationW = 0;
  int64_t strideH = 0;
  int64_t strideW = 0;
  int64_t effectiveFilterH = 0;
  int64_t effectiveFilterW = 0;
  int64_t padtop = 0;
  int64_t padbottom = 0;
  int64_t padleft = 0;
  int64_t padright = 0;

  std::vector<int64_t> dilation;
  dilation = GetAttrValue(op, "dilations");
  CHECK_OP_FUNC(!CheckListEmpty(string(op_name.GetString()), dilation, "dilations"), return GRAPH_FAILED,
                "Get dilations failed!");

  std::vector<int64_t> stride;
  stride = GetAttrValue(op, "strides");
  CHECK_OP_FUNC(!CheckListEmpty(string(op_name.GetString()), stride, "strides"), return GRAPH_FAILED,
                "Get stride failed!");

  std::string dataFormat = "";
  CHECK_OP_FUNC(ge::GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat), return GRAPH_FAILED,
                "get data_format attr failed");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto tensorDescIn = op_desc->MutableInputDesc(0);
  auto tensorDescW = op_desc->MutableInputDesc(1);

  auto &shapeIn = tensorDescIn->MutableShape();
  auto &shapeW = tensorDescW->MutableShape();

  bool unknown_rank = shapeIn.IsUnknownDimNum();

  auto dataTypeIn = tensorDescIn->GetDataType();

  bool get_dim_in_format = GetDimInFormat(string(op_name.GetString()), dataFormat, "N", nPosition) &&
    GetDimInFormat(string(op_name.GetString()), dataFormat, "C", cPosition) &&
    GetDimInFormat(string(op_name.GetString()), dataFormat, "H", hPosition) &&
    GetDimInFormat(string(op_name.GetString()), dataFormat, "W", wPosition);
  if (!get_dim_in_format) {
    return GRAPH_FAILED;
  }

  if (!unknown_rank) {
    // NC1HWC0(NCHW)
    inN = shapeIn.GetDim(nPosition);
    inC = shapeIn.GetDim(cPosition);
    inH = shapeIn.GetDim(hPosition);
    inW = shapeIn.GetDim(wPosition);
  }

  int64_t fnPosition = 0;
  int64_t fcPosition = 0;
  int64_t fhPosition = 0;
  int64_t fwPosition = 0;

  Format filterFormat = tensorDescW->GetFormat();
  std::string filterFormatStr = format2str[filterFormat];

  get_dim_in_format = GetDimInFormat(string(op_name.GetString()), filterFormatStr, "N", fnPosition) &&
    GetDimInFormat(string(op_name.GetString()), filterFormatStr, "C", fcPosition) &&
    GetDimInFormat(string(op_name.GetString()), filterFormatStr, "H", fhPosition) &&
    GetDimInFormat(string(op_name.GetString()), filterFormatStr, "W", fwPosition);
  if (!get_dim_in_format) {
    return GRAPH_FAILED;
  }

  filterN = shapeW.GetDim(fnPosition);
  filterC = shapeW.GetDim(fcPosition);
  filterH = shapeW.GetDim(fhPosition);
  filterW = shapeW.GetDim(fwPosition);

  // set range
  // when static op or dynamic op phase_running, is_dynamic == false
  bool is_dynamic = shapeIn.IsUnknownShape() && (!shapeIn.IsUnknownDimNum());

  if (is_dynamic && (inC == -1)) {
      shapeIn.SetDim(cPosition, filterC);
      inC = filterC;
  }

  CHECK_OP_FUNC((!unknown_rank) && (inC < 1), return GRAPH_FAILED,
                "x channel should be greater than or equal to 1. actual is: %d", (int)inC)

  int64_t groups = 1;
  op.GetAttr("groups", groups);
  groups = inC;
  op.SetAttr("groups", groups);
  PROFILING_PROTO_AFTER_GET_SHAPE_REG()
  dilationH = dilation.at(hPosition);
  dilationW = dilation.at(wPosition);
  strideH = stride.at(hPosition);
  strideW = stride.at(wPosition);

  if (GetPadDepthwiseConv2D(op, inH, inW, filterH, filterW, strideH, strideW, dilationH, dilationW, padtop,
                            padbottom, padleft, padright) == false) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "get pads attrs failed.");
    return GRAPH_FAILED;
  }

  effectiveFilterH = (filterH - 1) * dilationH + 1;
  effectiveFilterW = (filterW - 1) * dilationW + 1;
  CHECK(strideH == 0 || strideW == 0, CUBE_INNER_ERR_REPORT(op_name.GetString(), "stride is 0."), return GRAPH_FAILED);
  outH = (inH + padtop + padbottom - effectiveFilterH) / strideH + 1;
  outW = (inW + padleft + padright - effectiveFilterW) / strideW + 1;
  outC = filterN * filterC;
  if (unknown_rank) {
    outH = -1;
    outW = -1;
  }
  PROFILING_PROTO_AFTER_INFER_SHAPE_REG()
  auto tensordesc_output = op_desc->MutableOutputDesc(0);
  auto formatOut = tensordesc_output->GetFormat();
  auto &shapeOut = tensordesc_output->MutableShape();
  shapeOut.SetDimNum(4);
  // NC1HWC0(NCHW/NHWC)
  if (formatOut == FORMAT_NCHW) {
    shapeOut.SetDim(0, inN);
    shapeOut.SetDim(1, outC);
    shapeOut.SetDim(2, outH);
    shapeOut.SetDim(3, outW);
  } else if (formatOut == FORMAT_NHWC) {
    shapeOut.SetDim(0, inN);
    shapeOut.SetDim(1, outH);
    shapeOut.SetDim(2, outW);
    shapeOut.SetDim(3, outC);
  } else {
    CUBE_INNER_ERR_REPORT(op_name.GetString(),
                          "output y format should be NCHW or NHWC."
                          " actual is: %d", (int)formatOut);
    return GRAPH_FAILED;
  }

  if ((dataTypeIn == ge::DT_INT8) || (dataTypeIn == ge::DT_INT4)) {
    tensordesc_output->SetDataType(ge::DT_INT32);
  } else {
    tensordesc_output->SetDataType(dataTypeIn);
  }

  if (is_dynamic) {
    vector<int64_t> attr_params = {strideH, strideW, dilationH, dilationW,
                                   padtop + padbottom, padleft + padright,
                                   outC, filterH, filterW};
    if (!SetDepthwiseConv2dOutShapeRange(op, attr_params, shapeOut, tensorDescIn, tensordesc_output)) {
      return GRAPH_FAILED;
    }
  }
  // fuzz_build switch
  bool fuzz_build = false;
  op.GetAttr((ge::ATTR_NAME_FUZZ_BUILD).c_str(), fuzz_build);
  // fuzz build
  if ((!unknown_rank) && fuzz_build) {
    OP_LOGD(op_name.GetString(), "start fuzz build.");
    // change pads to -1 when padding is SAME
    std::string pad_str;
    op.GetAttr("padding", pad_str);
    if (pad_str == "SAME") {
      op.SetAttr("pads", {-1, -1, -1, -1});
      OP_LOGD(op_name.GetString(), "set pads to {-1, -1, -1, -1} when padding is SAME in fuzzy build.");
    }
  }

  OP_LOGI(op_name.GetString(), "leave op_proto inferfunction!");
  PROFILING_PROTO_END()
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DepthwiseConv2D, DepthwiseConv2DInferShape);

// Registered verify function
VERIFY_FUNC_REG(DepthwiseConv2D, DepthwiseConv2DVerify);

static graphStatus VerifyDepthwiseConv2DbpPadding(ge::Operator& op) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  std::string pad;
  if (op.GetAttr("padding", pad) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  if (pad.compare("SAME") != 0 && pad.compare("VALID") != 0) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "padding must be SAME or VALID. actual is: %s", pad.c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

static graphStatus VerifyDepthwiseConv2DbpPads(ge::Operator& op) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  std::vector<int64_t> pads;
  if (op.GetAttr("pads", pads) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  if (pads.size() < kConv2dPadSizeLimit) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "op pads's size is illegal,pads.size:%zu", pads.size());
    return GRAPH_FAILED;
  }
  OP_LOGI(op_name.GetString(), "op pads: [top, bottom, left, right] is %s", ops::to_string(pads).c_str());
  if (std::any_of(pads.begin(), pads.end(), [](int64_t val) {return val < 0;})) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "op get pads is illegal");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// ----------------DepthwiseConv2DBackpropInputD Op-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(DepthwiseConv2DBackpropInputD, DepthwiseConv2DBackpropInputDVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  std::vector<int64_t> input_size;
  input_size = GetAttrValue(op, "input_size");
  if (!CheckListEmpty(string(op_name.GetString()), input_size, "input_size")) {
    return GRAPH_FAILED;
  }

  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(string(op_name.GetString()), strides, "strides")) {
    return GRAPH_FAILED;
  }
  if (strides.size() != DIM_SIZE4) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "strides must be NCHW!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dilations;
  dilations = GetAttrValue(op, "dilations");
  if (!CheckListEmpty(string(op_name.GetString()), dilations, "dilations")) {
    return GRAPH_FAILED;
  }
  if (dilations.size() != DIM_SIZE4) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "dilations must be NCHW!");
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format != "NCHW" && data_format != "NHWC") {
      CUBE_INNER_ERR_REPORT(op_name.GetString(), "attr data_format(%s) only support NCHW and NHWC",
        data_format.c_str());
      return GRAPH_FAILED;
    }
  }
  if (GRAPH_SUCCESS != VerifyDepthwiseConv2DbpPads(op)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "get pads failed.");
    return GRAPH_FAILED;
  }

  if (op.GetInputDesc(0).GetDataType() != op.GetInputDesc(1).GetDataType()) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "The type of filter and out_backprop must be same!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DepthwiseConv2DBackpropInputDInferShape) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGI(op_name.GetString(), "Enter DepthwiseConv2DBackpropInputD inferfunction!");
  std::vector<int64_t> input_size;
  int64_t groups = 0;
  input_size = GetAttrValue(op, "input_size");

  DataType output_dtype = op.GetInputDescByName("out_backprop").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDescByName("input_grad");
  tensordesc_output.SetShape(Shape(input_size));
  tensordesc_output.SetDataType(output_dtype);
  (void)op.UpdateOutputDesc("input_grad", tensordesc_output);

  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(string(op_name.GetString()), strides, "strides")) {
    return GRAPH_FAILED;
  }
  if (strides.size() != DIM_SIZE4) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "strides must be NCHW!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dilations;
  dilations = GetAttrValue(op, "dilations");
  if (!CheckListEmpty(string(op_name.GetString()), dilations, "dilations")) {
    return GRAPH_FAILED;
  }
  if (dilations.size() != DIM_SIZE4) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "dilations must be NCHW!");
    return GRAPH_FAILED;
  }

  std::string dataFormat = "";
  CHECK_OP_FUNC(ge::GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat),
                return GRAPH_FAILED, "get data_format attr failed");

  int64_t hPosition = 0;
  int64_t wPosition = 0;
  int64_t fhPosition = 0;
  int64_t fwPosition = 0;
  int64_t fcPosition = 0;
  int64_t fnPosition = 0;
  int64_t inH = 0;
  int64_t inW = 0;
  int64_t filterH = 0;
  int64_t filterW = 0;
  int64_t dilationH = 0;
  int64_t dilationW = 0;
  int64_t strideH = 0;
  int64_t strideW = 0;
  int64_t padtop = 0;
  int64_t padbottom = 0;
  int64_t padleft = 0;
  int64_t padright = 0;

  auto tensorDescW = op.GetInputDesc(0);
  auto shapeW = tensorDescW.GetShape();

  Format filterFormat = tensorDescW.GetFormat();
  std::string filterFormatStr = format2str[filterFormat];
  if (!GetDimInFormat(string(op_name.GetString()), filterFormatStr, "H", fhPosition)) {
    return GRAPH_FAILED;
  }
  if (!GetDimInFormat(string(op_name.GetString()), filterFormatStr, "W", fwPosition)) {
    return GRAPH_FAILED;
  }
  if (!GetDimInFormat(string(op_name.GetString()), filterFormatStr, "C", fcPosition)) {
    return GRAPH_FAILED;
  }
  if (!GetDimInFormat(string(op_name.GetString()), filterFormatStr, "N", fnPosition)) {
    return GRAPH_FAILED;
  }

  filterH = shapeW.GetDim(fhPosition);
  filterW = shapeW.GetDim(fwPosition);
  groups = tensorDescW.GetOriginShape().GetDim(fcPosition) * tensorDescW.GetOriginShape().GetDim(fnPosition);
  op.SetAttr("groups", groups);

  if (!GetDimInFormat(string(op_name.GetString()), dataFormat, "H", hPosition)) {
    return GRAPH_FAILED;
  }
  if (!GetDimInFormat(string(op_name.GetString()), dataFormat, "W", wPosition)) {
    return GRAPH_FAILED;
  }

  // NC1HWC0(NCHW)
  inH = input_size[hPosition];
  inW = input_size[wPosition];

  dilationH = dilations.at(hPosition);
  dilationW = dilations.at(wPosition);
  strideH = strides.at(hPosition);
  strideW = strides.at(wPosition);

  CHECK_OP_FUNC(false == GetPadDepthwiseConv2D(op, inH, inW, filterH, filterW, strideH, strideW,
                dilationH, dilationW, padtop, padbottom, padleft, padright),
                return GRAPH_FAILED, "update pads attrs failed.");

  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DepthwiseConv2DBackpropInputD, DepthwiseConv2DBackpropInputDInferShape);
// Registered verify function
VERIFY_FUNC_REG(DepthwiseConv2DBackpropInputD, DepthwiseConv2DBackpropInputDVerify);

// ----------------DepthwiseConv2DBackpropInput Op-------------------
// Obtains the value of the constant tensor.
static void GetConstValue(const Tensor& const_tensor, const DataType& dtype, std::vector<int64_t>& const_data) {
  const uint8_t* constData = const_tensor.GetData();
  size_t size;
  if (dtype == ge::DT_INT32) {
    size = const_tensor.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back(*((int32_t*)constData + i));
    }
  } else {
    size = const_tensor.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back(*((int64_t*)constData + i));
    }
  }
}

static void set_conv2d_backprop_input_dim_range(const std::string& pad_str,
                                                size_t idx,
                                                const vector<int64_t>& attrParams,
                                                const std::vector<std::pair<int64_t, int64_t>>& dy_range,
                                                std::vector<std::pair<int64_t, int64_t>>& dx_range) {
  size_t attrIdx = 0;
  int64_t stride = attrParams[attrIdx++];
  int64_t kernel = attrParams[attrIdx++];
  int64_t pad = attrParams[attrIdx++];
  int64_t low = dy_range[idx].first;
  int64_t high = dy_range[idx].second;
  if (pad_str == "SAME") {
    dx_range[idx].first = stride * (low - 1) + 1;
    dx_range[idx].second = stride * high;
  } else {
    dx_range[idx].first = stride * (low - 1) + kernel - pad;
    dx_range[idx].second = stride * (high - 1) + kernel - pad + stride - 1;
  }
  dx_range[idx].first = std::max(dx_range[idx].first, kDynamicRangeLowerBound);
  dx_range[idx].second = std::min(dx_range[idx].second, kDynamicRangeUpperBound);
  if (high == -1) {
    dx_range[idx].second = high;
  }
}

static bool set_conv2d_backprop_input_out_range(ge::Operator& op, const vector<int32_t>& dy_position,
                                                const vector<int32_t>& input_position,
                                                const vector<int64_t>& k_value,
                                                const vector<int32_t>& stride_value,
                                                std::vector<std::pair<int64_t, int64_t>>& dx_range,
                                                std::vector<std::pair<int64_t, int64_t>>& dy_range,
                                                ge::GeTensorDescPtr& y_desc, std::string& pad_str,
                                                const std::vector<int64_t>& dy_sizes,
                                                const std::vector<int64_t>& dx_sizes) {
  std::vector<int32_t> pads_list;
  op.GetAttr("pads", pads_list);
  int32_t pad_up = pads_list[kConv2dPadUpIdx];
  int32_t pad_down = pads_list[kConv2dPadDownIdx];
  int32_t pad_left = pads_list[kConv2dPadLeftIdx];
  int32_t pad_right = pads_list[kConv2dPadRightIdx];
  int64_t khext = k_value[0];
  int64_t kwext = k_value[1];
  int32_t stride_h = stride_value[0];
  int32_t stride_w = stride_value[1];
  AscendString op_type;
  CHECK(op.GetOpType(op_type) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_type"), return false);
  if (string(op_type.GetString()) == "Conv2DTranspose") {
    std::vector<int32_t> output_padding_list;
    op.GetAttr("output_padding", output_padding_list);
    int32_t outputpadding_h = output_padding_list[dy_position[kHDimNCHWIdx]];
    int32_t outputpadding_w = output_padding_list[dy_position[kWDimNCHWIdx]];
    khext = outputpadding_h + khext;
    kwext = outputpadding_w + khext;
  }

  if (!dy_range.empty() && dy_range.size() == dy_sizes.size()) {
    dx_range[input_position[kNDimNCHWIdx]] = dy_range[dy_position[kNDimNCHWIdx]];
    if (dx_sizes[input_position[kHDimNCHWIdx]] == -1) {
      vector<int64_t> attr_params_h = {stride_h, khext, pad_up + pad_down};
      set_conv2d_backprop_input_dim_range(pad_str, input_position[kHDimNCHWIdx], attr_params_h, dy_range, dx_range);
    }
    if (dx_sizes[input_position[kWDimNCHWIdx]] == -1) {
      vector<int64_t> attr_params_w = {stride_w, kwext, pad_left + pad_right};
      set_conv2d_backprop_input_dim_range(pad_str, input_position[kWDimNCHWIdx], attr_params_w, dy_range, dx_range);
    }
    y_desc->SetShapeRange(dx_range);
  }
  return true;
}

static bool set_conv2d_backprop_input_out_shape_range(ge::Operator& op, std::string& pad_str,
                                                      const std::vector<int64_t>& dy_sizes,
                                                      Format dy_format,
                                                      std::vector<std::pair<int64_t, int64_t>>& dy_range,
                                                      const std::vector<int64_t>& filter_sizes, Format filter_format,
                                                      Format dx_format,
                                                      std::vector<std::pair<int64_t, int64_t>>& dx_range,
                                                      ge::GeTensorDescPtr& y_desc, ge::GeTensorDescPtr& x_desc,
                                                      const int64_t& groups,
                                                      bool& unknown_rank, const std::vector<int64_t>& attr_params) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  bool dy_range_invalid = dy_range.size() != dy_sizes.size() && dy_range.size() != 0 && !unknown_rank;
  if (dy_range_invalid) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(),
                          "length of range(%zu) in dyanmic shape must be equal to the length of shape(%zu), or equal to 0.",
                          dy_range.size(), dy_sizes.size());
    return false;
  }
  bool dy_range_empty = dy_range.size() == 0 && !unknown_rank;
  if (dy_range_empty) {
    dy_range.resize(dy_sizes.size());
    for (size_t i = 0; i < dy_sizes.size(); i++) {
      dy_range[i].first = std::max(dy_sizes[i], kDynamicRangeLowerBound);
      dy_range[i].second = dy_sizes[i];
    }
    x_desc->SetShapeRange(dy_range);
  }

  CHECK_KEY_IN_MAP(format2str, dx_format, "dx_format", return false);
  std::string dx_format_str = format2str[dx_format];
  int32_t h_input_position = dx_format_str.find("H");
  int32_t w_input_position = dx_format_str.find("W");
  int32_t c_input_position = dx_format_str.find("C");
  int32_t n_input_position = dx_format_str.find("N");

  CHECK_KEY_IN_MAP(format2str, filter_format, "filter_format", return false);
  std::string filter_format_str = format2str[filter_format];
  int32_t h_filter_position = filter_format_str.find("H");
  int32_t w_filter_position = filter_format_str.find("W");
  int32_t c_filter_position = filter_format_str.find("C");
  int64_t filter_h = filter_sizes[h_filter_position];
  int64_t filter_w = filter_sizes[w_filter_position];
  int64_t filter_c = filter_sizes[c_filter_position];

  if (GRAPH_SUCCESS == op.GetAttr("padding", pad_str)) {
    if (pad_str == "SAME") {
      op.SetAttr("pads", {-1, -1, -1, -1});
    } else if (pad_str == "VALID") {
      op.SetAttr("pads", {0, 0, 0, 0});
    }
  }

  bool set_value_range = !dx_range.empty() && dx_range.size() == kConv2dDimSizeLimit && !unknown_rank;
  if (set_value_range) {
    if (dx_range[c_input_position].first == -1 ||
        dx_range[c_input_position].first != dx_range[c_input_position].second) {
      int64_t cin = groups * filter_c;
      dx_range[c_input_position].first = cin;
      dx_range[c_input_position].second = cin;
    }
    CHECK_OP_FUNC((y_desc->SetShapeRange(dx_range)) != GRAPH_SUCCESS, return false, "do set dx range failed.");
    OP_LOGD(op_name.GetString(), "get value_range success from GE.");
  } else {
    std::vector<int64_t> dx_sizes = y_desc->MutableShape().GetDims();
    bool dx_sizes_invalid = dx_sizes.empty() || dx_sizes.size() != kConv2dDimSizeLimit;
    if (dx_sizes_invalid) {
      OP_LOGE(op_name.GetString(), "dx_sizes list should be 4D. actual is: %lu.", dx_sizes.size());
      map<string, string> err_map;
      err_map["param_name"] = "dx_sizes";
      err_map["op_name"] = op_name.GetString();
      err_map["expected_value"] = "4D";
      err_map["input_value"] = std::to_string(dx_sizes.size()) + "D.";
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }
    size_t idx = 0;
    int32_t stride_h = attr_params[idx++];
    int32_t stride_w = attr_params[idx++];
    int32_t dilation_h = attr_params[idx++];
    int32_t dilation_w = attr_params[idx++];

    if (unknown_rank) {
      vector<int64_t> dx_shape;
      dx_shape.resize(kConv2dDimSizeLimit);
      dx_shape[n_input_position] = -1;
      dx_shape[h_input_position] = -1;
      dx_shape[w_input_position] = -1;
      dx_shape[c_input_position] = groups * filter_c;
      y_desc->SetShape(GeShape(dx_shape));
      return true;
    }

    CHECK_KEY_IN_MAP(format2str, dy_format, "dy_format", return false);
    std::string dy_format_str = format2str[dy_format];
    int32_t w_dy_position = dy_format_str.find("W");
    int32_t h_dy_position = dy_format_str.find("H");
    int32_t c_dy_position = dy_format_str.find("C");
    int32_t n_dy_position = dy_format_str.find("N");

    int64_t dx_h = dx_sizes[h_input_position];
    int64_t dx_w = dx_sizes[w_input_position];

    int64_t khext = (filter_h - 1) * dilation_h + 1;
    int64_t kwext = (filter_w - 1) * dilation_w + 1;

    dx_range.resize(kConv2dDimSizeLimit);
    dx_range[c_input_position] = std::make_pair(filter_c * groups, filter_c * groups);
    dx_range[h_input_position] = std::make_pair(dx_h, dx_h);
    dx_range[w_input_position] = std::make_pair(dx_w, dx_w);
    dx_range[n_input_position] = std::make_pair(dy_sizes[n_dy_position], dy_sizes[n_dy_position]);

    vector<int32_t> dy_position = {n_dy_position, c_dy_position, h_dy_position, w_dy_position};
    vector<int32_t> input_position = {n_input_position, c_input_position, h_input_position, w_input_position};
    vector<int32_t> stride_value = {stride_h, stride_w};
    vector<int64_t> k_value = {khext, kwext};
    if (!set_conv2d_backprop_input_out_range(op, dy_position, input_position, k_value, stride_value, dx_range,
                                        dy_range, y_desc, pad_str, dy_sizes, dx_sizes)) {
        return false;
        }
  }
  return true;
}

static bool check_conv2d_backprop_input_pads(ge::Operator& op,
                                             const std::vector<int64_t>& dy_sizes,
                                             Format dy_format,
                                             const std::vector<int64_t>& filter_sizes, Format filter_format,
                                             const std::vector<int64_t>& dx_sizes, Format dx_format,
                                             const std::vector<int64_t>& attr_params) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  std::string pad_str;
  if (GRAPH_SUCCESS == op.GetAttr("padding", pad_str)) {
    return true;
  }
  size_t idx = 0;
  int64_t stride_h = attr_params[idx++];
  int64_t stride_w = attr_params[idx++];
  int64_t dilation_h = attr_params[idx++];
  int64_t dilation_w = attr_params[idx++];
  std::vector<int64_t> pads_list;
  op.GetAttr("pads", pads_list);
  int64_t pad_up = pads_list[kConv2dPadUpIdx];
  int64_t pad_down = pads_list[kConv2dPadDownIdx];
  int64_t pad_left = pads_list[kConv2dPadLeftIdx];
  int64_t pad_right = pads_list[kConv2dPadRightIdx];

  CHECK_KEY_IN_MAP(format2str, filter_format, "filter_format", return false);
  std::string filter_format_str = format2str[filter_format];
  int64_t h_filter_position = filter_format_str.find("H");
  int64_t w_filter_position = filter_format_str.find("W");
  int64_t filter_h = filter_sizes[h_filter_position];
  int64_t filter_w = filter_sizes[w_filter_position];
  CHECK_KEY_IN_MAP(format2str, dx_format, "dx_format", return false);
  std::string dx_format_str = format2str[dx_format];
  int64_t h_input_position = dx_format_str.find("H");
  int64_t w_input_position = dx_format_str.find("W");
  CHECK_KEY_IN_MAP(format2str, dy_format, "dy_format", return false);
  std::string dy_format_str = format2str[dy_format];
  int64_t w_dy_position = dy_format_str.find("W");
  int64_t h_dy_position = dy_format_str.find("H");

  int64_t dx_h = dx_sizes[h_input_position];
  int64_t dx_w = dx_sizes[w_input_position];
  int64_t khext = (filter_h - 1) * dilation_h + 1;
  int64_t kwext = (filter_w - 1) * dilation_w + 1;

  AscendString op_type;
  CHECK(op.GetOpType(op_type) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_type"), return false);
  if (string(op_type.GetString()) == "Conv2DTranspose") {
    std::vector<int64_t> output_padding_list;
    op.GetAttr("output_padding", output_padding_list);
    int64_t outputpadding_h = output_padding_list[h_dy_position];
    int64_t outputpadding_w = output_padding_list[w_dy_position];
    khext = outputpadding_h + ((filter_h - 1) * dilation_h + 1);
    kwext = outputpadding_w + ((filter_w - 1) * dilation_w + 1);
  }

  int64_t dy_h = dy_sizes[h_dy_position];
  int64_t dy_w = dy_sizes[w_dy_position];
  int64_t dy_h_new = (dx_h + pad_up + pad_down - khext) / stride_h + 1;
  int64_t dy_w_new = (dx_w + pad_left + pad_right - kwext) / stride_w + 1;

  CHECK_INFO(dy_h_new != dy_h || dy_w_new != dy_w, return false, "check pads attrs failed.");
  return true;
}

static void reset_conv2d_backprop_input_out_shape(const ge::Operator& op, const std::vector<int64_t>&dy_sizes,
                                                  Format dy_format, std::vector<int64_t>& input_sizes,
                                                  Format input_format) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return);

  CHECK_KEY_IN_MAP(format2str, input_format, "input_format", return);
  std::string dx_format_str = format2str[input_format];
  int32_t h_input_position = dx_format_str.find("H");
  int32_t w_input_position = dx_format_str.find("W");
  int32_t c_input_position = dx_format_str.find("C");
  int32_t n_input_position = dx_format_str.find("N");
  CHECK_KEY_IN_MAP(format2str, dy_format, "dy_format", return);
  std::string dy_format_str = format2str[dy_format];
  int32_t h_dy_position = dy_format_str.find("H");
  int32_t w_dy_position = dy_format_str.find("W");
  int32_t c_dy_position = dy_format_str.find("C");
  int32_t n_dy_position = dy_format_str.find("N");
  if (dy_sizes[n_dy_position] == -1) {
    input_sizes[n_input_position] = -1;
  }
  if (dy_sizes[c_dy_position] == -1) {
    OP_LOGW(op_name.GetString(), "input x channel is unknow, fixed channel = %d.", (int)input_sizes[c_input_position]);
  }
  if (dy_sizes[h_dy_position] == -1) {
    input_sizes[h_input_position] = -1;
  }
  if (dy_sizes[w_dy_position] == -1) {
    input_sizes[w_input_position] = -1;
  }
}

// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(DepthwiseConv2DBackpropInput, DepthwiseConv2DBackpropInputVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  TensorDesc tensordesc_input = op.GetInputDescByName("out_backprop");
  auto input_grad_shape = tensordesc_input.GetShape().GetDims();
  Tensor input_size_tensor;
  std::vector<int64_t> input_size;

  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(string(op_name.GetString()), strides, "strides")) {
    return GRAPH_FAILED;
  }
  if (strides.size() != DIM_SIZE4) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "strides must be NCHW!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dilations;
  dilations = GetAttrValue(op, "dilations");
  if (!CheckListEmpty(string(op_name.GetString()), dilations, "dilations")) {
    return GRAPH_FAILED;
  }
  if (dilations.size() != DIM_SIZE4) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "dilations must be NCHW!");
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format != "NCHW" && data_format != "NHWC") {
      CUBE_INNER_ERR_REPORT(op_name.GetString(), "attr data_format(%s) only support NCHW and NHWC",
        data_format.c_str());
      return GRAPH_FAILED;
    }
  }

  if (VerifyDepthwiseConv2DbpPadding(op) != GRAPH_SUCCESS && VerifyDepthwiseConv2DbpPads(op) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "get padding and pads both failed.");
    return GRAPH_FAILED;
  }

  // filter idx is 1 and out_backprop idx is 2
  if (op.GetInputDesc(1).GetDataType() != op.GetInputDesc(2).GetDataType()) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "The type of filter and out_backprop must be same!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DepthwiseConv2DBackpropInputInferShape) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGI(op_name.GetString(), "Enter DepthwiseConv2DBackpropInput inferfunction!");
  int64_t groups = 0;
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);
  std::vector<std::string> input_infer_depends = {"input_size"};
  op_desc->SetOpInferDepends(input_infer_depends);
  auto x_desc = op_desc->MutableInputDesc("out_backprop");
  auto filter_desc = op_desc->MutableInputDesc("filter");
  auto y_desc = op_desc->MutableOutputDesc("input_grad");
  auto input_sizes_desc = op_desc->MutableInputDesc("input_size");
  Format filter_format = filter_desc->GetFormat();
  Format input_format = y_desc->GetFormat();
  Format dy_format = x_desc->GetFormat();
  std::vector<int64_t> filter_sizes = filter_desc->MutableShape().GetDims();
  std::vector<int64_t> dy_sizes = x_desc->MutableShape().GetDims();

  // set dtype of output desc
  auto out_backprop_dtype = x_desc->GetDataType();
  y_desc->SetDataType(out_backprop_dtype);
  bool is_dynamic = false;
  bool unknown_rank = IsUnknownRankShape(dy_sizes);
  bool is_input_size_const = false;
  std::vector<int64_t> input_sizes;
  Tensor input_sizes_tensor;

  if (op.GetInputConstData("input_size", input_sizes_tensor) == GRAPH_SUCCESS) {
    DataType dtype = input_sizes_desc->GetDataType();
    GetConstValue(input_sizes_tensor, dtype, input_sizes);
    is_input_size_const = true;
  }
  if (IsUnKnownShape(dy_sizes)) {
    // when static op or dynamic op phase_running, is_dynamic == False
    is_dynamic = true;
    reset_range(op, "out_backprop");
  }

  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  CHECK_SIZE(strides.size() != DIM_SIZE4, return GRAPH_FAILED, "strides is invalid");

  std::vector<int64_t> dilations;
  dilations = GetAttrValue(op, "dilations");
  CHECK_SIZE(dilations.size() != DIM_SIZE4, return GRAPH_FAILED, "dilations is invalid");

  std::string dataFormat = "";
  CHECK_OP_FUNC(ge::GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat),
                return GRAPH_FAILED, "get data_format attr failed");

  int64_t h_position = 0;
  int64_t w_position = 0;
  int64_t fh_position = 0;
  int64_t fw_position = 0;
  int64_t fc_position = 0;
  int64_t fn_position = 0;
  int64_t in_h = 0;
  int64_t in_w = 0;
  int64_t filter_h = 0;
  int64_t filter_w = 0;
  int64_t dilation_h = 0;
  int64_t dilation_w = 0;
  int64_t stride_h = 0;
  int64_t stride_w = 0;
  int64_t padtop = 0;
  int64_t padbottom = 0;
  int64_t padleft = 0;
  int64_t padright = 0;

  auto tensorDescW = op.GetInputDesc(1);
  auto shapeW = tensorDescW.GetShape();
  Format filterFormat = tensorDescW.GetFormat();
  std::string filterFormatStr = format2str[filterFormat];
  bool filter_format_invalid = !GetDimInFormat(string(op_name.GetString()), filterFormatStr, "H", fh_position) ||
                               !GetDimInFormat(string(op_name.GetString()), filterFormatStr, "W", fw_position) ||
                               !GetDimInFormat(string(op_name.GetString()), filterFormatStr, "C", fc_position) ||
                               !GetDimInFormat(string(op_name.GetString()), filterFormatStr, "N", fn_position);
  if (filter_format_invalid) {
    return GRAPH_FAILED;
  }

  filter_h = shapeW.GetDim(fh_position);
  filter_w = shapeW.GetDim(fw_position);

  bool data_format_invalid = !GetDimInFormat(string(op_name.GetString()), dataFormat, "H", h_position) ||
                             !GetDimInFormat(string(op_name.GetString()), dataFormat, "W", w_position);
  if (data_format_invalid) {
    return GRAPH_FAILED;
  }

  dilation_h = dilations.at(h_position);
  dilation_w = dilations.at(w_position);
  stride_h = strides.at(h_position);
  stride_w = strides.at(w_position);

  bool need_infer_range = is_dynamic || (!is_input_size_const && unknown_rank);
  if (need_infer_range) {
    // get shape for output from input_size
    std::string pad_str;
    std::vector<std::pair<int64_t, int64_t>> dy_range;
    x_desc->GetShapeRange(dy_range);
    std::vector<std::pair<int64_t, int64_t>> dx_range;
    input_sizes_desc->GetValueRange(dx_range);
    if (filterFormatStr == "HWCN") {
      groups = 1;
    } else {
      groups = tensorDescW.GetOriginShape().GetDim(fn_position);
    }
    vector<int64_t> attr_params = {stride_h, stride_w, dilation_h, dilation_w};
    if (!set_conv2d_backprop_input_out_shape_range(op, pad_str, dy_sizes, dy_format, dy_range, filter_sizes,
                                                   filter_format, input_format, dx_range, y_desc, x_desc,
                                                   groups, unknown_rank, attr_params)) {
      return GRAPH_FAILED;
    }
    if (input_sizes.size() == 0) {
      for (size_t i = 0; i < dx_range.size(); i++) {
        input_sizes.push_back(dx_range[i].first == dx_range[i].second ? dx_range[i].first : -1);
      }
    }

    if (!unknown_rank) {
      reset_conv2d_backprop_input_out_shape(op, dy_sizes, dy_format, input_sizes, input_format);
    }
  }

  if (!is_dynamic && is_input_size_const) {
    // NC1HWC0(NCHW)
    in_h = input_sizes[h_position];
    in_w = input_sizes[w_position];
    CHECK_OP_FUNC(false == GetPadDepthwiseConv2D(op, in_h, in_w, filter_h, filter_w, stride_h, stride_w, dilation_h,
                  dilation_w, padtop, padbottom, padleft, padright), return GRAPH_FAILED, "update pads attrs failed.");
    vector<int64_t> attr_param = {stride_h, stride_w, dilation_h, dilation_w};
    bool check_pads_fail = !unknown_rank && !IsUnKnownShape(dy_sizes) &&
                           !check_conv2d_backprop_input_pads(op, dy_sizes, dy_format, filter_sizes, filter_format,
                                                             input_sizes, input_format, attr_param);
    if (check_pads_fail) {
      return GRAPH_FAILED;
    }
  }
  if (input_sizes.size() == kConv2dDimSizeLimit) {
    y_desc->SetShape(GeShape(input_sizes));
  }
  if (filterFormatStr == "HWCN") {
    groups = tensorDescW.GetOriginShape().GetDim(fc_position);
  } else {
    groups = tensorDescW.GetOriginShape().GetDim(fn_position);
  }
  op.SetAttr("groups", groups);
  OP_LOGI(op_name.GetString(), "Leaving DepthwiseConv2dBackpropInput inferfunction!");

  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DepthwiseConv2DBackpropInput, DepthwiseConv2DBackpropInputInferShape);
// Registered verify function
VERIFY_FUNC_REG(DepthwiseConv2DBackpropInput, DepthwiseConv2DBackpropInputVerify);

// ----------------DepthwiseConv2DBackpropFilterD Op-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(DepthwiseConv2DBackpropFilterD, DepthwiseConv2DBackpropFilterDVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  std::vector<int64_t> filter_size;
  filter_size = GetAttrValue(op, "filter_size");
  if (!CheckListEmpty(string(op_name.GetString()), filter_size, "filter_size")) {
    return GRAPH_FAILED;
  }
  if (filter_size.size() != DIM_SIZE4) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "Filter_size's dim must be 4!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(string(op_name.GetString()), strides, "strides")) {
    return GRAPH_FAILED;
  }
  if (strides.size() != DIM_SIZE4) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "strides must be 4!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dilations;
  dilations = GetAttrValue(op, "dilations");
  if (!CheckListEmpty(string(op_name.GetString()), dilations, "dilations")) {
    return GRAPH_FAILED;
  }
  if (dilations.size() != DIM_SIZE4) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "dilations must be 4!");
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format != "NCHW" && data_format != "NHWC") {
      CUBE_INNER_ERR_REPORT(op_name.GetString(),
                            "attr data_format(%s) only support NCHW and NHWC", data_format.c_str());
      return GRAPH_FAILED;
    }
  }
  if (GRAPH_SUCCESS != VerifyDepthwiseConv2DbpPads(op)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "get pads failed.");
    return GRAPH_FAILED;
  }

  if (op.GetInputDesc(0).GetDataType() != op.GetInputDesc(1).GetDataType()) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "The type of input and out_backprop must be same!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DepthwiseConv2DBackpropFilterDInferShape) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  std::vector<int64_t> filter_size;
  filter_size = GetAttrValue(op, "filter_size");

  DataType output_dtype = op.GetInputDescByName("out_backprop").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDescByName("filter_grad");
  tensordesc_output.SetShape(Shape(filter_size));
  tensordesc_output.SetDataType(output_dtype);
  (void)op.UpdateOutputDesc("filter_grad", tensordesc_output);

  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(string(op_name.GetString()), strides, "strides")) {
    return GRAPH_FAILED;
  }
  if (strides.size() != DIM_SIZE4) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "strides must be NCHW!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dilations;
  dilations = GetAttrValue(op, "dilations");
  if (!CheckListEmpty(string(op_name.GetString()), dilations, "dilations")) {
    return GRAPH_FAILED;
  }
  if (dilations.size() != DIM_SIZE4) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "dilations must be NCHW!");
    return GRAPH_FAILED;
  }

  std::string dataFormat = "";
  CHECK_OP_FUNC(ge::GRAPH_SUCCESS != op.GetAttr("data_format", dataFormat),
                return GRAPH_FAILED, "get data_format attr failed");

  int64_t hPosition = 0;
  int64_t wPosition = 0;
  int64_t fhPosition = 0;
  int64_t fwPosition = 0;
  int64_t cPosition = 0;
  int64_t inH = 0;
  int64_t inW = 0;
  int64_t filterH = 0;
  int64_t filterW = 0;
  int64_t dilationH = 0;
  int64_t dilationW = 0;
  int64_t strideH = 0;
  int64_t strideW = 0;
  int64_t padtop = 0;
  int64_t padbottom = 0;
  int64_t padleft = 0;
  int64_t padright = 0;

  Format filterFormat = tensordesc_output.GetFormat();
  std::string filterFormatStr = format2str[filterFormat];
  if (!GetDimInFormat(string(op_name.GetString()), filterFormatStr, "H", fhPosition)) {
    return GRAPH_FAILED;
  }
  if (!GetDimInFormat(string(op_name.GetString()), filterFormatStr, "W", fwPosition)) {
    return GRAPH_FAILED;
  }
  filterH = filter_size[fhPosition];
  filterW = filter_size[fwPosition];

  auto tensorDescIn = op.GetInputDesc(0);
  auto shapeIn = tensorDescIn.GetShape();

  if (!GetDimInFormat(string(op_name.GetString()), dataFormat, "H", hPosition)) {
    return GRAPH_FAILED;
  }
  if (!GetDimInFormat(string(op_name.GetString()), dataFormat, "W", wPosition)) {
    return GRAPH_FAILED;
  }
  if (!GetDimInFormat(string(op_name.GetString()), dataFormat, "C", cPosition)) {
    return GRAPH_FAILED;
  }
  int64_t groups = 0;
  groups = shapeIn.GetDim(cPosition);
  op.SetAttr("groups", groups);

  // NC1HWC0(NCHW)
  inH = shapeIn.GetDim(hPosition);
  inW = shapeIn.GetDim(wPosition);

  dilationH = dilations.at(hPosition);
  dilationW = dilations.at(wPosition);
  strideH = strides.at(hPosition);
  strideW = strides.at(wPosition);

  CHECK_OP_FUNC(false == GetPadDepthwiseConv2D(op, inH, inW, filterH, filterW, strideH, strideW,
                dilationH, dilationW, padtop, padbottom, padleft, padright),
                return GRAPH_FAILED, "update pads attrs failed.");

  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DepthwiseConv2DBackpropFilterD, DepthwiseConv2DBackpropFilterDInferShape);
// Registered verify function
VERIFY_FUNC_REG(DepthwiseConv2DBackpropFilterD, DepthwiseConv2DBackpropFilterDVerify);

// ----------------DepthwiseConv2DBackpropFilter Op-------------------
// Check the dtype and attr of the input tensor description.
IMPLEMT_VERIFIER(DepthwiseConv2DBackpropFilter, DepthwiseConv2DBackpropFilterVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  Tensor filter_size_tensor;
  std::vector<int64_t> filter_size;

  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  if (!CheckListEmpty(string(op_name.GetString()), strides, "strides")) {
    return GRAPH_FAILED;
  }
  if (strides.size() != DIM_SIZE4) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "strides must be 4!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dilations;
  dilations = GetAttrValue(op, "dilations");
  if (!CheckListEmpty(string(op_name.GetString()), dilations, "dilations")) {
    return GRAPH_FAILED;
  }
  if (dilations.size() != DIM_SIZE4) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "dilations must be 4!");
    return GRAPH_FAILED;
  }
  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format != "NCHW" && data_format != "NHWC") {
      CUBE_INNER_ERR_REPORT(op_name.GetString(), "attr data_format(%s) only support NCHW and NHWC",
        data_format.c_str());
      return GRAPH_FAILED;
    }
  }
  if (VerifyDepthwiseConv2DbpPadding(op) != GRAPH_SUCCESS && VerifyDepthwiseConv2DbpPads(op) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "get padding and pads both failed.");
    return GRAPH_FAILED;
  }

  // input index is 0, out_backprop index is 2
  CHECK_OP_FUNC(op.GetInputDesc(0).GetDataType() != op.GetInputDesc(2).GetDataType(), return GRAPH_FAILED,
                "The type of input and out_backprop must be same!");
  return GRAPH_SUCCESS;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DepthwiseConv2DBackpropFilterInferShape) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter DepthwiseConv2DBackpropFilter inferfunction!");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);
  std::vector<std::string> input_infer_depends = {"filter_size"};
  op_desc->SetOpInferDepends(input_infer_depends);

  Tensor filter_sizes_tensor;
  auto tensordesc_output = op_desc->MutableOutputDesc("filter_grad");
  auto input_desc = op_desc->MutableInputDesc("input");
  auto out_backprop_desc = op_desc->MutableInputDesc("out_backprop");
  Format filter_format = tensordesc_output->GetFormat();
  Format input_format = input_desc->GetFormat();
  Format out_backprop_format = out_backprop_desc->GetFormat();

  std::vector<int64_t> input_sizes = input_desc->MutableShape().GetDims();
  std::vector<int64_t> out_backprop_sizes = out_backprop_desc->MutableShape().GetDims();
  bool unknown_rank = IsUnknownRankShape(input_sizes);
  CHECK_KEY_IN_MAP(format2str, filter_format, "filter_format", return GRAPH_FAILED);
  std::string filter_format_str = format2str[filter_format];
  CHECK_KEY_IN_MAP(format2str, input_format, "input_format", return GRAPH_FAILED);
  std::string input_format_str = format2str[input_format];
  CHECK_KEY_IN_MAP(format2str, out_backprop_format, "out_backprop_format", return GRAPH_FAILED);
  std::string out_backprop_format_str = format2str[out_backprop_format];

  size_t input_c_position = input_format_str.find("C");
  size_t out_backprop_c_position = out_backprop_format_str.find("C");
  size_t filter_grad_n_position = filter_format_str.find("N");
  size_t filter_grad_c_position = filter_format_str.find("C");

  std::vector<int64_t> filter_size;
  bool is_filter_size_const = false;

  if (GRAPH_SUCCESS == op.GetInputConstData("filter_size", filter_sizes_tensor)) {
    is_filter_size_const = true;
    DataType dtype = op_desc->MutableInputDesc("filter_size")->GetDataType();
    GetConstValue(filter_sizes_tensor, dtype, filter_size);
  } else {
    filter_size.push_back(-1);
    filter_size.push_back(-1);
    filter_size.push_back(-1);
    filter_size.push_back(-1);
    if (!unknown_rank) {
      filter_size[filter_grad_c_position] = input_sizes[input_c_position];
      filter_size[filter_grad_n_position] =
            out_backprop_sizes[out_backprop_c_position] / input_sizes[input_c_position];
    }
  }

  tensordesc_output->SetShape(GeShape(filter_size));
  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  CHECK_SIZE(strides.size() != DIM_SIZE4, return GRAPH_FAILED, "strides is invalid");

  std::vector<int64_t> dilations;
  dilations = GetAttrValue(op, "dilations");
  CHECK_SIZE(dilations.size() != DIM_SIZE4, return GRAPH_FAILED, "dilations is invalid");

  std::string data_format = "";
  CHECK_OP_FUNC(ge::GRAPH_SUCCESS != op.GetAttr("data_format", data_format),
                return GRAPH_FAILED, "get data_format attr failed");

  int64_t h_position = 0;
  int64_t w_position = 0;
  int64_t fh_position = 0;
  int64_t fw_position = 0;
  int64_t fc_position = 0;
  int64_t fn_position = 0;
  int64_t in_h = 0;
  int64_t in_w = 0;
  int64_t filter_h = 0;
  int64_t filter_w = 0;
  int64_t filter_c = 0;
  int64_t filter_n = 0;
  int64_t dilation_h = 0;
  int64_t dilation_w = 0;
  int64_t stride_h = 0;
  int64_t stride_w = 0;
  int64_t padtop = 0;
  int64_t padbottom = 0;
  int64_t padleft = 0;
  int64_t padright = 0;

  bool filter_format_invalid = !GetDimInFormat(string(op_name.GetString()), filter_format_str, "H", fh_position) ||
                               !GetDimInFormat(string(op_name.GetString()), filter_format_str, "W", fw_position) ||
                               !GetDimInFormat(string(op_name.GetString()), filter_format_str, "C", fc_position) ||
                               !GetDimInFormat(string(op_name.GetString()), filter_format_str, "N", fn_position);
  if (filter_format_invalid) {
    return GRAPH_FAILED;
  }
  filter_h = filter_size[fh_position];
  filter_w = filter_size[fw_position];
  filter_c = filter_size[fc_position];
  filter_n = filter_size[fn_position];
  int64_t groups = 0;
  if (filter_format_str == "HWCN") {
    groups = filter_c;
  } else {
    groups = filter_n;
  }
  op.SetAttr("groups", groups);

  auto tensorDescIn = op.GetInputDesc(0);
  auto shapeIn = tensorDescIn.GetShape();

  bool data_format_invalid = !GetDimInFormat(string(op_name.GetString()), data_format, "H", h_position) ||
                             !GetDimInFormat(string(op_name.GetString()), data_format, "W", w_position);
  if (data_format_invalid) {
    return GRAPH_FAILED;
  }

  dilation_h = dilations.at(h_position);
  dilation_w = dilations.at(w_position);
  stride_h = strides.at(h_position);
  stride_w = strides.at(w_position);

  bool is_dynamic;
  is_dynamic = !is_filter_size_const || unknown_rank || IsUnKnownShape(input_sizes);
  SetInputConst(is_filter_size_const, IsUnKnownShape(input_sizes), unknown_rank, op_desc);
  if (!is_dynamic) {
    // NC1HWC0(NCHW)
    in_h = shapeIn.GetDim(h_position);
    in_w = shapeIn.GetDim(w_position);
    CHECK_OP_FUNC(false == GetPadDepthwiseConv2D(op, in_h, in_w, filter_h, filter_w, stride_h, stride_w, dilation_h,
                  dilation_w, padtop, padbottom, padleft, padright), return GRAPH_FAILED, "update pads attrs failed.");
    vector<int64_t> attr_param = {stride_h, stride_w, dilation_h, dilation_w};
    if (!check_conv2d_backprop_input_pads(op, out_backprop_sizes, out_backprop_format, filter_size, filter_format,
                                          input_sizes, input_format, attr_param)) {
      return GRAPH_FAILED;
    }
  } else {
    // update pads list by padding[SAME,VALID]
    reset_range(op, "input");
    reset_range(op, "out_backprop");
    std::string pad_str;
    if (GRAPH_SUCCESS == op.GetAttr("padding", pad_str)) {
      if (pad_str == "SAME") {
        op.SetAttr("pads", {-1, -1, -1, -1});
        OP_LOGD(op_name.GetString(), "set pads to {-1, -1, -1, -1} when padding is SAME in dynamic_shape");
      } else if (pad_str == "VALID") {
        op.SetAttr("pads", {0, 0, 0, 0});
        OP_LOGD(op_name.GetString(), "set pads to {0, 0, 0, 0} when padding is VALID in dynamic_shape");
      }
    }
  }
  OP_LOGD(op_name.GetString(), "Leaving DepthwiseConv2DBackpropFilter inferfunction!");
  return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DepthwiseConv2DBackpropFilter, DepthwiseConv2DBackpropFilterInferShape);
// Registered verify function
VERIFY_FUNC_REG(DepthwiseConv2DBackpropFilter, DepthwiseConv2DBackpropFilterVerify);

// --------------------------------BiasAddGrad---------------------------------
IMPLEMT_COMMON_INFERFUNC(BiasAddGradInferShape) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  static const int64_t x_input_idx = 0;
  static const int64_t y_output_idx = 0;
  auto input_desc = op_info->MutableInputDesc(x_input_idx);
  auto output_desc = op_info->MutableOutputDesc(y_output_idx);
  const ge::GeShape& input_shape = input_desc->MutableShape();
  ge::GeShape& output_shape = output_desc->MutableShape();
  DataType input_dtype = input_desc->GetDataType();
  output_desc->SetDataType(input_dtype);
  output_shape.SetDimNum(1);

  if (input_shape.IsUnknownDimNum()) {
    output_shape.SetDim(0, -1);
    std::vector<std::pair<int64_t, int64_t>> output_range{{0, -1}};
    output_desc->SetShapeRange(output_range);
    return GRAPH_SUCCESS;
  }
  std::string data_format;
  if (ge::GRAPH_SUCCESS != op.GetAttr("data_format", data_format)) {
    std::string err_msg = GetInputInvalidErrMsg("data_format");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  size_t dim_num = input_shape.GetDimNum();
  if (data_format == "NHWC") {
    if (dim_num < DIM_SIZE2 || dim_num > DIM_SIZE8) {
      string err_msg1 = ConcatString("The bias add grad op dimension(", dim_num,
                                     ") is not supported when format is NHWC!");
      std::string err_msg = OtherErrMsg(err_msg1);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
    output_shape.SetDim(0, input_shape.GetDim(dim_num - 1));
  } else if (data_format == "NCHW") {
    if (dim_num < DIM_SIZE2) {
      string err_msg1 = ConcatString("The bias add grad op dimension(", dim_num,
                                      ") is not supported when format is NCHW!");
      std::string err_msg = OtherErrMsg(err_msg1);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
    output_shape.SetDim(0, input_shape.GetDim(1));
  } else {
    string expected_format_list = ConcatString("NHWC, NCHW");
    std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format.c_str());
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  if (input_shape.IsUnknownShape()) {
    // update range
    std::vector<std::pair<int64_t, int64_t>> input_range;
    std::vector<std::pair<int64_t, int64_t>> output_range;
    input_desc->GetShapeRange(input_range);
    MakeUpShapeRange(input_shape.GetDims(), input_range);
    if (data_format == "NHWC") {
      output_range.push_back(input_range[dim_num - 1]);
    } else if (data_format == "NCHW") {
      output_range.push_back(input_range[1]);
    }

    output_desc->SetShapeRange(output_range);
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(BiasAddGrad, BiasAddGradInferShape);
// ------------------------------BiasAddGrad END-----------------------------------

// ============================Conv2Dbackprop===============================
static inline int32_t align_conv2dbp(int32_t x1, int32_t x2) {
  return ((x1 + x2 - 1) / x2) * x2;
}

static bool getStrideDilationHW(ge::Operator& op, int32_t& stride_h, int32_t& stride_w, int32_t& dilation_h,
                                int32_t& dilation_w) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  std::string dataFormat = "";
  int32_t hPosition = 0;
  int32_t wPosition = 0;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", dataFormat)) {
    if (dataFormat != "NCHW" && dataFormat != "NHWC") {
      OP_LOGE(op_name.GetString(), "conv2DBackprop's dataFormat error, should be HCHW OR NHWC!");
      map<string, string> err_map;
      err_map["param"] = "data_Format";
      err_map["op_name"] = "conv2Dbp";
      err_map["expected_format_list"] = "HCHW OR NHWC";
      err_map["format"] = dataFormat;
      std::string report_error_code = "E50002";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }
  } else {
    OP_LOGE(op_name.GetString(), "GetOpAttr dataFormat failed!");
    map<string, string> err_map;
    err_map["op_name"] = "conv2Dbp";
    err_map["param_name"] = "data_Format";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  // get H & W position
  hPosition = dataFormat.find("H");
  wPosition = dataFormat.find("W");

  std::vector<int32_t> strideList;
  if (GRAPH_SUCCESS == op.GetAttr("strides", strideList)) {
    if (strideList.empty()) {
      OP_LOGE(op_name.GetString(), "strideList from op is empty!");
      map<string, string> err_map;
      err_map["op_name"] = "conv2Dbp";
      err_map["param_name"] = "stride_List";
      std::string report_error_code = "E50030";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }
    stride_h = strideList[hPosition];
    stride_w = strideList[wPosition];
  } else {
    OP_LOGE(op_name.GetString(), "Get strides list failed!");
    map<string, string> err_map;
    err_map["op_name"] = "conv2Dbp";
    err_map["param_name"] = "stride_List";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  // check dilations shape
  std::vector<int32_t> dilationsList;
  if (GRAPH_SUCCESS == op.GetAttr("dilations", dilationsList)) {
    if (dilationsList.size() != kConv2dDimSizeLimit) {
      OP_LOGE(op_name.GetString(), "dilationsList list should be 4d");
      map<string, string> err_map;
      err_map["op_name"] = "Conv2DBackpropInput";
      err_map["param_name"] = "dilationsList";
      err_map["expected_length"] = std::to_string(kConv2dDimSizeLimit);
      err_map["length"] = std::to_string(dilationsList.size());
      std::string report_error_code = "E50035";
      (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }
    dilation_h = dilationsList[hPosition];
    dilation_w = dilationsList[wPosition];
  } else {
    OP_LOGE(op_name.GetString(), "get dilations list failed.");
    map<string, string> err_map;
    err_map["op_name"] = "Conv2DBackpropInput";
    err_map["param_name"] = "dilations";
    std::string report_error_code = "E50030";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  return true;
}

template <typename T1, typename T2>
static bool SetPadListByPaddingConv2dbp(ge::Operator& op, std::vector<T1>& inputSizes, Format inputFormat,
                                        std::vector<T2>& filterSizes, Format filterFormat) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  OP_LOGI(op_name.GetString(), "SetPadListByPaddingConv2dbp begin.");
  CHECK_OP_FUNC(FORMAT_RESERVED == inputFormat, return false, "get format failed: %d", inputFormat);
  CHECK_OP_FUNC(FORMAT_RESERVED == filterFormat, return false, "get format failed: %d", filterFormat);

  int32_t stride_h = 0;
  int32_t stride_w = 0;
  int32_t dilation_h = 0;
  int32_t dilation_w = 0;
  if (!getStrideDilationHW(op, stride_h, stride_w, dilation_h, dilation_w)) {
    OP_LOGE(op_name.GetString(), "op get strides failed.");
    map<string, string> err_map;
    err_map["op_name"] = "conv2Dbp";
    err_map["param_name"] = "stridet";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  CHECK_KEY_IN_MAP(format2str, inputFormat, "inputFormat", return false);
  std::string inputFormatStr = format2str[inputFormat];
  int32_t hInputPosition = inputFormatStr.find("H");
  CHECK_OP_FUNC(hInputPosition < 0, return false, "get position failed: %d", hInputPosition);
  int32_t wInputPosition = inputFormatStr.find("W");
  CHECK_OP_FUNC(wInputPosition < 0, return false, "get position failed: %d", wInputPosition);
  int32_t dx_h = inputSizes[hInputPosition];
  int32_t dx_w = inputSizes[wInputPosition];
  CHECK_KEY_IN_MAP(format2str, filterFormat, "filterFormat", return false);
  std::string filterFormatStr = format2str[filterFormat];
  int32_t hFilterPosition = filterFormatStr.find("H");
  CHECK_OP_FUNC(hFilterPosition < 0, return false, "get position failed: %d", hFilterPosition);
  int32_t wFilterPosition = filterFormatStr.find("W");
  CHECK_OP_FUNC(wFilterPosition < 0, return false, "get position failed: %d", wFilterPosition);
  int32_t filter_h = filterSizes[hFilterPosition];
  int32_t filter_w = filterSizes[wFilterPosition];

  int32_t filter_dilation_h = (filter_h - 1) * dilation_h + 1;
  int32_t filter_dilation_w = (filter_w - 1) * dilation_w + 1;
  std::string padding;
  std::vector<int32_t> pads;
  if (GRAPH_SUCCESS == op.GetAttr("padding", padding)) {
    int pad_h = 0;
    int32_t pad_up = 0;
    int32_t pad_down = 0;
    int pad_w = 0;
    int32_t pad_left = 0;
    int32_t pad_right = 0;
    if (padding == "SAME") {
      pad_h = std::max(align_conv2dbp(dx_h, stride_h) - stride_h + filter_dilation_h - dx_h, 0);
      pad_up = (pad_h >> 1);
      pad_down = pad_h - pad_up;
      pad_w = std::max(align_conv2dbp(dx_w, stride_w) - stride_w + filter_dilation_w - dx_w, 0);
      pad_left = (pad_w >> 1);
      pad_right = pad_w - pad_left;
    }
    pads.push_back(pad_up);
    pads.push_back(pad_down);
    pads.push_back(pad_left);
    pads.push_back(pad_right);

    op.SetAttr("pads", pads);
  }

  if (GRAPH_SUCCESS == op.GetAttr("pads", pads)) {
    if (pads.size() < kConv2dPadSizeLimit) {
      OP_LOGE(op_name.GetString(), "op pads's size is illegal.");
      map<string, string> err_map;
      err_map["op_name"] = "conv2Dbp";
      err_map["param_name"] = "pads";
      err_map["expected_length"] = "4";
      err_map["length"] = std::to_string(pads.size());
      std::string report_error_code = "E50035";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }
    if (std::any_of(pads.begin(), pads.end(), [](int32_t val){ return val < 0;})) {
      OP_LOGE(op_name.GetString(), "op get pads is illegal");
      map<string, string> err_map;
      err_map["op_name"] = "conv2Dbp";
      err_map["param_name"] = "pads";
      err_map["expected_value"] = ">= 0";
      err_map["input_value"] = ops::to_string(pads);
      std::string report_error_code = "E50029";
      (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }
  } else {
    OP_LOGE(op_name.GetString(), "op get pads failed.");
    map<string, string> err_map;
    err_map["op_name"] = "conv2Dbp";
    err_map["param_name"] = "pads";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  OP_LOGI(op_name.GetString(), "op set pads succ.");
  return true;
}

template <typename T1, typename T2>
static bool SetGroupsConv(ge::Operator& op, std::vector<T1>& input_sizes, Format input_format,
                          std::vector<T2>& filter_sizes, Format filter_format) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  OP_LOGI(op_name.GetString(), "Setgroups begin.");

  CHECK_OP_FUNC(FORMAT_RESERVED == input_format, return false, "get format failed: %d", input_format);
  CHECK_OP_FUNC(FORMAT_RESERVED == filter_format, return false, "get format failed: %d", filter_format);
  CHECK_KEY_IN_MAP(format2str, input_format, "input_format", return false);
  CHECK_KEY_IN_MAP(format2str, filter_format, "filter_format", return false);
  std::string input_format_str = format2str[input_format];
  std::string filter_format_str = format2str[filter_format];

  size_t format_len = input_format_str.length();
  if (filter_sizes.size() < format_len || input_sizes.size() < format_len) {
    OP_LOGE(op_name.GetString(), "filter_sizes or input_sizes is illegal");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["param_name"] = "filter_sizes and input_sizes";
    err_map["expected_length"] = std::to_string(format_len);
    err_map["length"] = std::to_string(filter_sizes.size()) + " and " + std::to_string(input_sizes.size());
    std::string report_error_code = "E50035";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  int32_t x_position = input_format_str.find("C");
  int32_t w_position = filter_format_str.find("C");
  int32_t x_c = input_sizes[x_position];
  int32_t w_c = filter_sizes[w_position];
  int32_t groups = 1;
  if (w_c == 0) {
    OP_LOGE(op_name.GetString(), "channel of filter can not be 0.");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "channel of filter can not be 0.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  } else if (x_c % w_c != 0) {
    OP_LOGE(op_name.GetString(), "fmap_channel %% filter_channel != 0");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "fmap_channel % filter_channel != 0";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  } else {
    groups = x_c / w_c;
  }
  int32_t groups_ori = 1;
  op.GetAttr("groups", groups_ori);
  if (groups_ori == 1) {
    op.SetAttr("groups", groups);
    OP_LOGI(op_name.GetString(), "op set groups succ.");
    return true;
  } else if (groups_ori != groups) {
    OP_LOGE(op_name.GetString(), "fmap_channel / filter_channel != groups");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "fmap_channel / filter_channel != groups";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  } else {
    return true;
  }
}

static bool VerifyConvPadding(const AscendString& op_name, const OpDescPtr& op_desc) {
  std::string padding;
  if (!AttrUtils::GetStr(op_desc, "padding", padding)) {
    OP_LOGW(op_name.GetString(), "get padding failed. try to get pads.");
    return false;
  }
  if (padding.compare("SAME") != 0 && padding.compare("VALID") != 0) {
    OP_LOGE(op_name.GetString(), "padding must be SAME or VALID.");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["expected_pad_mode"] = "SAME or VALID";
    err_map["actual_pad_mode"] = padding;
    std::string report_error_code = "E50050";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  return true;
}

static bool VerifyConv2dbpPads(const AscendString& op_name, const OpDescPtr& op_desc) {
  std::vector<int32_t> pads;
  if (!AttrUtils::GetListInt(op_desc, "pads", pads)) {
    OP_LOGW(op_name.GetString(), "op get pads failed.");
    return false;
  }
  if (pads.size() != kConv2dPadSizeLimit) {
    OP_LOGE(op_name.GetString(), "op pads's size is illegal.");
    map<string, string> err_map;
    err_map["op_name"] = "conv2Dbp";
    err_map["param_name"] = "pads";
    err_map["expected_length"] = "4";
    err_map["length"] = std::to_string(pads.size());
    std::string report_error_code = "E50035";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  if (std::any_of(pads.begin(), pads.end(), [](int32_t val){ return val < 0;})) {
    OP_LOGE(op_name.GetString(), "op get pads is illegal");
    map<string, string> err_map;
    err_map["op_name"] = "conv2Dbp";
    err_map["param_name"] = "pads";
    err_map["expected_value"] = ">= 0";
    err_map["input_value"] = ops::to_string(pads);
    std::string report_error_code = "E50029";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  return true;
}

static bool VerifyConv2dbpStrides(const AscendString& op_name, const OpDescPtr& op_desc) {
  std::vector<int32_t> stride_list;
  if (!AttrUtils::GetListInt(op_desc, "strides", stride_list)) {
    OP_LOGE(op_name.GetString(), "get strides list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "conv2Dbp";
    err_map["param_name"] = "strides list";
    std::string report_error_code = "E50030";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  if (stride_list.size() != kConv2dDimSizeLimit) {
    OP_LOGE(op_name.GetString(), "strides should be 4d.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "conv2Dbp";
    err_map["param_name"] = "strides's shape";
    err_map["expected_length"] = std::to_string(kConv2dDimSizeLimit);
    err_map["length"] = std::to_string(stride_list.size());
    std::string report_error_code = "E50035";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  return true;
}

static bool VerifyConv2dbpDalitions(const AscendString& op_name, const OpDescPtr& op_desc) {
  std::vector<int32_t> dilations_list;
  if (!AttrUtils::GetListInt(op_desc, "dilations", dilations_list)) {
    OP_LOGE(op_name.GetString(), "get dilations list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "conv2Dbp";
    err_map["param_name"] = "dilations";
    std::string report_error_code = "E50030";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  if (dilations_list.size() != kConv2dDimSizeLimit) {
    OP_LOGE(op_name.GetString(), "dilations should be 4d.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "conv2Dbp";
    err_map["param_name"] = "dilations_list";
    err_map["expected_length"] = std::to_string(kConv2dDimSizeLimit);
    err_map["length"] = std::to_string(dilations_list.size());
    std::string report_error_code = "E50035";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  return true;
}

static bool VerifyConv2dbpDataFormat(const AscendString& op_name, const OpDescPtr& op_desc) {
  std::string dataFormat = "";
  if (!AttrUtils::GetStr(op_desc, "data_format", dataFormat)) {
    OP_LOGE(op_name.GetString(), "GetOpAttr dataFormat failed!");
    map<string, string> err_map;
    err_map["op_name"] = "conv2Dbp";
    err_map["param_name"] = "data_Format";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  if (dataFormat != "NCHW" && dataFormat != "NHWC") {
    OP_LOGE(op_name.GetString(), "dataFormat of conv2Dbp is error, can only be HCHW OR NHWC!");
    map<string, string> err_map;
    err_map["param"] = "data_Format";
    err_map["op_name"] = "conv2Dbp";
    err_map["expected_format_list"] = "HCHW OR NHWC";
    err_map["format"] = dataFormat;
    std::string report_error_code = "E50002";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  return true;
}
static bool VerifyConv2dbpParas(const AscendString& op_name, ConstGeTensorDescPtr& input1_desc,
                                ConstGeTensorDescPtr& input2_desc) {
  const GeShape& input1_shape = input1_desc->GetShape();
  const GeShape& input2_shape = input2_desc->GetShape();

  // check input tensor shape
  if (!input1_shape.IsUnknownDimNum() && input1_shape.GetDimNum() != kConv2dDimSizeLimit) {
    OP_LOGE(op_name.GetString(), "filter or x's shape should be 4d.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "conv2Dbp";
    err_map["param_name"] = "filter or x Shape";
    err_map["expected_length"] = std::to_string(kConv2dDimSizeLimit);
    err_map["length"] = std::to_string(input1_shape.GetDimNum());
    std::string report_error_code = "E50035";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (!input2_shape.IsUnknownDimNum() && input2_shape.GetDimNum() != kConv2dDimSizeLimit) {
    OP_LOGE(op_name.GetString(), "out_backprop or x's shape should be 4d.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "conv2Dbp";
    err_map["param_name"] = "out_backprop or x shape";
    err_map["expected_length"] = std::to_string(kConv2dDimSizeLimit);
    err_map["length"] = std::to_string(input2_shape.GetDimNum());
    std::string report_error_code = "E50035";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  return true;
}

// -----------------------------conv3d_transpose_common_check----------------------------
template <typename T1>
static bool CheckVectorAllZero(const std::vector<T1>& list)
{
    for (const auto& iter : list) {
        if (iter != 0) {
            return false;
        }
    }
    return true;
}

static bool VerifyConv2dbpOutputPadding(const AscendString& op_name, const OpDescPtr& op_desc) {
  std::vector<int32_t> output_padding_list;
  if (!AttrUtils::GetListInt(op_desc, "output_padding", output_padding_list)) {
    OP_LOGE(op_name.GetString(), "get output_padding list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTranspose";
    err_map["param_name"] = "output_padding";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (output_padding_list.size() != kConv2dInputSizeLimit) {
    OP_LOGE(op_name.GetString(), "output_paddingList list should be 4d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "output_padding";
    err_map["op_name"] = "Conv2DTranspose";
    err_map["excepted_value"] = std::to_string(kConv2dInputSizeLimit);
    err_map["input_value"] = std::to_string(output_padding_list.size());
    return false;
  }
  return true;
}

// ----------------Conv2DBackpropInput-------------------
static graphStatus VerifyConv2dbpCommon(const AscendString& op_name, const OpDescPtr& op_desc,
                                        ConstGeTensorDescPtr& input1_desc, ConstGeTensorDescPtr& input2_desc) {
  // check paras: filter and input tensor
  if (!VerifyConv2dbpParas(op_name, input1_desc, input2_desc)) {
    return GRAPH_FAILED;
  }
  // check format
  if (!VerifyConv2dbpDataFormat(op_name, op_desc)) {
    return GRAPH_FAILED;
  }
  // check strides shape
  if (!VerifyConv2dbpStrides(op_name, op_desc)) {
    return GRAPH_FAILED;
  }
  // check dilations shape
  if (!VerifyConv2dbpDalitions(op_name, op_desc)) {
    return GRAPH_FAILED;
  }
  // check padding or pads
  if (!VerifyConvPadding(op_name, op_desc) && !VerifyConv2dbpPads(op_name, op_desc)) {
    OP_LOGE(op_name.GetString(), "padding and pads both check fail!");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "verify pads and verify padding";
    err_map["op_name"] = "Conv2DBackpropInput";
    err_map["excepted_value"] = std::to_string(GRAPH_SUCCESS);
    err_map["output_value"] = std::to_string(GRAPH_FAILED);
    std::string report_error_code = "E50029";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

static void InferHWConv2DbpInput(vector<int32_t>& pads, vector<int64_t>& output, vector<int64_t>& input,
                                 int32_t index, const SliceTuple& infer_hw) {
  int32_t kernel = 0;
  int32_t dilation = 0;
  int32_t stride = 0;
  int32_t input_max = 0;
  std::tie(kernel, dilation, stride, input_max) = infer_hw;
  int32_t dilate_kernel = (kernel - 1) * dilation + 1;
  int32_t pad_out = kernel - pads[index] - 1;
  int32_t start = std::ceil(static_cast<float>(output[0] - pad_out) / stride);
  int32_t end = (output[1] + dilate_kernel - 1 - pad_out) / stride;
  start = std::max(static_cast<int32_t>(0), start);
  end = std::min(end, input_max - 1);
  input = {start, end};
  int32_t oh = output[1] - output[0] + 1;
  int32_t ih = end - start + 1;
  if (output[0] != 0) {
    int32_t pad_pre_deconv = (stride - (output[0] - pad_out) % stride) % stride;
    pads[index] = kernel - pad_pre_deconv - 1;
  }
  pads[index+1] = stride * (ih - 1) + dilate_kernel - oh - pads[index];
}

static graphStatus data_slice_conv2d_backprop_input(ge::Operator& op, vector<vector<int64_t>>& y_data_slice,
                                                    vector<vector<int64_t>>& dedy_data_slice,
                                                    vector<vector<int64_t>>& filter_data_slice,
                                                    vector<int32_t>& y_position, vector<int32_t>& attr_param,
                                                    vector<int32_t>& input_sizes, vector<int32_t>& pad_list,
                                                    ge::GeTensorDescPtr& tensor_desc_filter,
                                                    ge::GeTensorDescPtr& tensor_desc_dedy, size_t& i) {
  size_t id_y_position = 0;
  int32_t n_y_position = y_position[id_y_position++];
  int32_t c_y_position = y_position[id_y_position++];
  int32_t h_y_position = y_position[id_y_position++];
  int32_t w_y_position = y_position[id_y_position++];
  size_t id_attr_param = 0;
  int32_t kh = attr_param[id_attr_param++];
  int32_t kw = attr_param[id_attr_param++];
  int32_t dilh = attr_param[id_attr_param++];
  int32_t dilw = attr_param[id_attr_param++];
  int32_t strh = attr_param[id_attr_param++];
  int32_t strw = attr_param[id_attr_param++];
  int32_t ih = attr_param[id_attr_param++];
  int32_t iw = attr_param[id_attr_param++];
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);
  AscendString op_type;
  CHECK(op.GetOpType(op_type) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);
  int32_t y_extend = y_data_slice[i][1] - y_data_slice[i][0] + 1;
  bool i_equal_two = i == kHDimNCHWIdx && (kh != 1 || strh != 1) && ih > 0;
  bool i_equal_three = i == kWDimNCHWIdx && (kw != 1 || strw != 1) && iw > 0;
  bool is_deconv_flag = string(op_type.GetString()) == "Deconvolution";
  bool not_set_input_size = is_deconv_flag || CheckVectorAllZero(input_sizes);
  SliceTuple infer_hw(kh, dilh, strh, ih);
  if (i == 1) {
    bool dim_shape_fail = string(op_type.GetString()) == "Conv2DBackpropInputD" &&
                          y_data_slice[i].size() < kDataSliceLen;
    if (dim_shape_fail) {
      CUBE_INNER_ERR_REPORT(op_name.GetString(), "Dim of shape failed");
      return GRAPH_FAILED;
    }
    bool dtype_spec = (string(op_type.GetString()) == "Deconvolution" ||
                       string(op_type.GetString()) == "Conv2DTranspose") &&
                       op.GetInputDescByName("x").GetDataType() == DT_INT8;
    if (dtype_spec) {
      filter_data_slice[1] = y_data_slice[i];
    }
    else {
      int64_t cin_start = y_data_slice[i][0] * kh * kw;
      int64_t cin_end = (y_data_slice[i][1] + 1) * kh * kw - 1;
      filter_data_slice[0] = {cin_start, cin_end};
    }
    CHECK_OP_FUNC(!AttrUtils::SetListListInt(tensor_desc_filter, ge::ATTR_NAME_DATA_SLICE, filter_data_slice),
                  return GRAPH_FAILED, "set filter data slice attr failed.");
    if (!not_set_input_size) {
      input_sizes[c_y_position] = y_extend * kBlockBaseSize;
      op.SetAttr("input_size", input_sizes);
    }
    OP_LOGI(op_name.GetString(), "infer input in Cin success");
    return GRAPH_SUCCESS;
  } else if (i_equal_two) {
    vector<int64_t> input_h;
    InferHWConv2DbpInput(pad_list, y_data_slice[i], input_h, 0, infer_hw);
    dedy_data_slice[i] = input_h;
    CHECK_OP_FUNC(!AttrUtils::SetListListInt(tensor_desc_dedy, ge::ATTR_NAME_DATA_SLICE, dedy_data_slice),
              return GRAPH_FAILED, "set dedy data slice attr failed.");
    if (!not_set_input_size) {
      input_sizes[h_y_position] = y_extend;
      op.SetAttr("input_size", input_sizes);
    }
    op.SetAttr("pads", pad_list);
    OP_LOGI(op_name.GetString(), "infer input in H success");
    return GRAPH_SUCCESS;
  } else if (i_equal_three) {
    vector<int64_t> input_w;
    infer_hw = std::make_tuple(kw, dilw, strw, iw);
    InferHWConv2DbpInput(pad_list, y_data_slice[i], input_w, kConv2dPadLeftIdx, infer_hw);
    dedy_data_slice[i] = input_w;
    CHECK_OP_FUNC(!AttrUtils::SetListListInt(tensor_desc_dedy, ge::ATTR_NAME_DATA_SLICE, dedy_data_slice),
              return GRAPH_FAILED, "set dedy data slice attr failed.");
    if (!not_set_input_size) {
      input_sizes[w_y_position] = y_extend;
      op.SetAttr("input_size", input_sizes);
    }
    op.SetAttr("pads", pad_list);
    OP_LOGI(op_name.GetString(), "infer input in W success");
    return GRAPH_SUCCESS;
  } else if (i == kConv2dDimSizeLimit) {
    OP_LOGI(op_name.GetString(), "cannot support cut in block_C");
    return NOT_SUPPORT_SLICE;
  }
  dedy_data_slice[i] = y_data_slice[i];
  CHECK_OP_FUNC(!AttrUtils::SetListListInt(tensor_desc_dedy, ge::ATTR_NAME_DATA_SLICE, dedy_data_slice),
            return GRAPH_FAILED, "set dedy data slice attr failed.");
  if (!not_set_input_size) {
    if (i == kHDimNCHWIdx) {
      input_sizes[h_y_position] = y_extend;
    } else if (i == kWDimNCHWIdx) {
      input_sizes[w_y_position] = y_extend;
    } else {
      input_sizes[n_y_position] = y_extend;
    }
    op.SetAttr("input_size", input_sizes);
  }
  OP_LOGI(op_name.GetString(), "infer input in N/H/W without overlap success");
  return GRAPH_SUCCESS;
}

// get infer date slice
IMPLEMT_INFER_DATA_SLICE(Conv2DBackpropInput, Conv2DBackpropInputInferDataSlice) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);
  OpDescPtr op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);

  // the filter tensor index is 1
  ConstGeTensorDescPtr filter_tensor_desc = op_desc->GetInputDescPtr(1);
  CHECK_PTR_NULL(filter_tensor_desc, "filter desc", return GRAPH_FAILED);
  // the out_backprop tensor index is 2
  ConstGeTensorDescPtr out_backprop_desc = op_desc->GetInputDescPtr(2);
  CHECK_PTR_NULL(out_backprop_desc, "out_backprop desc", return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter Conv2DBackpropInput InferDataSlice.");
  if (GRAPH_SUCCESS != VerifyConv2dbpCommon(op_name, op_desc, filter_tensor_desc, out_backprop_desc)) {
    return GRAPH_FAILED;
  }

  // get dedy shape, stride and dilation
  auto dedy_desc = op.GetInputDescByName("out_backprop");
  auto dedy_shape = dedy_desc.GetOriginShape().GetDims();
  auto dedy_format = dedy_desc.GetOriginFormat();
  std::vector<int32_t> stride_list;
  op.GetAttr("strides", stride_list);
  std::vector<int32_t> dilation_list;
  op.GetAttr("dilations", dilation_list);
  std::vector<int32_t> input_sizes;
  op.GetAttr("input_size", input_sizes);
  int32_t ih = -1;
  int32_t iw = -1;
  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;
  CHECK_SIZE(dedy_shape.size() != kConv2dDimSizeLimit, return GRAPH_FAILED, "dedy_shape is invalid");
  CHECK_SIZE(stride_list.size() != kConv2dDimSizeLimit, return GRAPH_FAILED, "stride is invalid");
  CHECK_SIZE(dilation_list.size() != kConv2dDimSizeLimit, return GRAPH_FAILED, "dilation is invalid");
  CHECK_SIZE(input_sizes.size() != kConv2dDimSizeLimit, return GRAPH_FAILED, "input sizes is invalid");

  if (dedy_format == FORMAT_NCHW) {
    if (dedy_shape != DYNAMIC_DIM_ALL) {
      ih = dedy_shape[kHDimNCHWIdx];
      iw = dedy_shape[kWDimNCHWIdx];
    }
    strh = stride_list[kHDimNCHWIdx];
    strw = stride_list[kWDimNCHWIdx];
    dilh = dilation_list[kHDimNCHWIdx];
    dilw = dilation_list[kWDimNCHWIdx];
  } else if (dedy_format == FORMAT_NHWC) {
    if (dedy_shape != DYNAMIC_DIM_ALL) {
      ih = dedy_shape[kHDimNHWCIdx];
      iw = dedy_shape[kWDimNHWCIdx];
    }
    strh = stride_list[kHDimNHWCIdx];
    strw = stride_list[kWDimNHWCIdx];
    dilh = dilation_list[kHDimNHWCIdx];
    dilw = dilation_list[kWDimNHWCIdx];
  }
  bool stride_invalid = (strh <= 0) || (strw <= 0);
  if (stride_invalid) {
    OP_LOGE(op_name.GetString(), "stride can not less than zero");
    ErrorManager::GetInstance().ATCReportErrMessage("E50029",
                                                    {"op_name", "param_name", "expected_value", "input_value"},
                                                    {op_name.GetString(), "strides", "positive",
                                                    std::to_string(strh) + ", " + std::to_string(strw)});
    return GRAPH_FAILED;
  }

  // get filter shape
  auto filter_desc = op.GetInputDescByName("filter");
  auto filter_shape = filter_desc.GetOriginShape().GetDims();
  auto filter_format = filter_desc.GetOriginFormat();
  int32_t kh = 0;
  int32_t kw = 0;
  if (filter_format == FORMAT_NCHW) {
    kh = filter_shape[kHDimNCHWIdx];
    kw = filter_shape[kWDimNCHWIdx];
  } else if (filter_format == FORMAT_NHWC) {
    kh = filter_shape[kHDimNHWCIdx];
    kw = filter_shape[kWDimNHWCIdx];
  } else if (filter_format == FORMAT_HWCN) {
    kh = filter_shape[0];
    kw = filter_shape[1];
  }

  auto y_desc = op.GetOutputDescByName("y");
  auto y_format = y_desc.GetOriginFormat();
  CHECK_KEY_IN_MAP(format2str, y_format, "y_format", return GRAPH_FAILED);
  std::string y_format_str = format2str[y_format];
  int32_t n_y_position = y_format_str.find("N");
  int32_t c_y_position = y_format_str.find("C");
  int32_t h_y_position = y_format_str.find("H");
  int32_t w_y_position = y_format_str.find("W");

  // get pads
  std::vector<int32_t> pad_list;
  op.GetAttr("pads", pad_list);
  if (pad_list.size() != kConv2dPadSizeLimit) {
    OP_LOGE(op_name.GetString(), "pad is invalid");
    ErrorManager::GetInstance().ATCReportErrMessage("E50058",
                                                    {"op_name", "description"},
                                                    {op_name.GetString(), "pad is invalid"});
    return GRAPH_FAILED;
  }

  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_dedy = op_desc->MutableInputDesc("out_backprop");
  GeTensorDescPtr tensor_desc_filter = op_desc->MutableInputDesc("filter");

  CHECK_PTR_NULL(tensor_desc_y, "tensor y desc", return GRAPH_FAILED);
  CHECK_PTR_NULL(tensor_desc_dedy, "tensor dedy desc", return GRAPH_FAILED);
  CHECK_PTR_NULL(tensor_desc_filter, "tensor filter desc", return GRAPH_FAILED);

  vector<vector<int64_t>> y_data_slice;
  vector<vector<int64_t>> dedy_data_slice = {{}, {}, {}, {}, {}};
  vector<vector<int64_t>> filter_data_slice = {{}, {}, {}, {}};

  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op_name.GetString(), "no data slice, not need infer input");
    return GRAPH_FAILED;
  }

  for (size_t i = 0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() > 1) {
      vector<int32_t> y_position = {n_y_position, c_y_position, h_y_position, w_y_position};
      vector<int32_t> attr_param = {kh, kw, dilh, dilw, strh, strw, ih, iw};
      return data_slice_conv2d_backprop_input(op, y_data_slice, dedy_data_slice, filter_data_slice, y_position,
                                              attr_param, input_sizes, pad_list, tensor_desc_filter,
                                              tensor_desc_dedy, i);
    }
  }

  OP_LOGI(op_name.GetString(), "no data slice, not need infer input");
  return GRAPH_FAILED;
}

static void GetConstValue(const Tensor &const_tensor, const DataType &dtype, GeShape &const_data) {
  const uint8_t *constData = const_tensor.GetData();
  size_t size;
  if (dtype == ge::DT_INT32) {
    size = const_tensor.GetSize() / sizeof(int32_t);
    const_data.SetDimNum(size);
    for (size_t i = 0; i < size; ++i) {
      const_data.SetDim(i, *((int32_t *)constData + i));
    }
  } else {
    size = const_tensor.GetSize() / sizeof(int64_t);
    const_data.SetDimNum(size);
    for (size_t i = 0; i < size; ++i) {
      const_data.SetDim(i, *((int64_t *)constData + i));
    }
  }
}

static bool CheckAndSetInputSizes(const Operator &op, const DataType &dtype, const std::string &tensor_name,
                                  bool &is_input_size_const, GeTensorDescPtr &y_desc) {
  Tensor input_sizes_tensor;
  if (op.GetInputConstData(tensor_name.c_str(), input_sizes_tensor) != GRAPH_SUCCESS) {
    return true;
  }
  GeShape input_sizes;
  GetConstValue(input_sizes_tensor, dtype, input_sizes);
  CHECK_SIZE(input_sizes.GetDimNum() != kConv2dDimSizeLimit, return false, "input_size is unvalid");
  is_input_size_const = true;
  y_desc->SetShape(input_sizes);
  return true;
}

static void SetConv2dBpInputOutRange(const OpDescPtr &op_desc, size_t idx, const AttrDataInfo &attrParams,
                                     const std::vector<std::pair<int64_t, int64_t>> &dy_range,
                                     std::vector<std::pair<int64_t, int64_t>> &dx_range) {
  OP_LOGD("", "start reset out range for -1");
  int64_t low = dy_range[idx].first;
  int64_t high = dy_range[idx].second;
  std::string padding;
  if (AttrUtils::GetStr(op_desc, "padding", padding) && padding == "SAME") {
    dx_range[idx].first = attrParams.stride * (low - 1) + 1;
    dx_range[idx].second = attrParams.stride * high;
  } else {
    dx_range[idx].first = attrParams.stride * (low - 1) + attrParams.kernel - attrParams.pads;
    dx_range[idx].second = attrParams.stride * (high - 1) + attrParams.kernel - attrParams.pads + attrParams.stride - 1;
  }
  dx_range[idx].first = std::max(dx_range[idx].first, kDynamicRangeLowerBound);
  dx_range[idx].second = std::min(dx_range[idx].second, kDynamicRangeUpperBound);
  if (high == -1) {
    dx_range[idx].second = high;
  }
}

static bool GetConv2dBpAttrInfo(const OpDetailInfo &op_info, const PosInfo &y_format_pos,
                                const ShapeValInfo &filter_value, AttrInfo &attr_paras) {
  std::vector<int32_t> strides_list;
  AttrUtils::GetListInt(op_info.op_desc, "strides", strides_list);
  std::vector<int32_t> dilations_list;
  AttrUtils::GetListInt(op_info.op_desc, "dilations", dilations_list);

  map<string, string> err_map;
  int32_t stride_h = strides_list[y_format_pos.h_position];
  int32_t stride_w = strides_list[y_format_pos.w_position];
  if (stride_h <= 0 || stride_w <= 0) {
    OP_LOGE(op_info.op_name.GetString(), "strides should be positive, actual is [%d,%d].", stride_h, stride_w);
    err_map["param_name"] = "strides";
    err_map["op_name"] = op_info.op_name.GetString();
    err_map["expected_value"] = "positive";
    err_map["input_value"] = std::to_string(stride_h) + ", " + std::to_string(stride_w);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  int32_t dilation_h = dilations_list[y_format_pos.h_position];
  int32_t dilation_w = dilations_list[y_format_pos.w_position];
  if (dilation_h <= 0 || dilation_w <= 0) {
    OP_LOGE(op_info.op_name.GetString(), "dilations should be positive, actual is [%d,%d].", dilation_h, dilation_w);
    err_map["param_name"] = "dilations";
    err_map["op_name"] = op_info.op_name.GetString();
    err_map["expected_value"] = "positive";
    err_map["input_value"] = std::to_string(dilation_h) + ", " + std::to_string(dilation_w);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  attr_paras.h_attr.stride = stride_h;
  attr_paras.w_attr.stride = stride_w;
  attr_paras.h_attr.kernel = (filter_value.h_value - 1) * dilation_h + 1;
  attr_paras.w_attr.kernel = (filter_value.w_value - 1) * dilation_w + 1;

  if (string(op_info.op_type.GetString()) == "Conv2DTranspose") {
    std::vector<int32_t> output_padding_list;
    AttrUtils::GetListInt(op_info.op_desc, "output_padding", output_padding_list);
    int32_t outputpadding_h = output_padding_list[y_format_pos.h_position];
    int32_t outputpadding_w = output_padding_list[y_format_pos.w_position];
    attr_paras.h_attr.kernel = outputpadding_h + ((filter_value.h_value - 1) * dilation_h + 1);
    attr_paras.w_attr.kernel = outputpadding_w + ((filter_value.w_value - 1) * dilation_w + 1);
  }

  attr_paras.h_attr.pads = 0;
  attr_paras.w_attr.pads = 0;
  std::vector<int32_t> pads_list;
  if (AttrUtils::GetListInt(op_info.op_desc, "pads", pads_list)) {
    attr_paras.h_attr.pads = pads_list[kConv2dPadUpIdx] + pads_list[kConv2dPadDownIdx];
    attr_paras.w_attr.pads = pads_list[kConv2dPadLeftIdx] + pads_list[kConv2dPadRightIdx];
  }
  return true;
}

static bool SetConv2dBpInputOutShapeRange(const OpDetailInfo &op_info, const PosInfo &y_format_pos,
                                          const ShapeValInfo &filter_value, const GeShape &input_shape,
                                          std::vector<std::pair<int64_t, int64_t>> &input_range,
                                          OutputTuple &y_infer_info) {
  GeTensorDescPtr y_desc;
  bool is_input_size_const;
  std::tie(y_desc, is_input_size_const) = y_infer_info;
  int64_t groups = 1;
  if (!AttrUtils::GetInt(op_info.op_desc, "groups", groups)) {
    OP_LOGD(op_info.op_name.GetString(), "no groups setting, use groups as 1");
  }

  if (input_shape.IsUnknownDimNum()) {
    GeShape unknown_rank_shape({-1, -1, -1, -1});
    unknown_rank_shape.SetDim(y_format_pos.c_position, groups * filter_value.c_value);
    y_desc->SetShape(unknown_rank_shape);
    return true;
  }

  const GeShape &y_shape = y_desc->GetShape();
  if (y_shape.GetDimNum() != kConv2dDimSizeLimit) {
    OP_LOGE(op_info.op_name.GetString(), "y_shape list should be 4D. actual is: %lu.", y_shape.GetDimNum());
    map<string, string> err_map;
    err_map["param_name"] = "y_shape";
    err_map["op_name"] = op_info.op_name.GetString();
    err_map["expected_value"] = "4D";
    err_map["input_value"] = std::to_string(y_shape.GetDimNum()) + "D.";
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  AttrInfo attr_paras;
  CHECK(!GetConv2dBpAttrInfo(op_info, y_format_pos, filter_value, attr_paras),
        OP_LOGE(op_info.op_name.GetString(), "failed to get attr params"), return false);

  int32_t n_y_position = y_format_pos.n_position;
  int32_t c_y_position = y_format_pos.c_position;
  int32_t h_y_position = y_format_pos.h_position;
  int32_t w_y_position = y_format_pos.w_position;
  int64_t dx_h = y_shape.GetDim(h_y_position);
  int64_t dx_w = y_shape.GetDim(w_y_position);
  int64_t dx_n = y_shape.GetDim(n_y_position);
  int64_t dx_c = filter_value.c_value * groups;
  int64_t input_h = input_shape.GetDim(h_y_position);
  int64_t input_w = input_shape.GetDim(w_y_position);
  std::vector<std::pair<int64_t, int64_t>> y_range;
  y_range.resize(kConv2dDimSizeLimit);
  y_range[n_y_position] = std::make_pair(dx_n, dx_n);
  y_range[h_y_position] = std::make_pair(dx_h, dx_h);
  y_range[w_y_position] = std::make_pair(dx_w, dx_w);
  y_range[c_y_position] = std::make_pair(dx_c, dx_c);
  if (input_range.size() == kConv2dDimSizeLimit) {
    bool is_infer_shape = !is_input_size_const || input_range[n_y_position].second != -1;
    if (is_infer_shape) {
      y_range[n_y_position] = input_range[n_y_position];
    }
    // input_size all zero need to infer range
    is_infer_shape = dx_h == -1 || (input_h == -1 && (input_range[h_y_position].second != -1 || dx_h == 0));
    if (is_infer_shape) {
      SetConv2dBpInputOutRange(op_info.op_desc, h_y_position, attr_paras.h_attr, input_range, y_range);
    }
    is_infer_shape = dx_w == -1 || (input_w == -1 && (input_range[w_y_position].second != -1 || dx_w == 0));
    if (is_infer_shape) {
      SetConv2dBpInputOutRange(op_info.op_desc, w_y_position, attr_paras.w_attr, input_range, y_range);
    }
    y_desc->SetShapeRange(y_range);
  }
  return true;
}

static void ResetConv2dbpOutShape(const AscendString &op_name, const GeShape &input_shape, GeShape &out_shape,
                                  const PosInfo &y_format_pos) {
  if (input_shape.GetDim(y_format_pos.n_position) == -1) {
    out_shape.SetDim(y_format_pos.n_position, -1);
  }
  if (input_shape.GetDim(y_format_pos.c_position) == -1) {
    OP_LOGW(op_name.GetString(), "input x channel is unknow, fixed channel = %d.",
            (int)input_shape.GetDim(y_format_pos.c_position));
  }
  if (input_shape.GetDim(y_format_pos.h_position) == -1) {
    out_shape.SetDim(y_format_pos.h_position, -1);
  }
  if (input_shape.GetDim(y_format_pos.w_position) == -1) {
    out_shape.SetDim(y_format_pos.w_position, -1);
  }
}

static bool GetPosInfoByFormat(const Format &format, PosInfo &format_pos) {
  std::string format_str = format2str[format];
  CHECK(format_str.length() != kConv2dDimSizeLimit, OP_LOGE("", "the format is not 4D"), return false);
  format_pos.n_position = format_str.find("N");
  format_pos.h_position = format_str.find("H");
  format_pos.w_position = format_str.find("W");
  format_pos.c_position = format_str.find("C");
  return true;
}

static bool ResetInputRange(const AscendString &op_name, const GeTensorDescPtr &input_desc, const GeShape &input_shape,
                            std::vector<std::pair<int64_t, int64_t>> &input_range) {
  CHECK(input_shape.IsUnknownDimNum(), OP_LOGD(op_name.GetString(), "input shape is -2"), return true);
  bool input_range_invalid = input_range.size() != 0 && input_range.size() != input_shape.GetDimNum();
  if (input_range_invalid) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(),
                          "input range size must be equal to shape size, or equal to 0 in dynamic.");
    return false;
  }

  size_t dim_num = input_shape.GetDimNum();
  if (input_range.size() == 0) {
    input_range.resize(dim_num);
    for (size_t i = 0; i < dim_num; i++) {
      input_range[i].first = std::max(input_shape.GetDim(i), kDynamicRangeLowerBound);
      input_range[i].second = input_shape.GetDim(i);
    }
  } else if (input_shape.IsUnknownShape()) {
    for (size_t i = 0; i < dim_num; i++) {
      int64_t dim = input_shape.GetDim(i);
      bool is_static_dim = dim > 0 && (input_range[i].first != dim || input_range[i].second != dim);
      if (is_static_dim) {
        input_range[i].first = dim;
        input_range[i].second = dim;
      }
    }
  }
  input_desc->SetShapeRange(input_range);
  return true;
}

static void SetPadsByPaddingForDynamic(const AscendString &op_name, const OpDescPtr &op_desc) {
  std::string padding;
  if (AttrUtils::GetStr(op_desc, "padding", padding)) {
    std::vector<int32_t> pads(kConv2dPadSizeLimit, 0);
    if (padding == "SAME") {
      pads.assign(kConv2dPadSizeLimit, -1);
      OP_LOGD(op_name.GetString(), "set pads to {-1, -1, -1, -1} when padding is SAME in dynamic_shape");
    }
    AttrUtils::SetListInt(op_desc, "pads", pads);
  }
}

static bool InferConv2dBpInputOutShapeRange(const OpDetailInfo &op_info, const GeTensorDescPtr &input_sizes_desc,
                                            const ShapeValInfo &filter_value, const GeTensorDescPtr &input_desc,
                                            OutputTuple& y_infer_info) {
  GeTensorDescPtr y_desc;
  bool is_input_size_const;
  std::tie(y_desc, is_input_size_const) = y_infer_info;
  const GeShape &input_shape = input_desc->GetShape();
  std::vector<std::pair<int64_t, int64_t>> input_range;
  input_desc->GetShapeRange(input_range);
  std::vector<std::pair<int64_t, int64_t>> input_sizes_range;
  input_sizes_desc->GetValueRange(input_sizes_range);

  Format y_format = y_desc->GetFormat();
  CHECK(FORMAT_RESERVED == y_format, OP_LOGE(op_info.op_name.GetString(), "get format failed: %d", y_format),
        return false);
  CHECK_KEY_IN_MAP(format2str, y_format, "y_format", return false);
  PosInfo y_format_pos;
  CHECK(!GetPosInfoByFormat(y_format, y_format_pos), OP_LOGE(op_info.op_name.GetString(), "get position info failed."),
        return false);

  if ((input_sizes_range.size() == kConv2dDimSizeLimit) && (input_range.size() == kConv2dDimSizeLimit)) {
    bool is_mod_value_range =
        input_sizes_range[y_format_pos.c_position].first == -1 ||
        input_sizes_range[y_format_pos.c_position].first != input_sizes_range[y_format_pos.c_position].second;
    if (is_mod_value_range) {
      int64_t groups = 1;
      if (!AttrUtils::GetInt(op_info.op_desc, "groups", groups)) {
        OP_LOGD(op_info.op_name.GetString(), "no groups setting, use groups as 1");
      }
      int64_t cin = groups * filter_value.c_value;
      input_sizes_range[y_format_pos.c_position].first = cin;
      input_sizes_range[y_format_pos.c_position].second = cin;
    }
    y_desc->SetShapeRange(input_sizes_range);
    OP_LOGD(op_info.op_name.GetString(), "get value_range success from GE.");
  } else {
    CHECK(!ResetInputRange(op_info.op_name, input_desc, input_shape, input_range),
          OP_LOGE(op_info.op_name.GetString(), "the input range is unvalid"), return false);
    CHECK(!SetConv2dBpInputOutShapeRange(op_info, y_format_pos, filter_value, input_shape, input_range, y_infer_info),
          OP_LOGE(op_info.op_name.GetString(), "set out shape and range failed."), return false);
  }

  SetPadsByPaddingForDynamic(op_info.op_name, op_info.op_desc);
  if (input_shape.IsUnknownDimNum()) {
    OP_LOGD(op_info.op_name.GetString(), "input_shape is -2, infer out shape and range success.");
    return true;
  }

  std::vector<std::pair<int64_t, int64_t>> y_range;
  y_desc->GetShapeRange(y_range);
  CHECK(y_range.size() != kConv2dDimSizeLimit, OP_LOGE(op_info.op_name.GetString(), "the out range must be 4D"),
        return false);
  GeShape out_shape({-1, -1, -1, -1});
  for (size_t i = 0; i < y_range.size(); i++) {
    if (y_range[i].first == y_range[i].second) {
      out_shape.SetDim(i, y_range[i].first);
    }
    OP_LOGD(op_info.op_name.GetString(), "dedx range[%u] is [%lld, %lld]", i, y_range[i].first, y_range[i].second);
  }
  ResetConv2dbpOutShape(op_info.op_name, input_shape, out_shape, y_format_pos);
  y_desc->SetShape(out_shape);
  OP_LOGD(op_info.op_name.GetString(), "infer out shape and range success.");
  return true;
}

static bool SetConvGroups(const OpDetailInfo &op_info, int64_t c_y_shape, int64_t c_filter) {
  OP_LOGD(op_info.op_name.GetString(), "Setgroups begin.");
  int32_t x_c = c_y_shape;
  int32_t w_c = c_filter;
  int32_t groups = 1;
  if (w_c == 0) {
    OP_LOGE(op_info.op_name.GetString(), "channel of filter can not be 0.");
    map<string, string> err_map;
    err_map["op_name"] = op_info.op_name.GetString();
    err_map["description"] = "channel of filter can not be 0.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  } else if (x_c % w_c != 0) {
    OP_LOGE(op_info.op_name.GetString(), "fmap_channel %% filter_channel != 0");
    map<string, string> err_map;
    err_map["op_name"] = op_info.op_name.GetString();
    err_map["description"] = "fmap_channel % filter_channel != 0";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  groups = x_c / w_c;

  int32_t groups_ori = 1;
  AttrUtils::GetInt(op_info.op_desc, "groups", groups_ori);
  if (groups_ori == 1) {
    AttrUtils::SetInt(op_info.op_desc, "groups", groups);
    OP_LOGD(op_info.op_name.GetString(), "op set groups succ, groups:%d.", groups);
    return true;
  } else if (groups_ori != groups) {
    OP_LOGE(op_info.op_name.GetString(), "fmap_channel / filter_channel != groups");
    map<string, string> err_map;
    err_map["op_name"] = op_info.op_name.GetString();
    err_map["description"] = "fmap_channel / filter_channel != groups";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  return true;
}

static bool SetConv2dbpPadsByPadding(const OpDetailInfo &op_info, const GeShape &inputSizes,
                                     const PosInfo &y_format_pos, const ShapeValInfo &filter_value) {
  OP_LOGD(op_info.op_name.GetString(), "Set Conv2dbp Padst By Padding begin.");
  AttrInfo attr_paras;
  CHECK(!GetConv2dBpAttrInfo(op_info, y_format_pos, filter_value, attr_paras),
        OP_LOGE(op_info.op_name.GetString(), "failed to get attr params"), return false);

  int32_t dx_h = inputSizes.GetDim(y_format_pos.h_position);
  int32_t dx_w = inputSizes.GetDim(y_format_pos.w_position);
  std::string padding;
  std::vector<int32_t> pads(kConv2dPadSizeLimit, 0);
  if (AttrUtils::GetStr(op_info.op_desc, "padding", padding)) {
    if (padding == "SAME") {
      int32_t pad_h = std::max((int32_t)(align_conv2dbp(dx_h, attr_paras.h_attr.stride) - attr_paras.h_attr.stride +
                                         attr_paras.h_attr.kernel - dx_h),
                               0);
      int32_t pad_up = (pad_h >> 1);
      int32_t pad_down = pad_h - pad_up;
      int32_t pad_w = std::max((int32_t)(align_conv2dbp(dx_w, attr_paras.w_attr.stride) - attr_paras.w_attr.stride +
                                         attr_paras.w_attr.kernel - dx_w),
                               0);
      int32_t pad_left = (pad_w >> 1);
      int32_t pad_right = pad_w - pad_left;
      pads[kConv2dPadUpIdx] = pad_up;
      pads[kConv2dPadDownIdx] = pad_down;
      pads[kConv2dPadLeftIdx] = pad_left;
      pads[kConv2dPadRightIdx] = pad_right;
    }
    AttrUtils::SetListInt(op_info.op_desc, "pads", pads);
  }

  CHECK(!VerifyConv2dbpPads(op_info.op_name, op_info.op_desc), OP_LOGE(op_info.op_name.GetString(), "pads is unvalid"),
        return false);
  OP_LOGD(op_info.op_name.GetString(), "op set pads succ.");
  return true;
}

static bool GetFilterValues(const GeTensorDescPtr &filter_desc, ShapeValInfo &filter_value) {
  Format filter_format = filter_desc->GetFormat();
  CHECK(FORMAT_RESERVED == filter_format, OP_LOGE("", "the filter format is illegal"), return false);
  CHECK_KEY_IN_MAP(format2str, filter_format, "filter_format", return false);
  PosInfo filter_format_pos;
  CHECK(!GetPosInfoByFormat(filter_format, filter_format_pos), OP_LOGE("", "get position info failed"), return false);
  const GeShape &filter_shape = filter_desc->GetShape();
  filter_value.n_value = filter_shape.GetDim(filter_format_pos.n_position);
  filter_value.h_value = filter_shape.GetDim(filter_format_pos.h_position);
  filter_value.w_value = filter_shape.GetDim(filter_format_pos.w_position);
  filter_value.c_value = filter_shape.GetDim(filter_format_pos.c_position);
  return true;
}

static bool checkConv2dBppads(const OpDetailInfo &op_info, const GeShape &input_shape, const ShapeValInfo &filter_value,
                              const PosInfo &y_format_pos, const GeShape &y_shape) {
  std::string padding;
  if (AttrUtils::GetStr(op_info.op_desc, "padding", padding)) {
    return true;
  }

  AttrInfo attr_paras;
  CHECK(!GetConv2dBpAttrInfo(op_info, y_format_pos, filter_value, attr_paras),
        OP_LOGE(op_info.op_name.GetString(), "failed to get attr params"), return false);
  int64_t dx_h = y_shape.GetDim(y_format_pos.h_position);
  int64_t dx_w = y_shape.GetDim(y_format_pos.w_position);
  int64_t dy_h = input_shape.GetDim(y_format_pos.h_position);
  int64_t dy_w = input_shape.GetDim(y_format_pos.w_position);
  int64_t dy_h_new = (dx_h + attr_paras.h_attr.pads - attr_paras.h_attr.kernel) / attr_paras.h_attr.stride + 1;
  int64_t dy_w_new = (dx_w + attr_paras.w_attr.pads - attr_paras.w_attr.kernel) / attr_paras.w_attr.stride + 1;

  CHECK(dy_h_new != dy_h || dy_w_new != dy_w, OP_LOGE(op_info.op_name.GetString(), "check pads attrs failed."),
        return false);
  return true;
}

IMPLEMT_INFERFUNC(Conv2DBackpropInput, Conv2DBackpropInputInfer) {
  OpDetailInfo op_info;
  CHECK(op.GetName(op_info.op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);
  op_info.op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_info.op_desc, "op desc", return GRAPH_FAILED);
  CHECK_OP_FUNC(op.GetOpType(op_info.op_type) != GRAPH_SUCCESS, return GRAPH_FAILED, "failed to get op_type");

  OP_LOGI(op_info.op_name.GetString(), "Enter Conv2DBackpropInput inferfunction!");
  std::vector<std::string> input_infer_depends = {"input_size"};
  op_info.op_desc->SetOpInferDepends(input_infer_depends);

  GeTensorDescPtr y_desc = op_info.op_desc->MutableOutputDesc(0);
  CHECK_PTR_NULL(y_desc, "tensor y desc", return GRAPH_FAILED);
  GeTensorDescPtr input_sizes_desc = op_info.op_desc->MutableInputDesc(0);
  CHECK_PTR_NULL(input_sizes_desc, "tensor input_sizes desc", return GRAPH_FAILED);
  GeTensorDescPtr filter_desc = op_info.op_desc->MutableInputDesc(1);
  CHECK_PTR_NULL(filter_desc, "tensor filter desc", return GRAPH_FAILED);
  // the index of out_backprop desc tensor is 2
  GeTensorDescPtr out_backprop_desc = op_info.op_desc->MutableInputDesc(2);
  CHECK_PTR_NULL(out_backprop_desc, "tensor out_backprop desc", return GRAPH_FAILED);

  Format y_format = y_desc->GetFormat();
  CHECK_OP_FUNC(FORMAT_RESERVED == y_format, return GRAPH_FAILED, "get format failed: %d", y_format);
  CHECK_KEY_IN_MAP(format2str, y_format, "y_format", return GRAPH_FAILED);
  PosInfo y_format_pos;
  CHECK_OP_FUNC(!GetPosInfoByFormat(y_format, y_format_pos), return GRAPH_FAILED, "get position info failed.");
  const GeShape &out_backprop_shape = out_backprop_desc->GetShape();

  bool is_dynamic = out_backprop_shape.IsUnknownShape();
  DataType input_sizes_dtype = input_sizes_desc->GetDataType();
  bool is_input_size_const = false;
  CHECK_OP_FUNC(!CheckAndSetInputSizes(op, input_sizes_dtype, "input_size", is_input_size_const, y_desc),
                return GRAPH_FAILED, "get input_size const fail.");

  ShapeValInfo filter_value;
  CHECK_OP_FUNC(!GetFilterValues(filter_desc, filter_value), return GRAPH_FAILED, "get filter_value fail.");
  OutputTuple y_infer_info(y_desc, is_input_size_const);
  // dynamic shape scene
  if (!is_input_size_const || is_dynamic) {
    CHECK_OP_FUNC(!InferConv2dBpInputOutShapeRange(op_info, input_sizes_desc, filter_value, out_backprop_desc,
                                                   y_infer_info),
                  return GRAPH_FAILED, "infer out shape and range fail.");
    GeShape &out_shape = y_desc->MutableShape();
    bool is_force_dynamic =
        (out_shape.GetDimNum() == kConv2dDimSizeLimit && !is_dynamic && !out_shape.IsUnknownShape());
    if (is_force_dynamic) {
      OP_LOGD(op_info.op_name.GetString(),
              "input_sizes is Data but all tensors have no -1, so force to set HW in y to -1.");
      out_shape.SetDim(y_format_pos.h_position, -1);
      out_shape.SetDim(y_format_pos.w_position, -1);
    }
  }

  const GeShape &y_shape = y_desc->GetShape();
  bool is_strict_static = is_input_size_const && !is_dynamic;
  if (is_strict_static) {
    CHECK_OP_FUNC(!SetConv2dbpPadsByPadding(op_info, y_shape, y_format_pos, filter_value), return GRAPH_FAILED,
                  "update pads list by padding failed.");
    CHECK_OP_FUNC(!checkConv2dBppads(op_info, out_backprop_shape, filter_value, y_format_pos, y_shape),
                  return GRAPH_FAILED, "check Conv2dBp pads failed.");
  }

  CHECK_OP_FUNC(!SetConvGroups(op_info, y_shape.GetDim(y_format_pos.c_position), filter_value.c_value),
                return GRAPH_FAILED, "Set groups for Conv2DBackpropInput failed.");

  // set dtype of output desc
  DataType out_backprop_dtype = out_backprop_desc->GetDataType();
  y_desc->SetDataType(out_backprop_dtype);
  if (out_backprop_dtype == DT_INT8) {
    y_desc->SetDataType(DT_INT32);
  }

  OP_LOGI(op_info.op_name.GetString(), "Leaving Conv2DBackpropInput inferfunction!");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv2DBackpropInput, Conv2DBackpropInputVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);
  OpDescPtr op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);

  // the filter tensor index is 1
  ConstGeTensorDescPtr filter_desc = op_desc->GetInputDescPtr(1);
  CHECK_PTR_NULL(filter_desc, "filter desc", return GRAPH_FAILED);
  // the out_backprop tensor index is 2
  ConstGeTensorDescPtr out_backprop_desc = op_desc->GetInputDescPtr(2);
  CHECK_PTR_NULL(out_backprop_desc, "out_backprop desc", return GRAPH_FAILED);

  OP_LOGI(op_name.GetString(), "Enter Conv2DBackpropInput verifyfunction!");
  if (GRAPH_SUCCESS != VerifyConv2dbpCommon(op_name, op_desc, filter_desc, out_backprop_desc)) {
    return GRAPH_FAILED;
  }
  OP_LOGI(op_name.GetString(), "Leaving Conv2DBackpropInput verifyfunction!");
  return GRAPH_SUCCESS;
}

INFER_DATA_SLICE_FUNC_REG(Conv2DBackpropInput, Conv2DBackpropInputInferDataSlice);
INFER_FUNC_REG(Conv2DBackpropInput, Conv2DBackpropInputInfer);
VERIFY_FUNC_REG(Conv2DBackpropInput, Conv2DBackpropInputVerify);

// ----------------Conv2DBackpropInputD-------------------

// get infer date slice
IMPLEMT_INFER_DATA_SLICE(Conv2DBackpropInputD, Conv2DBackpropInputDInferDataSlice) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter Conv2DBackpropInput InferDataSlice.");

  // get dedy shape, stride and dilation
  auto dedy_desc = op.GetInputDescByName("out_backprop");
  auto dedy_shape = dedy_desc.GetOriginShape().GetDims();
  auto dedy_format = dedy_desc.GetOriginFormat();
  std::vector<int32_t> stride_list;
  op.GetAttr("strides", stride_list);
  std::vector<int32_t> dilation_list;
  op.GetAttr("dilations", dilation_list);
  std::vector<int32_t> input_sizes;
  op.GetAttr("input_size", input_sizes);

  int32_t ih = -1;
  int32_t iw = -1;
  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;
  if (dedy_shape.size() != kConv2dDimSizeLimit) {
    OP_LOGE(op_name.GetString(), "dedy_shape is invalid");
    ErrorManager::GetInstance().ATCReportErrMessage("E50058",
                                                    {"op_name", "description"},
                                                    {op_name.GetString(), "dedy_shape is invalid"});
    return GRAPH_FAILED;
  }
  if (stride_list.size() != kConv2dDimSizeLimit) {
    OP_LOGE(op_name.GetString(), "stride is invalid");
    ErrorManager::GetInstance().ATCReportErrMessage("E50058",
                                                    {"op_name", "description"},
                                                    {op_name.GetString(), "stride is invalid"});
    return GRAPH_FAILED;
  }
  if (dilation_list.size() != kConv2dDimSizeLimit) {
    OP_LOGE(op_name.GetString(), "dilation is invalid");
    ErrorManager::GetInstance().ATCReportErrMessage("E50058",
                                                    {"op_name", "description"},
                                                    {op_name.GetString(), "dilation is invalid"});
    return GRAPH_FAILED;
  }
  if (input_sizes.size() != kConv2dDimSizeLimit) {
    OP_LOGE(op_name.GetString(), "input sizes is invalid");
    ErrorManager::GetInstance().ATCReportErrMessage("E50058",
                                                    {"op_name", "description"},
                                                    {op_name.GetString(), "input sizes is invalid"});
    return GRAPH_FAILED;
  }

  if (dedy_format == FORMAT_NCHW) {
    if (dedy_shape != DYNAMIC_DIM_ALL) {
      ih = dedy_shape[kHDimNCHWIdx];
      iw = dedy_shape[kWDimNCHWIdx];
    }
    strh = stride_list[kHDimNCHWIdx];
    strw = stride_list[kWDimNCHWIdx];
    dilh = dilation_list[kHDimNCHWIdx];
    dilw = dilation_list[kWDimNCHWIdx];
  } else if (dedy_format == FORMAT_NHWC) {
    if (dedy_shape != DYNAMIC_DIM_ALL) {
      ih = dedy_shape[kHDimNHWCIdx];
      iw = dedy_shape[kWDimNHWCIdx];
    }
    strh = stride_list[kHDimNHWCIdx];
    strw = stride_list[kWDimNHWCIdx];
    dilh = dilation_list[kHDimNHWCIdx];
    dilw = dilation_list[kWDimNHWCIdx];
  }
  bool stride_invalid = (strh <= 0) || (strw <= 0);
  if (stride_invalid) {
    OP_LOGE(op_name.GetString(), "stride can not less than zero");
    ErrorManager::GetInstance().ATCReportErrMessage("E50029",
                                                    {"op_name", "param_name", "expected_value", "input_value"},
                                                    {op_name.GetString(), "strides", "positive",
                                                    std::to_string(strh) + ", " + std::to_string(strw)});
    return GRAPH_FAILED;
  }
  // get filter shape
  auto filter_desc = op.GetInputDescByName("filter");
  auto filter_shape = filter_desc.GetOriginShape().GetDims();
  auto filter_format = filter_desc.GetOriginFormat();
  int32_t kh = 0;
  int32_t kw = 0;
  if (filter_format == FORMAT_NCHW) {
    kh = filter_shape[kHDimNCHWIdx];
    kw = filter_shape[kWDimNCHWIdx];
  } else if (filter_format == FORMAT_NHWC) {
    kh = filter_shape[kHDimNHWCIdx];
    kw = filter_shape[kWDimNHWCIdx];
  } else if (filter_format == FORMAT_HWCN) {
    kh = filter_shape[0];
    kw = filter_shape[1];
  }

  auto y_desc = op.GetOutputDescByName("y");
  auto y_format = y_desc.GetOriginFormat();
  CHECK_KEY_IN_MAP(format2str, y_format, "y_format", return GRAPH_FAILED);
  std::string y_format_str = format2str[y_format];
  int32_t n_y_position = y_format_str.find("N");
  int32_t c_y_position = y_format_str.find("C");
  int32_t h_y_position = y_format_str.find("H");
  int32_t w_y_position = y_format_str.find("W");

  // get pads
  std::vector<int32_t> pad_list;
  op.GetAttr("pads", pad_list);
  if (pad_list.size() != kConv2dPadSizeLimit) {
    OP_LOGE(op_name.GetString(), "pad is invalid");
    ErrorManager::GetInstance().ATCReportErrMessage("E50058",
                                                    {"op_name", "description"},
                                                    {op_name.GetString(), "pad is invalid"});
    return GRAPH_FAILED;
  }
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_dedy = op_desc->MutableInputDesc("out_backprop");
  GeTensorDescPtr tensor_desc_filter = op_desc->MutableInputDesc("filter");

  CHECK_PTR_NULL(tensor_desc_y, "tensor y desc", return GRAPH_FAILED);
  CHECK_PTR_NULL(tensor_desc_dedy, "tensor dedy desc", return GRAPH_FAILED);
  CHECK_PTR_NULL(tensor_desc_filter, "tensor filter desc", return GRAPH_FAILED);

  vector<vector<int64_t>> y_data_slice;
  vector<vector<int64_t>> dedy_data_slice = {{}, {}, {}, {}, {}};
  vector<vector<int64_t>> filter_data_slice = {{}, {}, {}, {}};

  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op_name.GetString(), "no data slice, not need infer input");
    return GRAPH_FAILED;
  }

  for (size_t i = 0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() > 0) {
      vector<int32_t> y_position = {n_y_position, c_y_position, h_y_position, w_y_position};
      vector<int32_t> attr_param = {kh, kw, dilh, dilw, strh, strw, ih, iw};
      return data_slice_conv2d_backprop_input(op, y_data_slice, dedy_data_slice, filter_data_slice, y_position,
                                              attr_param, input_sizes, pad_list, tensor_desc_filter,
                                              tensor_desc_dedy, i);
    }
  }

  OP_LOGI(op_name.GetString(), "data slice without overlap, not need infer input");
  return GRAPH_FAILED;
}

IMPLEMT_INFERFUNC(Conv2DBackpropInputD, Conv2DBackpropInputDInfer) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGI(op_name.GetString(), "Enter Conv2DBackpropInputD inferfunction!");

  auto out_backprop_desc = op.GetInputDescByName("out_backprop");
  // get dtype for output from out_backprop
  auto out_backprop_dtype = out_backprop_desc.GetDataType();
  // get shape for output from input_size
  std::vector<int32_t> input_sizes;
  if (GRAPH_SUCCESS == op.GetAttr("input_size", input_sizes)) {
    if (input_sizes.size() != kConv2dDimSizeLimit) {
      OP_LOGE(op_name.GetString(), "input_size list should be 4d.");
      map<std::string, std::string> err_map;
      err_map["op_name"] = "Conv2DBackpropInput";
      err_map["param_name"] = "input_size";
      err_map["expected_length"] = "4";
      err_map["length"] = std::to_string(input_sizes.size());
      std::string report_error_code = "E50035";
      (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op_name.GetString(), "get input_size list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropInput";
    err_map["param_name"] = "input_size";
    std::string report_error_code = "E50030";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // set dtype of output desc
  auto y_desc = op.GetOutputDescByName("y");
  if (out_backprop_dtype == DT_INT8) {
    y_desc.SetDataType(DT_INT32);
  } else {
    y_desc.SetDataType(out_backprop_dtype);
  }

  // set shape of output desc, input_size should match the format of y
  std::vector<int64_t> y_shape;
  y_shape.push_back(input_sizes[0]);
  y_shape.push_back(input_sizes[1]);
  y_shape.push_back(input_sizes[kWDimNHWCIdx]);
  y_shape.push_back(input_sizes[kCDimNHWCIdx]);
  y_desc.SetShape(ge::Shape(y_shape));

  // update output desc
  if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", y_desc)) {
    OP_LOGE(op_name.GetString(), "update output desc failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropInput";
    err_map["param_name"] = "updating result";
    err_map["rule_desc"] = "updata OutputDesc";
    err_map["param_value"] = "failed";
    std::string report_error_code = "E50012";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> filter_sizes = op.GetInputDescByName("filter").GetShape().GetDims();
  Format filter_format = op.GetInputDescByName("filter").GetFormat();
  Format input_format = y_desc.GetFormat();
  CHECK_OP_FUNC(FORMAT_RESERVED == filter_format, return GRAPH_FAILED, "get format failed: %d", filter_format);
  CHECK_OP_FUNC(FORMAT_RESERVED == input_format, return GRAPH_FAILED, "get format failed: %d", input_format);
  if (false == SetGroupsConv(op, input_sizes, input_format, filter_sizes, filter_format)) {
    OP_LOGE(op_name.GetString(), "Set groups for Conv2DBackpropInputD failed.");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "Set groups for Conv2DBackpropInputD failed.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  // update pads list by padding[SAME,VALID]
  if (false == SetPadListByPaddingConv2dbp(op, input_sizes, input_format, filter_sizes, filter_format)) {
    OP_LOGE(op_name.GetString(), "Conv2DBackpropInputD update pads list by padding failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropInput";
    err_map["param_name"] = "updating result";
    err_map["rule_desc"] = "updata pads list by padding";
    err_map["param_value"] = "failed";
    std::string report_error_code = "E50012";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  OP_LOGI(op_name.GetString(), "Leaving Conv2DBackpropInputD inferfunction!");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv2DBackpropInputD, Conv2DBackpropInputDVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);
  OpDescPtr op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);

  // the filter tensor index is 0
  ConstGeTensorDescPtr filter_desc = op_desc->GetInputDescPtr(0);
  CHECK_PTR_NULL(filter_desc, "filter desc", return GRAPH_FAILED);
  // the out_backprop tensor index is 1
  ConstGeTensorDescPtr out_backprop_desc = op_desc->GetInputDescPtr(1);
  CHECK_PTR_NULL(out_backprop_desc, "out_backprop desc", return GRAPH_FAILED);

  OP_LOGI(op_name.GetString(), "Enter Conv2DBackpropInputD verifyfunction!");
  if (GRAPH_SUCCESS != VerifyConv2dbpCommon(op_name, op_desc, filter_desc, out_backprop_desc)) {
    return GRAPH_FAILED;
  }
  OP_LOGI(op_name.GetString(), "Leaving Conv2DBackpropInputD verifyfunction!");
  return GRAPH_SUCCESS;
}

INFER_DATA_SLICE_FUNC_REG(Conv2DBackpropInputD, Conv2DBackpropInputDInferDataSlice);
INFER_FUNC_REG(Conv2DBackpropInputD, Conv2DBackpropInputDInfer);
VERIFY_FUNC_REG(Conv2DBackpropInputD, Conv2DBackpropInputDVerify);

// ----------------Conv2DBackpropFilter-------------------
bool InferConv2DBackpropFilter(ge::Operator& op) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  auto y_tensor = op.GetOutputDescByName("y");
  auto filter_format = y_tensor.GetOriginFormat();
  CHECK_KEY_IN_MAP(format2str, filter_format, "filter_format", return false);
  std::string filter_format_str = format2str[filter_format];
  int32_t n_filter_position = filter_format_str.find("N");
  // get shape for output from filter_size
  std::vector<int32_t> filter_sizes;
  if (GRAPH_SUCCESS != op.GetAttr("filter_size", filter_sizes)) {
    OP_LOGE(op_name.GetString(), "get filter_size list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropFilter";
    err_map["param_name"] = "filter_size";
    std::string report_error_code = "E50030";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  CHECK_SIZE(filter_sizes.size() != 4, return false, "filter sizes is invalid");

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return false);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_dedy = op_desc->MutableInputDesc("out_backprop");
  CHECK_PTR_NULL(tensor_desc_y, "tensor y desc", return false);
  CHECK_PTR_NULL(tensor_desc_dedy, "tensor dedy desc", return false);

  vector<vector<int64_t>> y_data_slice;
  vector<vector<int64_t>> dedy_data_slice = {{}, {}, {}, {}, {}};

  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op_name.GetString(), "no data slice, not need infer input");
    return false;
  }

  for (size_t i = 0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() > 0) {
      int32_t y_extend = y_data_slice[i][1] - y_data_slice[i][0] + 1;
      if (i == 1) {
        dedy_data_slice[i] = y_data_slice[i];
        CHECK_OP_FUNC(!AttrUtils::SetListListInt(tensor_desc_dedy, ge::ATTR_NAME_DATA_SLICE, dedy_data_slice),
                      return false, "set dedy data slice attr failed.");
        filter_sizes[n_filter_position] = y_extend * kBlockBaseSize;
        op.SetAttr("filter_size", filter_sizes);
        OP_LOGI(op_name.GetString(), "infer input in Cout success");
        return true;
      }
      OP_LOGI(op_name.GetString(), "can not supported split in Cin, H and W");
      return false;
    }
  }
  OP_LOGI(op_name.GetString(), "no data slice, not need infer input");
  return false;
}

IMPLEMT_INFERFUNC(Conv2DBackpropFilter, Conv2DBackpropFilterInfer) {
  OpDetailInfo op_info;
  CHECK(op.GetName(op_info.op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);
  op_info.op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_info.op_desc, "op desc", return GRAPH_FAILED);
  CHECK(op.GetOpType(op_info.op_type) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_type"), return GRAPH_FAILED);

  OP_LOGI(op_info.op_name.GetString(), "Enter Conv2DBackpropFilter inferfunction!");
  std::vector<std::string> input_infer_depends = {"filter_size"};
  op_info.op_desc->SetOpInferDepends(input_infer_depends);

  GeTensorDescPtr y_desc = op_info.op_desc->MutableOutputDesc(0);
  CHECK_PTR_NULL(y_desc, "tensor y desc", return GRAPH_FAILED);
  GeTensorDescPtr filter_sizes_desc = op_info.op_desc->MutableInputDesc(1);
  CHECK_PTR_NULL(filter_sizes_desc, "tensor filter_sizes desc", return GRAPH_FAILED);
  // the index of out_backprop desc tensor is 2
  GeTensorDescPtr out_backprop_desc = op_info.op_desc->MutableInputDesc(2);
  CHECK_PTR_NULL(out_backprop_desc, "tensor out_backprop desc", return GRAPH_FAILED);
  GeTensorDescPtr x_desc = op_info.op_desc->MutableInputDesc(0);
  CHECK_PTR_NULL(x_desc, "tensor x desc", return GRAPH_FAILED);

  Format filter_format = y_desc->GetFormat();
  CHECK_OP_FUNC(FORMAT_RESERVED == filter_format, return GRAPH_FAILED, "get format failed: %d", filter_format);
  CHECK_KEY_IN_MAP(format2str, filter_format, "filter_format", return GRAPH_FAILED);
  PosInfo filter_format_pos;
  CHECK_OP_FUNC(!GetPosInfoByFormat(filter_format, filter_format_pos), return GRAPH_FAILED,
                "get filter position info failed.");
  Format x_format = x_desc->GetFormat();
  CHECK_OP_FUNC(FORMAT_RESERVED == x_format, return GRAPH_FAILED, "get format failed: %d", x_format);
  CHECK_KEY_IN_MAP(format2str, x_format, "x_format", return GRAPH_FAILED);
  PosInfo x_format_pos;
  CHECK_OP_FUNC(!GetPosInfoByFormat(x_format, x_format_pos), return GRAPH_FAILED, "get position info failed.");

  DataType filter_sizes_dtype = filter_sizes_desc->GetDataType();
  bool is_filter_sizes_const = false;
  CHECK_OP_FUNC(!CheckAndSetInputSizes(op, filter_sizes_dtype, "filter_size", is_filter_sizes_const, y_desc),
                return GRAPH_FAILED, "get filter_sizes const fail.");

  const GeShape &x_sizes = x_desc->GetShape();
  bool unknown_rank_x = x_sizes.IsUnknownDimNum();
  int32_t x_c = -1;
  if (!unknown_rank_x) {
    x_c = x_sizes.GetDim(x_format_pos.c_position);
  }
  const GeShape &out_backprop_sizes = out_backprop_desc->GetShape();
  bool unknown_rank_outbp = out_backprop_sizes.IsUnknownDimNum();

  // dynamic shape scene
  if (!is_filter_sizes_const) {
    GeShape filter_sizes({-1, -1, -1, -1});
    if (!unknown_rank_outbp) {
      filter_sizes.SetDim(filter_format_pos.n_position, out_backprop_sizes.GetDim(x_format_pos.c_position));
    }
    int32_t groups_ori = 1;
    AttrUtils::GetInt(op_info.op_desc, "groups", groups_ori);
    CHECK_INFO(groups_ori == 0, return GRAPH_FAILED, "Get illegal groups: groups should not be zero.");
    CHECK_INFO(x_c % groups_ori != 0, return GRAPH_FAILED,
               "Get illegal groups: fmap's channel must be a multiple of groups.");
    filter_sizes.SetDim(filter_format_pos.c_position, x_c / groups_ori);
    y_desc->SetShape(filter_sizes);
  }

  SetInputConst(is_filter_sizes_const, x_sizes.IsUnknownShape(), unknown_rank_x, op_info.op_desc);
  const GeShape &y_shape = y_desc->GetShape();
  ShapeValInfo filter_value;
  filter_value.c_value = y_shape.GetDim(filter_format_pos.c_position);
  filter_value.h_value = y_shape.GetDim(filter_format_pos.h_position);
  filter_value.w_value = y_shape.GetDim(filter_format_pos.w_position);
  filter_value.n_value = y_shape.GetDim(filter_format_pos.n_position);
  bool is_strict_static = is_filter_sizes_const && !x_sizes.IsUnknownShape() && !unknown_rank_x;
  if (is_strict_static) {
    CHECK_OP_FUNC(!SetConv2dbpPadsByPadding(op_info, x_sizes, x_format_pos, filter_value), return GRAPH_FAILED,
                  "update pads list by padding failed.");
    if (!unknown_rank_outbp) {
      CHECK_OP_FUNC(!checkConv2dBppads(op_info, out_backprop_sizes, filter_value, x_format_pos, x_sizes),
                    return GRAPH_FAILED, "check Conv2dBp pads failed.");
    }
  } else {
    SetPadsByPaddingForDynamic(op_info.op_name, op_info.op_desc);
  }

  bool ignore_set_group = unknown_rank_x || filter_value.c_value < 1 || x_c < 1;
  if (ignore_set_group) {
    OP_LOGD(op_info.op_name.GetString(), "ignore set groups.");
  } else {
    CHECK_OP_FUNC(!SetConvGroups(op_info, x_c, filter_value.c_value), return GRAPH_FAILED,
                  "Set groups for Conv2DBackpropFilter failed.");
  }
  OP_LOGI(op_info.op_name.GetString(), "Leaving Conv2DBackpropFilter inferfunction!");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv2DBackpropFilter, Conv2DBackpropFilterVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);
  OpDescPtr op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);

  // the x tensor index is 0
  ConstGeTensorDescPtr x_desc = op_desc->GetInputDescPtr(0);
  CHECK_PTR_NULL(x_desc, "x desc", return GRAPH_FAILED);
  // the out_backprop tensor index is 2
  ConstGeTensorDescPtr out_backprop_desc = op_desc->GetInputDescPtr(2);
  CHECK_PTR_NULL(out_backprop_desc, "out_backprop desc", return GRAPH_FAILED);

  OP_LOGI(op_name.GetString(), "Enter Conv2DBackpropFilter verifyfunction!");
  if (GRAPH_SUCCESS != VerifyConv2dbpCommon(op_name, op_desc, x_desc, out_backprop_desc)) {
    return GRAPH_FAILED;
  }
  OP_LOGI(op_name.GetString(), "Leaving Conv2DBackpropFilter verifyfunction!");
  return GRAPH_SUCCESS;
}

IMPLEMT_INFER_DATA_SLICE(Conv2DBackpropFilter, Conv2DBackpropFilterInferDataSlice) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter Conv2DBackpropFilter InferDataSlice.");
  if (!InferConv2DBackpropFilter(op)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Conv2DBackpropFilter, Conv2DBackpropFilterInfer);
VERIFY_FUNC_REG(Conv2DBackpropFilter, Conv2DBackpropFilterVerify);
INFER_DATA_SLICE_FUNC_REG(Conv2DBackpropFilter, Conv2DBackpropFilterInferDataSlice);

// ----------------Conv2DBackpropFilterD-------------------
IMPLEMT_INFERFUNC(Conv2DBackpropFilterD, Conv2DBackpropFilterDInfer) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGI(op_name.GetString(), "Enter Conv2DBackpropFilterD inferfunction!");

  // get shape for output from filter_size
  std::vector<int32_t> filter_sizes;
  if (GRAPH_SUCCESS == op.GetAttr("filter_size", filter_sizes)) {
    if (filter_sizes.size() != kConv2dDimSizeLimit) {
      OP_LOGE(op_name.GetString(), "filter_size list should be 4d.");
      map<std::string, std::string> err_map;
      err_map["op_name"] = "Conv2DBackpropFilter";
      err_map["param_name"] = "filter_size";
      err_map["expected_length"] = "4";
      err_map["length"] = std::to_string(filter_sizes.size());
      std::string report_error_code = "E50035";
      (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op_name.GetString(), "get filter_size list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropFilter";
    err_map["param_name"] = "filter_size";
    std::string report_error_code = "E50030";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  auto y_desc = op.GetOutputDescByName("y");
  // set dtype of output desc
  y_desc.SetDataType(DT_FLOAT);

  // set shape of output desc, filter_size should match the format of y
  std::vector<int64_t> y_shape;
  y_shape.push_back(filter_sizes[0]);
  y_shape.push_back(filter_sizes[1]);
  y_shape.push_back(filter_sizes[kWDimNHWCIdx]);
  y_shape.push_back(filter_sizes[kCDimNHWCIdx]);
  y_desc.SetShape(ge::Shape(y_shape));

  // update output desc
  if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", y_desc)) {
    OP_LOGE(op_name.GetString(), "update output desc failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropFilter";
    err_map["param_name"] = "updating result";
    err_map["rule_desc"] = "updata OutputDesc";
    err_map["param_value"] = "failed";
    std::string report_error_code = "E50012";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> x_sizes = op.GetInputDescByName("x").GetShape().GetDims();
  Format x_format = op.GetInputDescByName("x").GetFormat();
  Format filter_format = y_desc.GetFormat();
  CHECK_OP_FUNC(FORMAT_RESERVED == x_format, return GRAPH_FAILED, "get format failed: %d", x_format);
  CHECK_OP_FUNC(FORMAT_RESERVED == filter_format, return GRAPH_FAILED, "get format failed: %d", filter_format);
  if (false == SetGroupsConv(op, x_sizes, x_format, filter_sizes, filter_format)) {
    OP_LOGE(op_name.GetString(), "Set groups for Conv2DBackpropFilterD failed.");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "Set groups for Conv2DBackpropFilterD failed";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  // update pads list by padding[SAME,VALID]
  if (false == SetPadListByPaddingConv2dbp(op, x_sizes, x_format, filter_sizes, filter_format)) {
    OP_LOGE(op_name.GetString(), "update pads list by padding failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DBackpropFilter";
    err_map["param_name"] = "updding result";
    err_map["rule_desc"] = "updata pads list by padding";
    err_map["param_value"] = "failed";
    std::string report_error_code = "E50012";
    (void)ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  OP_LOGI(op_name.GetString(), "Leaving Conv2DBackpropFilterD inferfunction!");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv2DBackpropFilterD, Conv2DBackpropFilterDVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);
  OpDescPtr op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);
  // the x tensor index is 0
  ConstGeTensorDescPtr x_desc = op_desc->GetInputDescPtr(0);
  CHECK_PTR_NULL(x_desc, "x desc", return GRAPH_FAILED);
  // the out_backprop tensor index is 1
  ConstGeTensorDescPtr out_backprop_desc = op_desc->GetInputDescPtr(1);
  CHECK_PTR_NULL(out_backprop_desc, "out_backprop desc", return GRAPH_FAILED);

  OP_LOGI(op_name.GetString(), "Enter Conv2DBackpropFilterD verifyfunction!");
  if (GRAPH_SUCCESS != VerifyConv2dbpCommon(op_name, op_desc, x_desc, out_backprop_desc)) {
    return GRAPH_FAILED;
  }
  OP_LOGI(op_name.GetString(), "Leaving Conv2DBackpropFilterD verifyfunction!");
  return GRAPH_SUCCESS;
}

IMPLEMT_INFER_DATA_SLICE(Conv2DBackpropFilterD, Conv2DBackpropFilterDInferDataSlice) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter Conv2DBackpropFilterD InferDataSlice.");
  if (!InferConv2DBackpropFilter(op)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Conv2DBackpropFilterD, Conv2DBackpropFilterDInfer);
VERIFY_FUNC_REG(Conv2DBackpropFilterD, Conv2DBackpropFilterDVerify);
INFER_DATA_SLICE_FUNC_REG(Conv2DBackpropFilterD, Conv2DBackpropFilterDInferDataSlice);

// --------------------------Conv2D------------------------------
/*!
  * @brief Convert different framework pad param to ir pads:
  *
  * [_padding]: 4D lsit, format sensitive, need convert to pads
  * [padding]: 'SAME' or 'VALID', need convert to pads
  * [pads]: 4D list, format sensitive, no need convert
  *
  * @param op Conv2D operator.
  * @param ih, iw  Input images H/W size.
  * @param kh, kw  Input filter H/W size.
  * @param strh, strw  Input stride H/W value.
  * @param dilh, dilw  Input dilation H/W value.
  * @param padt, padb, padl, padr Top, bottom, left, right padding.
  * @return bool Whether the pads setting is correct.
  */
static bool GetPadConv2D(const AscendString& op_name, ge::OpDescPtr& op_desc,
                         int32_t ih, int32_t iw, int32_t kh, int32_t kw, int32_t strh, int32_t strw,
                         int32_t dilh, int32_t dilw, int32_t& padt, int32_t& padb, int32_t& padl, int32_t& padr) {
  std::string pad_str;
  std::vector<int32_t> pad_list;
  if (ge::AttrUtils::GetStr(op_desc, "padding", pad_str) && pad_str.compare("EXPLICIT") != 0) {
    if (pad_str.compare("SAME") == 0) {
      int32_t tails_h = ih % strh;
      int32_t tails_w = iw % strw;
      int32_t dkh = dilh * (kh - 1) + 1;
      int32_t dkw = dilw * (kw - 1) + 1;
      int32_t pad_h = std::max((tails_h > 0 ? dkh - tails_h : dkh - strh), 0);
      int32_t pad_w = std::max((tails_w > 0 ? dkw - tails_w : dkw - strw), 0);
      pad_list.push_back((pad_h >> 1));
      pad_list.push_back((pad_h >> 1) + (pad_h & 1));
      pad_list.push_back((pad_w >> 1));
      pad_list.push_back((pad_w >> 1) + (pad_w & 1));
    } else if (pad_str.compare("VALID") == 0) {
      pad_list.push_back(0);
      pad_list.push_back(0);
      pad_list.push_back(0);
      pad_list.push_back(0);
    } else {
      OP_LOGE(op_name.GetString(),
              "padding should be SAME or VALID."
              " actual is: %s.",
              pad_str.c_str());
      map<string, string> err_map;
      err_map["op_name"] = op_name.GetString();
      err_map["expected_pad_mode"] = "SAME or VALID";
      err_map["actual_pad_mode"] = pad_str;
      std::string report_error_code = "E50050";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }
    OP_LOGD(op_name.GetString(),
            "pads info is %s.", ops::to_string(pad_list).c_str());
    ge::AttrUtils::SetListInt(op_desc, "pads", pad_list);
  }

  // handle attr auto_pad from ONNX
  if (ge::AttrUtils::GetStr(op_desc, "auto_pad", pad_str)) {
    if (pad_str.compare("SAME_UPPER") == 0) {
      int32_t tails_h = ih % strh;
      int32_t tails_w = iw % strw;
      int32_t dkh = dilh * (kh - 1) + 1;
      int32_t dkw = dilw * (kw - 1) + 1;
      int32_t pad_h = std::max((tails_h > 0 ? dkh - tails_h : dkh - strh), 0);
      int32_t pad_w = std::max((tails_w > 0 ? dkw - tails_w : dkw - strw), 0);
      pad_list.push_back((pad_h >> 1));
      pad_list.push_back((pad_h >> 1) + (pad_h & 1));
      pad_list.push_back((pad_w >> 1));
      pad_list.push_back((pad_w >> 1) + (pad_w & 1));
      ge::AttrUtils::SetListInt(op_desc, "pads", pad_list);
    } else if (pad_str.compare("SAME_LOWER") == 0) {
      int32_t tails_h = ih % strh;
      int32_t tails_w = iw % strw;
      int32_t dkh = dilh * (kh - 1) + 1;
      int32_t dkw = dilw * (kw - 1) + 1;
      int32_t pad_h = std::max((tails_h > 0 ? dkh - tails_h : dkh - strh), 0);
      int32_t pad_w = std::max((tails_w > 0 ? dkw - tails_w : dkw - strw), 0);
      pad_list.push_back((pad_h >> 1) + (pad_h & 1));
      pad_list.push_back((pad_h >> 1));
      pad_list.push_back((pad_w >> 1) + (pad_w & 1));
      pad_list.push_back((pad_w >> 1));
      ge::AttrUtils::SetListInt(op_desc, "pads", pad_list);
    } else if (pad_str.compare("NOTSET") == 0) {
    } else if (pad_str.compare("VALID") == 0) {
      pad_list.push_back(0);
      pad_list.push_back(0);
      pad_list.push_back(0);
      pad_list.push_back(0);
      ge::AttrUtils::SetListInt(op_desc, "pads", pad_list);
    } else {
      OP_LOGE(op_name.GetString(),
              "padding should be SAME or VALID."
              " actual is: %s.",
              pad_str.c_str());
      map<string, string> err_map;
      err_map["op_name"] = op_name.GetString();
      err_map["expected_pad_mode"] = "NOTSET, SAME_UPPER, SAME_LOWER or VALID";
      err_map["actual_pad_mode"] = pad_str;
      std::string report_error_code = "E50050";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);

      return false;
    }
  }

  std::vector<int32_t> pads_list;
  ge::AttrUtils::GetListInt(op_desc, "pads", pads_list);
  auto p_size = pads_list.size();
  if (pads_list.empty() || p_size != kConv2dPadSizeLimit) {
    OP_LOGE(op_name.GetString(), "pads list should be 4D. actual is: %lu.", p_size);
    map<string, string> err_map;
    err_map["param_name"] = "pads";
    err_map["op_name"] = op_name.GetString();
    err_map["expected_value"] = "4D";
    err_map["input_value"] = std::to_string(p_size) + "D.";
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  padt = pads_list[kConv2dPadUpIdx];
  padb = pads_list[kConv2dPadDownIdx];
  padl = pads_list[kConv2dPadLeftIdx];
  padr = pads_list[kConv2dPadRightIdx];
  // >>> start: cut off right/bottom pad when output size is 1
  AscendString op_type = op_desc->GetType().c_str();
  auto &x_shape = op_desc->MutableInputDesc(0)->MutableShape();
  bool unknown_rank = x_shape.IsUnknownDimNum();
  bool valid = !unknown_rank && string(op_type.GetString()) == "Conv2D";
  if (valid) {
    bool cut_pad = false;
    if (ih != -1) {
      int32_t ho = (ih + padt + padb - (kh - 1) * dilh - 1) / strh + 1;
      int32_t hr = (ih + padt + padb - (kh - 1) * dilh - 1) % strh;
      cut_pad = ho == 1 && hr <= padb;
      if (cut_pad) {
        padb -= hr;
      }
    }
    if (iw != -1) {
      int32_t wo = (iw + padl + padr - (kw - 1) * dilw - 1) / strw + 1;
      int32_t wr = (iw + padl + padr - (kw - 1) * dilw - 1) % strw;
      cut_pad = wo == 1 && wr <= padr;
      if (cut_pad) {
        padr -= wr;
      }
    }
    cut_pad = pads_list[kConv2dPadDownIdx] != padb || pads_list[kConv2dPadRightIdx] != padr;
    if (cut_pad) {
      pads_list[kConv2dPadDownIdx] = padb;
      pads_list[kConv2dPadRightIdx] = padr;
      OP_LOGD(op_name.GetString(),
              "pads after cutting off is %s.", ops::to_string(pads_list).c_str());
      ge::AttrUtils::SetListInt(op_desc, "pads", pads_list);
    }
  }
  // <<< end: cut off right/bottom pad when output size is 1
  bool negative_pad = (padt < 0 || padb < 0 || padl < 0 || padr < 0);
  bool unknown_shape = x_shape.IsUnknownShape();
  if ((!unknown_shape) && (!unknown_rank) && negative_pad) {
    OP_LOGE(op_name.GetString(),
            "pads should be positive, "
            " actual is [%d,%d,%d,%d].",
            padt, padb, padl, padr);
    map<string, string> err_map;
    err_map["param_name"] = "pads";
    err_map["op_name"] = op_name.GetString();
    err_map["expected_value"] = "positive";
    err_map["input_value"] =
        std::to_string(padt) + ", " + std::to_string(padb) + ", " + std::to_string(padl) + ", " + std::to_string(padr);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  return true;
}


/*!
  * @brief Get 2D(H/W) stride and dilation params to infershape output.
  *
  * [strides]: 4D list, format sensitive, according to first input tensor format
  * [dilations]: 4D list, format sensitive
  *
  * @param op Conv2D operator.
  * @param refer Valid value reference format.
  * @param strh, strw  Input stride H/W value.
  * @param dilh, dilw  Input dilation H/W value.
  * @return bool Whether the strides, dilations settings are correct.
  */
static bool GetAttrsConv2D(const AscendString& op_name, ge::OpDescPtr& op_desc, Format refer, int32_t& strh,
                           int32_t& strw, int32_t& dilh, int32_t& dilw) {
  std::vector<int32_t> stride_list;
  ge::AttrUtils::GetListInt(op_desc, "strides", stride_list);
  auto s_size = stride_list.size();
  if (stride_list.empty() || s_size != kConv2dStridesSizeLimit) {
    OP_LOGE(op_name.GetString(), "strides list should be 4D. actual is: %lu.", s_size);
    map<string, string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = op_name.GetString();
    err_map["expected_value"] = "4D";
    err_map["input_value"] = std::to_string(s_size) + "D.";
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  std::vector<int32_t> dilation_list;
  ge::AttrUtils::GetListInt(op_desc, "dilations", dilation_list);
  auto d_size = dilation_list.size();
  if (dilation_list.empty() || d_size != kConv2dDilationSizeLimit) {
    OP_LOGE(op_name.GetString(), "dilations list should be 4D. actual is: %lu.", d_size);
    map<string, string> err_map;
    err_map["param_name"] = "dilations";
    err_map["op_name"] = op_name.GetString();
    err_map["expected_value"] = "4D";
    err_map["input_value"] = std::to_string(d_size) + "D.";
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (refer == FORMAT_NCHW) {
    strh = stride_list[kHDimNCHWIdx];
    strw = stride_list[kWDimNCHWIdx];
    dilh = dilation_list[kHDimNCHWIdx];
    dilw = dilation_list[kWDimNCHWIdx];
  } else if (refer == FORMAT_NHWC) {
    strh = stride_list[kHDimNHWCIdx];
    strw = stride_list[kWDimNHWCIdx];
    dilh = dilation_list[kHDimNHWCIdx];
    dilw = dilation_list[kWDimNHWCIdx];
  }
  if (strh <= 0 || strw <= 0) {
    OP_LOGE(op_name.GetString(),
            "strides should be positive,"
            " actual is [%d,%d].",
            strh, strw);
    map<string, string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = op_name.GetString();
    err_map["expected_value"] = "positive";
    err_map["input_value"] = std::to_string(strh) + ", " + std::to_string(strw);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  if (dilh <= 0 || dilw <= 0) {
    OP_LOGE(op_name.GetString(),
            "dilations should be positive,"
            " actual is [%d,%d].",
            dilh, dilw);
    map<string, string> err_map;
    err_map["param_name"] = "dilations";
    err_map["op_name"] = op_name.GetString();
    err_map["expected_value"] = "positive";
    err_map["input_value"] = std::to_string(dilh) + ", " + std::to_string(dilw);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  return true;
}

static void GetConv2dOutShapeRange(const std::string& pad_str,
                                   size_t idx,
                                   int32_t stride, int32_t dilation, int32_t pad, int32_t kernel,
                                   const std::vector<std::pair<int64_t, int64_t>>& fm_range,
                                   std::vector<std::pair<int64_t, int64_t>>& out_range) {
  int32_t low = fm_range[idx].first;
  int32_t high = fm_range[idx].second;
  if (stride <= 0) {
    OP_LOGE("GetConv2dOutShapeRange", "diviser stride must > 0");
    return;
  }
  if (pad_str == "SAME") {
    out_range[idx].first = (low + stride -1) / stride;
    out_range[idx].second = (high + stride -1) / stride;
  } else {
    out_range[idx].first = (low + pad - dilation * (kernel - 1) - 1) / stride + 1;
    out_range[idx].second = (high + pad - dilation * (kernel - 1) - 1) / stride + 1;
  }
  out_range[idx].first = std::max(out_range[idx].first, kDynamicRangeLowerBound);
  out_range[idx].second = std::min(out_range[idx].second, kDynamicRangeUpperBound);
  if (high == -1) {
    out_range[idx].second = high;
  }
}

static bool SetConv2dOutShapeRange(const AscendString& op_name,
                                   ge::OpDescPtr& op_desc,
                                   const vector<int32_t>& attr_params,
                                   ge::GeShape& y_shape,
                                   ge::GeTensorDescPtr& x_tensor,
                                   ge::GeTensorDescPtr& y_tensor) {
  auto &x_shape = x_tensor->MutableShape();
  auto x_format = x_tensor->GetFormat();

  size_t idx = 0;
  int32_t strh = attr_params[idx++];
  int32_t strw = attr_params[idx++];
  int32_t dilh = attr_params[idx++];
  int32_t dilw = attr_params[idx++];
  int32_t padh = attr_params[idx++];
  int32_t padw = attr_params[idx++];
  int32_t kn = attr_params[idx++];
  int32_t kh = attr_params[idx++];
  int32_t kw = attr_params[idx++];

  size_t idx_h = 0;
  size_t idx_w = 0;
  size_t idx_c = 0;
  if (x_format == FORMAT_NHWC) {
    idx_h = kHDimNHWCIdx;
    idx_w = kWDimNHWCIdx;
    idx_c = kCDimNHWCIdx;
  } else if (x_format == FORMAT_NCHW) {
    idx_c = kCDimNCHWIdx;
    idx_h = kHDimNCHWIdx;
    idx_w = kWDimNCHWIdx;
  }

  // update pads if padding is SAME
  std::string pad_str;
  if (x_shape.GetDimNum() == kConv2dInputSizeLimit && ge::AttrUtils::GetStr(op_desc, "padding", pad_str) &&
      pad_str == "SAME" && (x_shape.GetDim(idx_h) == -1 or x_shape.GetDim(idx_w) == -1)) {
    ge::AttrUtils::SetListInt(op_desc, "pads", {-1, -1, -1, -1});
    OP_LOGD(op_name.GetString(), "set pads to {-1, -1, -1, -1} when padding is SAME in dynamic_shape");
  }

  OP_LOGD(op_name.GetString(), "dynamic shape set range");
  std::vector<std::pair<int64_t, int64_t>> fm_range;
  x_tensor->GetShapeRange(fm_range);
  if (x_shape.GetDim(idx_h) == -1) {
    y_shape.SetDim(idx_h, -1);
  }
  if (x_shape.GetDim(idx_w) == -1) {
    y_shape.SetDim(idx_w, -1);
  }
  if (!fm_range.empty()) {
    for (size_t i = 0; i < fm_range.size(); i++) {
      OP_LOGD(op_name.GetString(), "fmap Range[%u] is (%lld, %lld)", i, fm_range[i].first, fm_range[i].second);
    }

    std::vector<std::pair<int64_t, int64_t>> out_range(fm_range);
    out_range[idx_c] = std::make_pair((int64_t)kn, (int64_t)kn);
    out_range[idx_h] = std::make_pair(y_shape.GetDim(idx_h), y_shape.GetDim(idx_h));
    out_range[idx_w] = std::make_pair(y_shape.GetDim(idx_w), y_shape.GetDim(idx_w));
    if (x_shape.GetDim(idx_h) == -1) {
      GetConv2dOutShapeRange(pad_str, idx_h, strh, dilh, padh, kh, fm_range, out_range);
    }
    if (x_shape.GetDim(idx_w) == -1) {
      GetConv2dOutShapeRange(pad_str, idx_w, strw, dilw, padw, kw, fm_range, out_range);
    }
    y_tensor->SetShapeRange(out_range);
    for (size_t i = 0; i < out_range.size(); i++) {
      OP_LOGD(op_name.GetString(), "output Range[%u] is (%lld, %lld)", i, out_range[i].first, out_range[i].second);
    }
  }
  return true;
}

static graphStatus CheckConv2DBias(const AscendString& op_name, const OpDescPtr& op_desc, int32_t outChannel)
{
  auto bias_tensor = op_desc->MutableInputDesc(2); // bias is the third value of input
  if (bias_tensor == nullptr) {
    return GRAPH_SUCCESS;
  }

  auto &bias_shape = bias_tensor->MutableShape();
  size_t bias_dim_num = bias_shape.GetDimNum();
  if (bias_dim_num != 1 && bias_dim_num != 4) { // bias shape should be 1D or 4D
    OP_LOGE(op_name.GetString(), "input bias shape should be 1D or 4D.");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "input bias shape should be 1D or 4D.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
 // The index of the C dimension in 4D, default value is 0
  size_t idx_c = 0;
  if (bias_dim_num == 4) {  // bias shape is 4D
    auto bias_format = bias_tensor->GetFormat();
    if (bias_format == FORMAT_NCHW) {
      idx_c = 1;
    } else if (bias_format == FORMAT_NHWC) {
      idx_c = 3; // The 4th index of the C dimension in 4D
    } else {
      OP_LOGE(op_name.GetString(), "input bias format should be NCHW or NHWC when shape is 4D.");
      map<string, string> err_map;
      err_map["op_name"] = op_name.GetString();
      err_map["description"] = "input bias format should be NCHW or NHWC when shape is 4D.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  }

  for (size_t i = 0; i < bias_dim_num; i++) {
    if (i == idx_c) {
      if (bias_shape.GetDim(i) != outChannel) {
        OP_LOGE(op_name.GetString(), "input bias size of dim_c should be equal to out_channels.");
        map<string, string> err_map;
        err_map["op_name"] = op_name.GetString();
        err_map["description"] = "input bias size of dim_c should be equal to out_channels.";
        std::string report_error_code = "E50060";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
      }

      continue;
    }

    if (bias_shape.GetDim(i) != 1) {
        OP_LOGE(op_name.GetString(), "input bias size of dim[N, H, W] should be equal to 1.");
        map<string, string> err_map;
        err_map["op_name"] = op_name.GetString();
        err_map["description"] = "input bias size of dim[N, H, W] should be equal to 1.";
        std::string report_error_code = "E50060";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

/*!
  * @brief Infer output shape and dtype, dtype is same to first input tensor, Output
  *        format is set by ge parser process already.
  * @param Conv2DInfer Conv2D infershape function.
  * @return Status The processing flow result.
  */
IMPLEMT_INFERFUNC(Conv2D, Conv2DInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);

  AscendString op_name = op_desc->GetName().c_str();
  PROFILING_PROTO_INIT(op_name.GetString());
  OP_LOGD(op_name.GetString(), "Enter Conv2DInfer.");

  auto x_tensor = op_desc->MutableInputDesc(0);
  auto w_tensor = op_desc->MutableInputDesc(1);
  CHECK_PTR_NULL(x_tensor, "tensor x desc", return GRAPH_FAILED);
  CHECK_PTR_NULL(w_tensor, "tensor filter desc", return GRAPH_FAILED);

  auto &x_shape = x_tensor->MutableShape();
  auto &w_shape = w_tensor->MutableShape();
  bool unknown_rank = x_shape.IsUnknownDimNum();
  if ((!unknown_rank && x_shape.GetDimNum() != 4) || w_shape.GetDimNum() != 4) {
    OP_LOGE(op_name.GetString(), "x_shape dimnum is %lu", x_shape.GetDimNum());
    return GRAPH_FAILED;
  }
  auto x_format = x_tensor->GetFormat();
  auto w_format = w_tensor->GetFormat();
  CHECK_OP_FUNC(FORMAT_RESERVED == x_format, return GRAPH_FAILED, "get format failed: %d", x_format);
  CHECK_OP_FUNC(FORMAT_RESERVED == w_format, return GRAPH_FAILED, "get format failed: %d", w_format);

  int32_t in = -1;
  int32_t ic = -1;
  int32_t ih = -1;
  int32_t iw = -1;
  int32_t kn = 0;
  int32_t kc = 0;
  int32_t kh = 0;
  int32_t kw = 0;
  if (x_format == FORMAT_NCHW) {
    if (!unknown_rank) {
      in = x_shape.GetDim(0);
      ic = x_shape.GetDim(1);
      ih = x_shape.GetDim(2);
      iw = x_shape.GetDim(3);
    }
  } else if (x_format == FORMAT_NHWC) {
    if (!unknown_rank) {
      in = x_shape.GetDim(0);
      ic = x_shape.GetDim(3);
      ih = x_shape.GetDim(1);
      iw = x_shape.GetDim(2);
    }
  } else {
    OP_LOGE(op_name.GetString(),
            "input x format should be NCHW or NHWC."
            " actual is: %s",
            TypeUtils::FormatToSerialString(x_format).c_str());
    map<string, string> err_map;
    err_map["param"] = "x";
    err_map["op_name"] = op_name.GetString();
    err_map["expected_format_list"] = "NCHW or NHWC";
    err_map["format"] = TypeUtils::FormatToSerialString(x_format);
    std::string report_error_code = "E50002";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (w_format == FORMAT_NCHW) {
    kn = w_shape.GetDim(0);
    kc = w_shape.GetDim(1);
    kh = w_shape.GetDim(2);
    kw = w_shape.GetDim(3);
  } else if (w_format == FORMAT_NHWC) {
    kn = w_shape.GetDim(0);
    kc = w_shape.GetDim(3);
    kh = w_shape.GetDim(1);
    kw = w_shape.GetDim(2);
  } else if (w_format == FORMAT_HWCN) {
    kn = w_shape.GetDim(3);
    kc = w_shape.GetDim(2);
    kh = w_shape.GetDim(0);
    kw = w_shape.GetDim(1);
  } else {
    OP_LOGE(op_name.GetString(),
            "input filter format should be NCHW, NHWC or HWCN."
            " actual is: %s",
            TypeUtils::FormatToSerialString(w_format).c_str());
    map<string, string> err_map;
    err_map["param"] = "filter";
    err_map["op_name"] = op_name.GetString();
    err_map["expected_format_list"] = "NCHW, NHWC or HWCN";
    err_map["format"] = TypeUtils::FormatToSerialString(w_format);
    std::string report_error_code = "E50002";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (CheckConv2DBias(op_name, op_desc, kn) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  // set data_format: copy value of x_format to data_format
  // data_format will be used to infer position of H/W
  // in strides and dilations(previously used ori_format)
  std::string data_format;
  std::string attr_data_format = "data_format";
  std::string data_format_NCHW = "NCHW";
  std::string data_format_NHWC = "NHWC";

  if (ge::AttrUtils::GetStr(op_desc, attr_data_format, data_format)) {
    OP_LOGI(data_format.c_str(), "conv before set data_format");
  }

  if (x_format == ge::FORMAT_NCHW) {
    ge::AttrUtils::SetStr(op_desc, attr_data_format, data_format_NCHW);
  } else {
    ge::AttrUtils::SetStr(op_desc, attr_data_format, data_format_NHWC);
  }

  ge::AttrUtils::GetStr(op_desc, attr_data_format, data_format);
  OP_LOGI(data_format.c_str(), "conv after set data_format");

  int64_t groups = 1;
  ge::AttrUtils::GetInt(op_desc, "groups", groups);
  bool is_dynamic = false;
  // when static op or dynamic op phase_running, is_dynamic == False
  if (x_shape.IsUnknownShape() && (!x_shape.IsUnknownDimNum())) {
    is_dynamic = true;
  }
  if (is_dynamic && (ic == -1)) {
    ic = kc * groups;
    OP_LOGW(op_name.GetString(),
            "input x channel is unknow, fixed channel = %d, "
            "in_channels should be kc*grous[%d * %d]", (int)ic, (int)kc, (int)groups);
  }
  if ((!unknown_rank) && (groups == 1)) {
    if ((ic > 0) && (ic % kc == 0)) {
      groups = ic / kc;
      ge::AttrUtils::SetInt(op_desc, "groups", groups);
      OP_LOGD(op_name.GetString(), "parameter groups is implicitly changed.");
    } else {
      OP_LOGE(op_name.GetString(), "in_channels(>0) should be divisible by kernel_channels when groups = 1.");
      map<string, string> err_map;
      err_map["op_name"] = op_name.GetString();
      err_map["description"] = "in_channels(>0) should be divisible by kernel_channels when groups = 1.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  }
  if ((!unknown_rank) && (ic != kc * groups)) {
    OP_LOGE(op_name.GetString(),
            "x channel should be equal to filter channel*groups. "
            "x format is: %s, filter format is: %s, "
            "x shape is: [%ld,%ld,%ld,%ld], filter shape is: [%ld,%ld,%ld,%ld], "
            "groups is: %d.",
            TypeUtils::FormatToSerialString(x_format).c_str(), TypeUtils::FormatToSerialString(w_format).c_str(),
            x_shape.GetDim(0), x_shape.GetDim(1), x_shape.GetDim(2),
            x_shape.GetDim(3), w_shape.GetDim(0), w_shape.GetDim(1),
            w_shape.GetDim(2), w_shape.GetDim(3), (int)groups);
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["x_shape"] = std::to_string(x_shape.GetDim(0)) + ", " + std::to_string(x_shape.GetDim(1)) + ", " +
                         std::to_string(x_shape.GetDim(2)) + ", " + std::to_string(x_shape.GetDim(3));
    err_map["filter_shape"] = std::to_string(w_shape.GetDim(0)) + ", " + std::to_string(w_shape.GetDim(1)) + ", " +
                              std::to_string(w_shape.GetDim(2)) + ", " + std::to_string(w_shape.GetDim(3));
    err_map["groups"] = std::to_string(groups);

    std::string report_error_code = "E50059";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  if (kn % groups != 0) {
    OP_LOGE(op_name.GetString(), "out_channels should be divisible by groups.");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "out_channels should be divisible by groups.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  PROFILING_PROTO_AFTER_GET_SHAPE_REG()
  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;
  int32_t padt = 0;
  int32_t padb = 0;
  int32_t padl = 0;
  int32_t padr = 0;
  if (!GetAttrsConv2D(op_name, op_desc, x_format, strh, strw, dilh, dilw) ||
      !GetPadConv2D(op_name, op_desc, ih, iw, kh, kw, strh, strw, dilh, dilw, padt, padb, padl, padr)) {
    return GRAPH_FAILED;
  }

  int64_t ihPad = (ih + padt + padb - dilh * (kh - 1) - 1);
  int64_t iwPad = (iw + padl + padr - dilw * (kw - 1) - 1);
  int64_t oh = ihPad / strh + 1;
  int64_t ow = iwPad / strw + 1;
  if (unknown_rank) {
    oh = -1;
    ow = -1;
  }
  if ((ih > 0) && (kh > 0) && (iw > 0) && (kw > 0)) {
    if ((ihPad < 0) || (iwPad < 0)) {
      OP_LOGE(op_name.GetString(), "image size after padding should be greater than or equal to filter size.");
      map<string, string> err_map;
      err_map["op_name"] = op_name.GetString();
      err_map["description"] = "image size after padding should be greater than or equal to filter size.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    }
  }
  PROFILING_PROTO_AFTER_INFER_SHAPE_REG()
  auto y_tensor = op_desc->MutableOutputDesc(0);
  auto y_format = y_tensor->GetFormat();
  auto &y_shape = y_tensor->MutableShape();
  y_shape.SetDimNum(4);
  CHECK_OP_FUNC(FORMAT_RESERVED == y_format, return GRAPH_FAILED, "get format failed: %d", y_format)
  if (y_format == FORMAT_NCHW) {
    y_shape.SetDim(0, in);
    y_shape.SetDim(1, kn);
    y_shape.SetDim(2, oh);
    y_shape.SetDim(3, ow);
  } else if (y_format == FORMAT_NHWC) {
    y_shape.SetDim(0, in);
    y_shape.SetDim(1, oh);
    y_shape.SetDim(2, ow);
    y_shape.SetDim(3, kn);
  } else {
    OP_LOGE(op_name.GetString(),
            "output y format should be NCHW or NHWC."
            " actual is: %s",
            TypeUtils::FormatToSerialString(y_format).c_str());
    map<string, string> err_map;
    err_map["param"] = "y";
    err_map["op_name"] = op_name.GetString();
    err_map["expected_format_list"] = "NCHW or NHWC";
    err_map["format"] = TypeUtils::FormatToSerialString(y_format);
    std::string report_error_code = "E50002";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  auto x_dtype = x_tensor->GetDataType();
  if (x_dtype == ge::DT_INT8) {
    y_tensor->SetDataType(ge::DT_INT32);
  } else if (x_dtype == ge::DT_INT4) {
    y_tensor->SetDataType(ge::DT_INT32);
  } else {
    y_tensor->SetDataType(x_dtype);
  }

  // fuzz_build switch
  bool fuzz_build = false;
  ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_FUZZ_BUILD.c_str(), fuzz_build);

  // set Range
  if (is_dynamic) {
    OP_LOGD(op_name.GetString(), "start accurate build.");
    vector<int32_t> attr_params = {strh, strw, dilh, dilw,
                                   padt + padb, padl + padr,
                                   kn, kh, kw};
    if (!SetConv2dOutShapeRange(op_name, op_desc, attr_params, y_shape, x_tensor, y_tensor)) {
      return GRAPH_FAILED;
    }
  }
  // fuzz build allow shape dim -1 with range
  if ((!unknown_rank) && fuzz_build) {
    OP_LOGD(op_name.GetString(), "start fuzz build.");
    // change pad to -1 when padding is SAME
    std::string pad_str;
    ge::AttrUtils::GetStr(op_desc, "padding", pad_str);
    if (pad_str == "SAME") {
      ge::AttrUtils::SetListInt(op_desc, "pads", {-1, -1, -1, -1});
      OP_LOGD(op_name.GetString(), "set pads to {-1, -1, -1, -1} when padding is SAME in fuzzy build");
    }
  }
  OP_LOGD(op_name.GetString(), "Leave Conv2DInfer.");
  PROFILING_PROTO_END()
  return GRAPH_SUCCESS;
}

/*!
  * @brief Verify the required 2 input tensor, optional bias ignored, verify
  *        strides and dilations attrs, pads ignored.
  * @param Conv2D Operator type.
  * @param Conv2DVerify Input validity check function.
  * @return Status The processing flow result.
  */
IMPLEMT_VERIFIER(Conv2D, Conv2DVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter Conv2DVerify.");
  auto x_tensor = op.GetInputDesc(0);
  auto w_tensor = op.GetInputDesc(1);
  auto x_shape = x_tensor.GetOriginShape().GetDims();
  auto w_shape = w_tensor.GetOriginShape().GetDims();
  auto offset_w_tensor = op.GetInputDesc(3);

  if (offset_w_tensor.GetOriginShape().GetDims().size() != 0) {
    OP_LOGE(op_name.GetString(), "input offset_w is not supported.");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "input offset_w is not supported.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  bool unknown_rank = IsUnknownRankShape(x_shape);
  if ((!unknown_rank) && (x_shape.size() != 4)) {
    if (x_shape.size() == 0) {
      OP_LOGE(op_name.GetString(), "input x shape is empty.");
      map<string, string> err_map;
      err_map["op_name"] = op_name.GetString();
      err_map["description"] = "input x shape is empty.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    } else {
      OP_LOGE(op_name.GetString(), "input x shape should be 4D.");
      map<string, string> err_map;
      err_map["op_name"] = op_name.GetString();
      err_map["description"] = "input x shape should be 4D.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    }
    return GRAPH_FAILED;
  }
  if (w_shape.size() != 4) {
    if (w_shape.size() == 0) {
      OP_LOGE(op_name.GetString(), "input filter shape is empty.");
      map<string, string> err_map;
      err_map["op_name"] = op_name.GetString();
      err_map["description"] = "input filter shape is empty.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    } else {
      OP_LOGE(op_name.GetString(), "input filter shape should be 4D.");
      map<string, string> err_map;
      err_map["op_name"] = op_name.GetString();
      err_map["description"] = "input filter shape should be 4D.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    }
    return GRAPH_FAILED;
  }

  std::vector<int32_t> stride_list;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stride_list)) {
    OP_LOGE(op_name.GetString(), "get strides list failed.");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "get strides list failed.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  std::vector<int32_t> dilation_list;
  if (GRAPH_SUCCESS != op.GetAttr("dilations", dilation_list)) {
    OP_LOGE(op_name.GetString(), "get dilations list failed.");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "get dilations list failed.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  OP_LOGD(op_name.GetString(), "Leave Conv2DVerify.");
  return GRAPH_SUCCESS;
}

/*!
  * @brief provide Conv2D operator slice data
  * @param Conv2D Operator type.
  * @param Conv2DInferDataSlice slice data function
  * @return Status The processing flow result.
  */
IMPLEMT_INFER_DATA_SLICE(Conv2D, Conv2DInferDataSlice) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter Conv2D InferDataSlice");
  // get input h/w, filter h/w, pad_h,pad_w, stride_h, stride_w, dilation_h,dilation_w
  auto x_tensor = op.GetInputDesc(0);
  auto w_tensor = op.GetInputDesc(1);

  auto x_shape = x_tensor.GetOriginShape().GetDims();
  auto w_shape = w_tensor.GetOriginShape().GetDims();

  auto x_format = x_tensor.GetOriginFormat();
  auto w_format = w_tensor.GetOriginFormat();

  CHECK(IsUnknownRankShape(x_shape),
        CUBE_INNER_ERR_REPORT(op_name.GetString(), "input x shape [-2] do not support split."),
        return GRAPH_FAILED);

  std::vector<int32_t> stride_list;
  std::vector<int32_t> dilation_list;
  std::vector<int32_t> pad_list;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stride_list) || GRAPH_SUCCESS != op.GetAttr("dilations", dilation_list)
      || GRAPH_SUCCESS != op.GetAttr("pads", pad_list)) {
    return GRAPH_FAILED;
  }
  CHECK(pad_list.size() < 4, CUBE_INNER_ERR_REPORT(op_name.GetString(), "pads size less then 4."),
    return GRAPH_FAILED);

  int32_t ih = 0;
  int32_t iw = 0;
  int32_t kh = 0;
  int32_t kw = 0;
  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;
  int32_t padt = pad_list[0];
  int32_t padl = pad_list[2];

  if (x_format == FORMAT_NCHW) {
    ih = x_shape[2];
    iw = x_shape[3];
    strh = stride_list[2];
    strw = stride_list[3];
    dilh = dilation_list[2];
    dilw = dilation_list[3];
  } else if (x_format == FORMAT_NHWC) {
    ih = x_shape[1];
    iw = x_shape[2];
    strh = stride_list[1];
    strw = stride_list[2];
    dilh = dilation_list[1];
    dilw = dilation_list[2];
  } else {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "x format is valid, the error x format is: %d", x_format);
    return GRAPH_FAILED;
  }

  if (w_format == FORMAT_NCHW) {
    kh = w_shape[2];
    kw = w_shape[3];
  } else if (w_format == FORMAT_NHWC) {
    kh = w_shape[1];
    kw = w_shape[2];
  } else if (w_format == FORMAT_HWCN) {
    kh = w_shape[0];
    kw = w_shape[1];
  } else {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "weight format is valid, the error w format is: %d", w_format);
    return GRAPH_FAILED;
  }

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
  vector<vector<int64_t>> y_data_slice;
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}};
  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op_name.GetString(), "no data slice, not need infer input");
    return GRAPH_FAILED;
  }
  bool have_slice = false;
  vector<int> new_pad_lists = pad_list;
  for (size_t i = 0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() > 0) {
      have_slice = true;
      if (i == 2) {
        CHECK(ih == -1,
              CUBE_INNER_ERR_REPORT(op_name.GetString(), "input x dynamic h do not support split."),
              return GRAPH_FAILED);
        vector<int64_t> ih_slice;
        int64_t top_add_pad = pad_list[0];
        int64_t bom_add_pad = pad_list[1];
        InferHWConv2D(ih, kh, padt, strh, dilh, y_data_slice[i], ih_slice, &top_add_pad, &bom_add_pad);
        OP_LOGD(op_name.GetString(), "conv2d h axis slice ori_scope is [%d,%d], output scope is [%d,%d]",
                ih_slice[0], ih_slice[1], y_data_slice[i][0], y_data_slice[i][1]);
        new_pad_lists[0] = top_add_pad;
        new_pad_lists[1] = bom_add_pad;
        x_data_slice[i] = ih_slice;
      } else if (i == 3) {
        CHECK(iw == -1,
              CUBE_INNER_ERR_REPORT(op_name.GetString(), "input x dynamic w do not support split."),
              return GRAPH_FAILED);
        vector<int64_t> iw_slice;
        int64_t left_add_pad = pad_list[2];
        int64_t right_add_pad = pad_list[3];
        InferHWConv2D(iw, kw, padl, strw, dilw, y_data_slice[i], iw_slice, &left_add_pad, &right_add_pad);
        OP_LOGD(op_name.GetString(), "conv2d w axis slice ori_scope is [%d,%d], output scope is [%d,%d]",
                iw_slice[0], iw_slice[1], y_data_slice[i][0], y_data_slice[i][1]);
        new_pad_lists[2] = left_add_pad;
        new_pad_lists[3] = right_add_pad;
        x_data_slice[i] = iw_slice;
      } else {
        bool is_dyn = (i == 0) && (x_shape[0] == -1);
        vector<int64_t> dyn_slice = {-1, -1};
        x_data_slice[i] = is_dyn ? dyn_slice : y_data_slice[i];
      }
    }
  }
  op.SetAttr("pads", new_pad_lists);
  OP_LOGD(op_name.GetString(), "conv2d new pad lists is [%d,%d,%d,%d]", new_pad_lists[0],
          new_pad_lists[1], new_pad_lists[2], new_pad_lists[3]);

  if (!have_slice) {
    return GRAPH_FAILED;
  }
  if (!AttrUtils::SetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
    return GRAPH_FAILED;
  }
  OP_LOGD(op_name.GetString(), "Calc Conv2D InferDataSlice end!");
  return GRAPH_SUCCESS;
}

INFER_DATA_SLICE_FUNC_REG(Conv2D, Conv2DInferDataSlice);
INFER_FUNC_REG(Conv2D, Conv2DInfer);
VERIFY_FUNC_REG(Conv2D, Conv2DVerify);

// --------------------------Conv2DCompress------------------------------

/*
 * Infer output shape and dtype, dtype is same to first input tensor
 * Output format is set by ge parser process already
 */
IMPLEMT_INFERFUNC(Conv2DCompress, Conv2DCompressInfer) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter Conv2DCompressInfer.");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);
  auto xTensor = op.get_input_desc_x();
  auto wTensor = op.get_input_desc_filter_compress();

  auto xShape = xTensor.GetShape().GetDims();
  auto wShape = wTensor.GetShape().GetDims();
  auto xFormat = xTensor.GetFormat();
  auto wFormat = wTensor.GetFormat();
  CHECK_OP_FUNC(FORMAT_RESERVED == xFormat, return GRAPH_FAILED, "get format failed: %d", xFormat);
  CHECK_OP_FUNC(FORMAT_RESERVED == wFormat, return GRAPH_FAILED, "get format failed: %d", wFormat);

  int32_t in = 0;
  int32_t ic = 0;
  int32_t ih = 0;
  int32_t iw = 0;
  int32_t kn = 0;
  int32_t kc = 0;
  int32_t kh = 0;
  int32_t kw = 0;
  if (xFormat == FORMAT_NCHW && xShape.size() == kDeformDimSizeLimit) {
    in = xShape[0];
    ic = xShape[1];
    ih = xShape[2];
    iw = xShape[3];
  } else if (xFormat == FORMAT_NHWC && xShape.size() == kDeformDimSizeLimit) {
    in = xShape[0];
    ic = xShape[3];
    ih = xShape[1];
    iw = xShape[2];
  } else {
    CUBE_INNER_ERR_REPORT(op_name.GetString(),
                          "input x format should be NCHW or NHWC and the dim of x_shape should be 4.");
    return GRAPH_FAILED;
  }

  if (wFormat == FORMAT_NCHW && wShape.size() == kDeformDimSizeLimit) {
    kn = wShape[0];
    kc = wShape[1];
    kh = wShape[2];
    kw = wShape[3];
  } else if (wFormat == FORMAT_NHWC && wShape.size() == kDeformDimSizeLimit) {
    kn = wShape[0];
    kc = wShape[3];
    kh = wShape[1];
    kw = wShape[2];
  } else if (wFormat == FORMAT_HWCN && wShape.size() == kDeformDimSizeLimit) {
    kn = wShape[3];
    kc = wShape[2];
    kh = wShape[0];
    kw = wShape[1];
  } else {
    CUBE_INNER_ERR_REPORT(op_name.GetString(),
                         "input filter format should be NCHW, NHWC or HWCN and the dim of w_shape should be 4.");
    return GRAPH_FAILED;
  }

  // set data_format: copy value of xFormat to data_format
  // data_format will be used to infer position of H/W
  // in strides and dilations(previously used ori_format)
  std::string data_format;
  std::string attr_data_format = "data_format";
  std::string expected_data_format = (xFormat == ge::FORMAT_NCHW) ? "NCHW" : "NHWC";

  if (GRAPH_SUCCESS == op.GetAttr(attr_data_format, data_format)) {
    OP_LOGI(data_format.c_str(), "conv compress before set data_format");
  }

  op.SetAttr(attr_data_format, expected_data_format);

  op.GetAttr(attr_data_format, data_format);
  OP_LOGI(data_format.c_str(), "conv compress after set data_format");

  int64_t groups = 1;
  if (GRAPH_SUCCESS != op.GetAttr("groups", groups)) {
    OP_LOGI(op_name.GetString(), "no groups setting, use groups as 1");
  }
  OP_LOGI(op_name.GetString(), "groups is %lld", groups);

  CHECK_OP_FUNC(ic != kc * groups, return GRAPH_FAILED, "input x channel should be equal to filter.");

  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;
  int32_t padt = 0;
  int32_t padb = 0;
  int32_t padl = 0;
  int32_t padr = 0;
  CHECK_OP_FUNC(false == GetAttrsConv2D(op_name, op_desc, xFormat, strh, strw, dilh, dilw),
                return GRAPH_FAILED, "get attrs failed.");
  CHECK_OP_FUNC(false == GetPadConv2D(op_name, op_desc, ih, iw, kh, kw, strh, strw, dilh, dilw, padt, padb, padl, padr),
                return GRAPH_FAILED, "get pads attrs failed.");

  int64_t oh = (ih + padt + padb - dilh * (kh - 1) - 1) / strh + 1;
  int64_t ow = (iw + padl + padr - dilw * (kw - 1) - 1) / strw + 1;

  vector<int64_t> yShape;
  auto yTensor = op.get_output_desc_y();
  auto yFormat = yTensor.GetFormat();
  CHECK_OP_FUNC(FORMAT_RESERVED == yFormat, return GRAPH_FAILED, "get format failed: %d", yFormat)
  if (yFormat == FORMAT_NCHW) {
    yShape.push_back(in);
    yShape.push_back(kn);
    yShape.push_back(oh);
    yShape.push_back(ow);
  } else if (yFormat == FORMAT_NHWC) {
    yShape.push_back(in);
    yShape.push_back(oh);
    yShape.push_back(ow);
    yShape.push_back(kn);
  } else {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "output y format should be NCHW or NHWC.");
    return GRAPH_FAILED;
  }
  yTensor.SetShape(Shape(yShape));
  auto xDtype = xTensor.GetDataType();
  auto yDtype = xDtype;
  if (xDtype == ge::DT_INT8) {
    yDtype = ge::DT_INT32;
  }
  yTensor.SetDataType(yDtype);
  CHECK_OP_FUNC(GRAPH_SUCCESS != op.update_output_desc_y(yTensor), return GRAPH_FAILED, "update output desc failed.");

  OP_LOGD(op_name.GetString(), "Leave Conv2DCompressInfer.");
  return GRAPH_SUCCESS;
}

/*
 * Verify the required 2 input tensor, optional bias ignored
 * Verify strides and dilations attrs, pads ignored
 */
IMPLEMT_VERIFIER(Conv2DCompress, Conv2DCompressVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter Conv2DCompressVerify.");
  auto xTensor = op.get_input_desc_x();
  auto wTensor = op.get_input_desc_filter_compress();

  auto xShape = xTensor.GetShape().GetDims();
  auto wShape = wTensor.GetShape().GetDims();

  if (xShape.size() != 4) {
    if (xShape.size() == 0) {
      CUBE_INNER_ERR_REPORT(op_name.GetString(), "input x shape is empty.");
    } else {
      CUBE_INNER_ERR_REPORT(op_name.GetString(), "only support 2D compress convolution.");
    }
    return GRAPH_FAILED;
  }
  if (wShape.size() != 4) {
    if (wShape.size() == 0) {
      CUBE_INNER_ERR_REPORT(op_name.GetString(), "input filter_compress shape is empty.");
    } else {
      CUBE_INNER_ERR_REPORT(op_name.GetString(), "only support 2D compress convolution.");
    }
    return GRAPH_FAILED;
  }

  auto xDtype = xTensor.GetDataType();
  auto wDtype = wTensor.GetDataType();

  CHECK_OP_FUNC(xDtype != wDtype, return GRAPH_FAILED, "input x dtype is differ from filter_compress dtype."
                                                        " actual x dtype is: %d filter dtype is: %d",
                                                        (int)xDtype, (int)wDtype)

  std::vector<int32_t> strideList;
  CHECK_OP_FUNC(GRAPH_SUCCESS != op.GetAttr("strides", strideList), return GRAPH_FAILED, "get strides list failed.");
  std::vector<int32_t> dilationList;
  CHECK_OP_FUNC(GRAPH_SUCCESS != op.GetAttr("dilations", dilationList),
                return GRAPH_FAILED, "get dilations list failed.");

  OP_LOGD(op_name.GetString(), "Leave Conv2DCompressVerify.");
  return GRAPH_SUCCESS;
}

/*!
 * @brief provide Conv2DCompress operator slice data
 * @param Conv2DCompress Operator type
 * @param Conv2DCompressInferDataSlice slice data function
 * @return Status The processing flow result.
 */
IMPLEMT_INFER_DATA_SLICE(Conv2DCompress, Conv2DCompressInferDataSlice) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);
  OP_LOGD(op_name.GetString(), "Enter Conv2DCompress InferDataSlice");
  // get input h/w, filter h/w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w
  auto x_tensor = op.GetInputDescByName("x");
  auto w_tensor = op.GetInputDescByName("filter_compress");
  auto x_shape = x_tensor.GetOriginShape().GetDims();
  auto w_shape = w_tensor.GetOriginShape().GetDims();

  static uint32_t TENSOR_SHAPE_SIZE = 4;
  static uint32_t ATTR_SHAPE_SIZE = 4;
  bool bRet = x_shape.size() < TENSOR_SHAPE_SIZE || w_shape.size() < TENSOR_SHAPE_SIZE;
  CHECK(bRet, CUBE_INNER_ERR_REPORT(op_name.GetString(), "tensor size less then 4."), return GRAPH_FAILED);

  auto x_format = x_tensor.GetOriginFormat();
  auto w_format = w_tensor.GetOriginFormat();

  std::vector<int32_t> stride_list;
  std::vector<int32_t> dilation_list;
  std::vector<int32_t> pad_list;
  bRet = op.GetAttr("strides", stride_list) != GRAPH_SUCCESS || \
         op.GetAttr("dilations", dilation_list) != GRAPH_SUCCESS || \
         op.GetAttr("pads", pad_list) != GRAPH_SUCCESS;
  CHECK(bRet, CUBE_INNER_ERR_REPORT(op_name.GetString(), "get attr failed."), return GRAPH_FAILED);
  bRet = stride_list.size() < ATTR_SHAPE_SIZE || \
         dilation_list.size() < ATTR_SHAPE_SIZE || \
         pad_list.size() < ATTR_SHAPE_SIZE;
  CHECK(bRet, CUBE_INNER_ERR_REPORT(op_name.GetString(), "attrs size less then 4."), return GRAPH_FAILED);

  int32_t ih = 0;
  int32_t iw = 0;
  int32_t kh = 0;
  int32_t kw = 0;
  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;
  int32_t padt = pad_list[0];
  int32_t padl = pad_list[2];

  if (x_format == FORMAT_NCHW) {
    ih = x_shape[2];
    iw = x_shape[3];
    strh = stride_list[2];
    strw = stride_list[3];
    dilh = dilation_list[2];
    dilw = dilation_list[3];
  } else if (x_format == FORMAT_NHWC) {
    ih = x_shape[1];
    iw = x_shape[2];
    strh = stride_list[1];
    strw = stride_list[2];
    dilh = dilation_list[1];
    dilw = dilation_list[2];
  } else {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "x format is valid, the error x format is: %d", x_format);
    return GRAPH_FAILED;
  }

  if (w_format == FORMAT_NCHW) {
    kh = w_shape[2];
    kw = w_shape[3];
  } else if (w_format == FORMAT_NHWC) {
    kh = w_shape[1];
    kw = w_shape[2];
  } else if (w_format == FORMAT_HWCN) {
    kh = w_shape[0];
    kw = w_shape[1];
  } else {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "weight format is valid, the error w format is: %d", w_format);
    return GRAPH_FAILED;
  }

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
  vector<vector<int64_t>> y_data_slice;
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}};
  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op_name.GetString(), "no data slice, not need infer input");
    return GRAPH_FAILED;
  }
  bool have_slice = false;
  vector<int> new_pad_lists = pad_list;
  for (size_t i = 0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() > 0) {
      have_slice = true;
      if (i == 2) {
        vector<int64_t> ih_slice;
        int64_t top_add_pad = pad_list[0];
        int64_t bom_add_pad = pad_list[1];
        InferHWConv2D(ih, kh, padt, strh, dilh, y_data_slice[i], ih_slice, &top_add_pad, &bom_add_pad);
        OP_LOGD(op_name.GetString(), "conv2d h axis slice ori_scope is [%d, %d], output scope is [%d, %d]",
                ih_slice[0], ih_slice[1], y_data_slice[i][0], y_data_slice[i][1]);
        new_pad_lists[0] = top_add_pad;
        new_pad_lists[1] = bom_add_pad;
        x_data_slice[i] = ih_slice;
      } else if (i == 3) {
        vector<int64_t> iw_slice;
        int64_t left_add_pad = pad_list[2];
        int64_t right_add_pad = pad_list[3];
        InferHWConv2D(iw, kw, padl, strw, dilw, y_data_slice[i], iw_slice, &left_add_pad, &right_add_pad);
        OP_LOGD(op_name.GetString(), "conv2d w axis slice ori_scope is [%d, %d], output scope is [%d, %d]",
                iw_slice[0], iw_slice[1], y_data_slice[i][0], y_data_slice[i][1]);
        new_pad_lists[2] = left_add_pad;
        new_pad_lists[3] = right_add_pad;
        x_data_slice[i] = iw_slice;
      } else {
        x_data_slice[i] = y_data_slice[i];
      }
    }
  }

  op.SetAttr("pads", new_pad_lists);
  OP_LOGD(op_name.GetString(), "Conv2DCompress new pad lists is [%d, %d, %d, %d]", new_pad_lists[0],
          new_pad_lists[1], new_pad_lists[2], new_pad_lists[3]);

  if (!have_slice) {
    return GRAPH_FAILED;
  }
  if (!AttrUtils::SetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice)) {
    return GRAPH_FAILED;
  }
  OP_LOGD(op_name.GetString(), "Calc Conv2DCompress InferDataSlice end!");
  return GRAPH_SUCCESS;
}

INFER_DATA_SLICE_FUNC_REG(Conv2DCompress, Conv2DCompressInferDataSlice);
INFER_FUNC_REG(Conv2DCompress, Conv2DCompressInfer);
VERIFY_FUNC_REG(Conv2DCompress, Conv2DCompressVerify);

// ----------------DeformableConv2d-----------------------
/*!
  * @brief Get 2D(H/W) stride, dilation and pad params to infershape output.
  *
  * [strides]: 4D list, format sensitive, according to first input tensor format
  * [dilations]: 4D list, format sensitive
  * [pads]: 4D list
  *
  * @param op DeformableConv2D operator.
  * @param refer Valid value reference format.
  * @param strh, strw  Input stride H/W value.
  * @param dilh, dilw  Input dilation H/W value.
  * @param padt, padb, padl, padr  Input top/bottom/left/right pad value.
  * @return bool Whether the strides, dilations settings are correct.
  */
static bool GetAttrsDfmConv2D(ge::Operator& op, Format refer, int32_t& strh, int32_t& strw, int32_t& dilh,
                              int32_t& dilw, int32_t& padt, int32_t& padb, int32_t& padl, int32_t& padr) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  std::vector<int32_t> stride_list;
  op.GetAttr("strides", stride_list);
  auto s_size = stride_list.size();
  if (stride_list.empty() || s_size != kConv2dStridesSizeLimit) {
    OP_LOGE(op_name.GetString(), "strides list should be 4D. actual is: %lu.", s_size);
    map<string, string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = op_name.GetString();
    err_map["expected_value"] = "4D";
    err_map["input_value"] = std::to_string(s_size) + "D.";
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  std::vector<int32_t> dilation_list;
  op.GetAttr("dilations", dilation_list);
  auto d_size = dilation_list.size();
  if (dilation_list.empty() || d_size != kConv2dDilationSizeLimit) {
    OP_LOGE(op_name.GetString(), "dilations list should be 4D. actual is: %lu.", d_size);
    map<string, string> err_map;
    err_map["param_name"] = "dilations";
    err_map["op_name"] = op_name.GetString();
    err_map["expected_value"] = "4D";
    err_map["input_value"] = std::to_string(d_size) + "D.";
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  int32_t strn = 0;
  int32_t strc = 0;
  int32_t diln = 0;
  int32_t dilc = 0;
  if (refer == FORMAT_NCHW) {
    strn = stride_list[kNDimIdx];
    strc = stride_list[kCDimNCHWIdx];
    strh = stride_list[kHDimNCHWIdx];
    strw = stride_list[kWDimNCHWIdx];
    diln = dilation_list[kNDimIdx];
    dilc = dilation_list[kCDimNCHWIdx];
    dilh = dilation_list[kHDimNCHWIdx];
    dilw = dilation_list[kWDimNCHWIdx];
  } else if (refer == FORMAT_NHWC) {
    strn = stride_list[kNDimIdx];
    strc = stride_list[kCDimNHWCIdx];
    strh = stride_list[kHDimNHWCIdx];
    strw = stride_list[kWDimNHWCIdx];
    diln = dilation_list[kNDimIdx];
    dilc = dilation_list[kCDimNHWCIdx];
    dilh = dilation_list[kHDimNHWCIdx];
    dilw = dilation_list[kWDimNHWCIdx];
  }
  if (strh <= 0 || strw <= 0) {
    OP_LOGE(op_name.GetString(),
            "strides should be positive,"
            " actual is [%d,%d].",
            strh, strw);
    map<string, string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = op_name.GetString();
    err_map["expected_value"] = "positive";
    err_map["input_value"] = std::to_string(strh) + ", " + std::to_string(strw);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  if (strn != 1 || strc != 1) {
    OP_LOGE(op_name.GetString(), "strides N/C dimensions must be set to 1.");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "strides N/C dimensions must be set to 1.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  if (dilh <= 0 || dilw <= 0) {
    OP_LOGE(op_name.GetString(),
            "dilations should be positive,"
            " actual is [%d,%d].",
            dilh, dilw);
    map<string, string> err_map;
    err_map["param_name"] = "dilations";
    err_map["op_name"] = op_name.GetString();
    err_map["expected_value"] = "positive";
    err_map["input_value"] = std::to_string(dilh) + ", " + std::to_string(dilw);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  if (diln != 1 || dilc != 1) {
    OP_LOGE(op_name.GetString(), "dilations N/C dimensions must be set to 1.");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "dilations N/C dimensions must be set to 1.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  std::vector<int32_t> pads_list;
  op.GetAttr("pads", pads_list);
  auto p_size = pads_list.size();
  if (pads_list.empty() || p_size != kConv2dPadSizeLimit) {
    OP_LOGE(op_name.GetString(), "pads list should be 4D. actual is: %lu.", p_size);
    map<string, string> err_map;
    err_map["param_name"] = "pads";
    err_map["op_name"] = op_name.GetString();
    err_map["expected_value"] = "4D";
    err_map["input_value"] = std::to_string(p_size) + "D.";
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  padt = pads_list[kConv2dPadUpIdx];
  padb = pads_list[kConv2dPadDownIdx];
  padl = pads_list[kConv2dPadLeftIdx];
  padr = pads_list[kConv2dPadRightIdx];
  bool negative_pads = padt < 0 || padb < 0 || padl < 0 || padr < 0;
  if (negative_pads) {
    OP_LOGE(op_name.GetString(),
            "pads should be positive, "
            " actual is [%d,%d,%d,%d].",
            padt, padb, padl, padr);
    map<string, string> err_map;
    err_map["param_name"] = "pads";
    err_map["op_name"] = op_name.GetString();
    err_map["expected_value"] = "positive";
    err_map["input_value"] =
        std::to_string(padt) + ", " + std::to_string(padb) + ", " + std::to_string(padl) + ", " + std::to_string(padr);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  return true;
}

/*!
  * @brief Infer output shape and dtype, dtype is same to first input tensor
  *        Output format is set by ge parser process already
  * @param DeformableConv2D Operator type.
  * @param DeformableConv2DInfer Infer function.
  * @return Status The processing flow result.
  */
IMPLEMT_INFERFUNC(DeformableConv2D, DeformableConv2DInfer) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter DeformableConv2DInfer");
  auto x_tensor = op.GetInputDescByName("x");
  auto w_tensor = op.GetInputDescByName("filter");

  auto x_shape = x_tensor.GetShape().GetDims();
  auto w_shape = w_tensor.GetShape().GetDims();
  if (x_shape.size() != 4 || w_shape.size() != 4) {
    return GRAPH_FAILED;
  }
  auto x_format = x_tensor.GetFormat();
  auto w_format  = w_tensor.GetFormat();
  CHECK_OP_FUNC(FORMAT_RESERVED == x_format, return GRAPH_FAILED, "get format failed: %d", x_format);
  CHECK_OP_FUNC(FORMAT_RESERVED == w_format, return GRAPH_FAILED, "get format failed: %d", w_format);

  int32_t in = 0;
  int32_t ic = 0;
  int32_t ih = 0;
  int32_t iw = 0;
  int32_t kn = 0;
  int32_t kc = 0;
  int32_t kh = 0;
  int32_t kw = 0;

  // set fm
  if (x_format == FORMAT_NCHW) {
    in = x_shape[0];
    ic = x_shape[1];
    ih = x_shape[2];
    iw = x_shape[3];
  } else if (x_format == FORMAT_NHWC) {
    in = x_shape[0];
    ic = x_shape[3];
    ih = x_shape[1];
    iw = x_shape[2];
  } else {
    OP_LOGE(op_name.GetString(), "input x format should be NCHW or NHWC. actual is: %s",
            TypeUtils::FormatToSerialString(x_format).c_str());
    map<string, string> err_map;
    err_map["param"] = "x";
    err_map["op_name"] = op_name.GetString();
    err_map["expected_format_list"] = "NCHW or NHWC";
    err_map["format"] = TypeUtils::FormatToSerialString(x_format);
    std::string report_error_code = "E50002";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // set kernel
  if (w_format == FORMAT_NCHW) {
    kn = w_shape[0];
    kc = w_shape[1];
    kh = w_shape[2];
    kw = w_shape[3];
  } else if (w_format == FORMAT_HWCN) {
    kn = w_shape[3];
    kc = w_shape[2];
    kh = w_shape[0];
    kw = w_shape[1];
  } else {
    OP_LOGE(op_name.GetString(),
            "input filter format should be NCHW or HWCN. actual is: %s",
            TypeUtils::FormatToSerialString(w_format).c_str());
    map<string, string> err_map;
    err_map["param"] = "filter";
    err_map["op_name"] = op_name.GetString();
    err_map["expected_format_list"] = "NCHW, NHWC or HWCN";
    err_map["format"] = TypeUtils::FormatToSerialString(w_format);
    std::string report_error_code = "E50002";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  auto bias_tensor = op.GetInputDescByName("bias");
  auto bias_shape = bias_tensor.GetShape().GetDims();
  if (bias_shape.size() == 1 && bias_shape[0] != kn) {
    OP_LOGE(op_name.GetString(), "input bias size should be equal to out_channels.");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "input bias size should be equal to out_channels.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  } else if (bias_shape.size() > 1) {
    OP_LOGE(op_name.GetString(), "input bias shape should be 1D.");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "input bias shape should be 1D.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // set data_format: copy value of x_format to data_format
  // data_format will be used to infer position of H/W
  // in strides and dilations(previously used ori_format)
  std::string data_format;
  std::string attr_data_format = "data_format";
  std::string data_format_NCHW = "NCHW";
  std::string data_format_NHWC = "NHWC";

  if (x_format == ge::FORMAT_NCHW) {
    op.SetAttr(attr_data_format, data_format_NCHW);
  } else {
    op.SetAttr(attr_data_format, data_format_NHWC);
  }

  // set strides, dilations, pad
  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;
  int32_t padt = 0;
  int32_t padb = 0;
  int32_t padl = 0;
  int32_t padr = 0;
  if (!GetAttrsDfmConv2D(op, x_format, strh, strw, dilh, dilw, padt, padb, padl, padr)) {
    return GRAPH_FAILED;
  }

  int64_t ih_pad = (ih + padt + padb - dilh * (kh - 1) - 1);
  int64_t iw_pad = (iw + padl + padr - dilw * (kw - 1) - 1);
  int64_t oh = ih_pad / strh + 1;
  int64_t ow = iw_pad / strw + 1;
  if ((ih > 0) && (kh > 0) && (iw > 0) && (kw > 0)) {
    if ((ih_pad < 0) || (iw_pad < 0)) {
      OP_LOGE(op_name.GetString(), "image size after padding should be greater than or equal to filter size.");
      map<string, string> err_map;
      err_map["op_name"] = op_name.GetString();
      err_map["description"] = "image size after padding should be greater than or equal to filter size.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    }
  }

  int64_t groups = 1;
  op.GetAttr("groups", groups);
  if (ic != kc * groups) {
    OP_LOGE(op_name.GetString(),
            "x channel should be equal to filter channel*groups. "
            "x format is: %s, filter format is: %s, "
            "x shape is: [%d,%d,%d,%d], filter shape is: [%d,%d,%d,%d], "
            "groups is: %d.",
            TypeUtils::FormatToSerialString(x_format).c_str(), TypeUtils::FormatToSerialString(w_format).c_str(),
            (int)x_shape[0], (int)x_shape[1], (int)x_shape[2], (int)x_shape[3], (int)w_shape[0], (int)w_shape[1],
            (int)w_shape[2], (int)w_shape[3], (int)groups);
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["x_shape"] = std::to_string(x_shape[0]) + ", " + std::to_string(x_shape[1]) + ", " +
                         std::to_string(x_shape[2]) + ", " + std::to_string(x_shape[3]);
    err_map["filter_shape"] = std::to_string(w_shape[0]) + ", " + std::to_string(w_shape[1]) + ", " +
                              std::to_string(w_shape[2]) + ", " + std::to_string(w_shape[3]);
    err_map["groups"] = std::to_string(groups);

    std::string report_error_code = "E50059";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  if (kn % groups != 0) {
    OP_LOGE(op_name.GetString(), "out_channels should be divisible by groups.");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "out_channels should be divisible by groups.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  auto offset_tensor = op.GetInputDescByName("offsets");
  auto offset_format  = offset_tensor.GetFormat();
  auto offset_shape = offset_tensor.GetShape().GetDims();
  if (offset_shape.size() != 4) {
    return GRAPH_FAILED;
  }
  int32_t dfm_group = 1;
  op.GetAttr("deformable_groups", dfm_group);
  if (dfm_group < 1 || ic % dfm_group != 0) {
    OP_LOGE(op_name.GetString(), "deformable_groups should be positive and can divide in_channels");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "deformable_groups should be positive and can divide in_channels.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  // >>> start: the restriction comes from DeformableOffsets
  bool modulated = true;
  op.GetAttr("modulated", modulated);
  CHECK_OP_FUNC(!modulated, return GRAPH_FAILED, "Currently modulated must be true.");
  // <<< end: the restriction comes from DeformableOffsets
  std::vector<int64_t> exp_shape;
  if (offset_format == FORMAT_NCHW) {
    exp_shape.push_back(in);
    exp_shape.push_back(dfm_group * kh * kw * 3);
    exp_shape.push_back(oh);
    exp_shape.push_back(ow);
  } else if (offset_format == FORMAT_NHWC) {
    exp_shape.push_back(in);
    exp_shape.push_back(oh);
    exp_shape.push_back(ow);
    exp_shape.push_back(dfm_group * kh * kw * 3);
  } else {
    OP_LOGE(op_name.GetString(), "input offsets format should be NCHW or NHWC. actual is: %s",
            TypeUtils::FormatToSerialString(offset_format).c_str());
    map<string, string> err_map;
    err_map["param"] = "offsets";
    err_map["op_name"] = op_name.GetString();
    err_map["expected_format_list"] = "NCHW or NHWC";
    err_map["format"] = TypeUtils::FormatToSerialString(offset_format);
    std::string report_error_code = "E50002";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  if (exp_shape != offset_shape) {
    OP_LOGE(op_name.GetString(), "input offsets shape should be [%d,%d,%d,%d].",
      (int)exp_shape[0], (int)exp_shape[1], (int)exp_shape[2], (int)exp_shape[3]);
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "input offsets shape should be [" + std::to_string(exp_shape[0]) + ", " +
                             std::to_string(exp_shape[1]) + ", " + std::to_string(exp_shape[2]) + ", " +
                             std::to_string(exp_shape[3]) + "].";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  vector<int64_t> y_shape;
  auto y_tensor = op.GetOutputDescByName("y");
  auto y_format = y_tensor.GetFormat();
  CHECK_OP_FUNC(FORMAT_RESERVED == y_format, return GRAPH_FAILED, "get format failed: %d", y_format)
  if (y_format == FORMAT_NCHW) {
    y_shape.push_back(in);
    y_shape.push_back(kn);
    y_shape.push_back(oh);
    y_shape.push_back(ow);
  } else if (y_format == FORMAT_NHWC) {
    y_shape.push_back(in);
    y_shape.push_back(oh);
    y_shape.push_back(ow);
    y_shape.push_back(kn);
  } else {
    OP_LOGE(op_name.GetString(), "output y format should be NCHW or NHWC. actual is: %s",
            TypeUtils::FormatToSerialString(y_format).c_str());
    map<string, string> err_map;
    err_map["param"] = "y";
    err_map["op_name"] = op_name.GetString();
    err_map["expected_format_list"] = "NCHW or NHWC";
    err_map["format"] = TypeUtils::FormatToSerialString(y_format);
    std::string report_error_code = "E50002";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  y_tensor.SetShape(Shape(y_shape));
  auto x_dtype = x_tensor.GetDataType();
  if (x_dtype == ge::DT_INT8) {
    y_tensor.SetDataType(ge::DT_INT32);
  } else {
    y_tensor.SetDataType(x_dtype);
  }

  if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", y_tensor)) {
    OP_LOGE(op_name.GetString(), "update output desc failed.");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "update output desc failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  OP_LOGD(op_name.GetString(), "Leave DeformableConv2DInfer.");
  return GRAPH_SUCCESS;
}

/*!
  * @brief Verify the required 3 input tensor, optional bias ignored, verify
  *        strides and dilations attrs, pads ignored.
  * @param DeformableConv2D Operator type.
  * @param DeformableConv2DVerify Input validity check function.
  * @return Status The processing flow result.
  */
IMPLEMT_VERIFIER(DeformableConv2D, DeformableConv2DVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter DeformableConv2DVerify.");
  auto x_tensor = op.GetInputDescByName("x");
  auto w_tensor = op.GetInputDescByName("filter");
  auto offset_tensor = op.GetInputDescByName("offsets");
  auto x_shape = x_tensor.GetOriginShape().GetDims();
  auto w_shape = w_tensor.GetOriginShape().GetDims();
  auto offset_shape = offset_tensor.GetOriginShape().GetDims();

  if (x_shape.size() != 4) {
    if (x_shape.size() == 0) {
      OP_LOGE(op_name.GetString(), "input x shape is empty.");
      map<string, string> err_map;
      err_map["op_name"] = op_name.GetString();
      err_map["description"] = "input x shape is empty.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    } else {
      OP_LOGE(op_name.GetString(), "input x shape should be 4D.");
      map<string, string> err_map;
      err_map["op_name"] = op_name.GetString();
      err_map["description"] = "input x shape should be 4D.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    }
    return GRAPH_FAILED;
  }
  if (w_shape.size() != 4) {
    if (w_shape.size() == 0) {
      OP_LOGE(op_name.GetString(), "input filter shape is empty.");
      map<string, string> err_map;
      err_map["op_name"] = op_name.GetString();
      err_map["description"] = "input filter shape is empty.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    } else {
      OP_LOGE(op_name.GetString(), "input filter shape should be 4D.");
      map<string, string> err_map;
      err_map["op_name"] = op_name.GetString();
      err_map["description"] = "input filter shape should be 4D.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    }
    return GRAPH_FAILED;
  }
  if (offset_shape.size() != 4) {
    if (offset_shape.size() == 0) {
      OP_LOGE(op_name.GetString(), "input offsets shape is empty.");
      map<string, string> err_map;
      err_map["op_name"] = op_name.GetString();
      err_map["description"] = "input offsets shape is empty.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    } else {
      OP_LOGE(op_name.GetString(), "input offsets shape should be 4D.");
      map<string, string> err_map;
      err_map["op_name"] = op_name.GetString();
      err_map["description"] = "input offsets shape should be 4D.";
      std::string report_error_code = "E50060";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    }
    return GRAPH_FAILED;
  }

  auto x_dtype = x_tensor.GetDataType();
  auto w_dtype = w_tensor.GetDataType();

  if (x_dtype != w_dtype) {
    OP_LOGE(op_name.GetString(),
            "input x dtype is differ from filter dtype. actual x dtype is: %d filter dtype is: %d",
            (int)x_dtype, (int)w_dtype);
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["param1"] = "x";
    err_map["param1_data_type"] = std::to_string(x_dtype);
    err_map["param2"] = "filter";
    err_map["param2_data_type"] = std::to_string(w_dtype);
    err_map["rule"] = "input x dtype is same as filter dtype";
    std::string report_error_code = "E50004";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> stride_list;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stride_list)) {
    OP_LOGE(op_name.GetString(), "get strides list failed.");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "get strides list failed.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  std::vector<int32_t> dilation_list;
  if (GRAPH_SUCCESS != op.GetAttr("dilations", dilation_list)) {
    OP_LOGE(op_name.GetString(), "get dilations list failed.");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "get dilations list failed.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  OP_LOGD(op_name.GetString(), "Leave DeformableConv2DVerify.");
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DeformableConv2D, DeformableConv2DInfer);
VERIFY_FUNC_REG(DeformableConv2D, DeformableConv2DVerify);
// ----------------Deconvolution--------------------------
static bool GetAttrsDeconv(ge::Operator& op, Format refer, int32_t& strh,
                           int32_t& strw, int32_t& dilh, int32_t& dilw) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  std::vector<int32_t> stride_list;
  op.GetAttr("strides", stride_list);
  auto s_size = stride_list.size();
  // in deconv stride len is 2
  if (s_size != 2) {
    OP_LOGE(op_name.GetString(),
            "strides list should be 2d."
            " actual is: %lu.",
            s_size);
    string sizealue = ConcatString(s_size);
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["param_name"] = "sSize";
    err_map["expected_value"] = "2d";
    err_map["input_value"] = sizealue;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  std::vector<int32_t> dilation_list;
  op.GetAttr("dilations", dilation_list);
  auto d_size = dilation_list.size();
  if (d_size != kConv2dDimSizeLimit) {
    OP_LOGE(op_name.GetString(),
            "dilations list should be 4d."
            " actual is: %lu.",
            d_size);
    string realvalue = ConcatString(d_size);
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["param_name"] = "dSize";
    err_map["expected_value"] = "4d";
    err_map["input_value"] = realvalue;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  strh = stride_list[0];
  strw = stride_list[1];
  if (refer == FORMAT_NCHW) {
    dilh = dilation_list[kHDimNCHWIdx];
    dilw = dilation_list[kWDimNCHWIdx];
  } else {
    return false;
  }
  if (strh <= 0 || strw <= 0) {
    OP_LOGE(op_name.GetString(),
            "strides should be positive, "
            " actual is [%d,%d].",
            strh, strw);
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["param_name"] = "strides";
    err_map["expected_value"] = "positive";
    err_map["input_value"] = "[" + std::to_string(strh) + "," + std::to_string(strw) + "]";
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  return true;
}


IMPLEMT_INFER_DATA_SLICE(Deconvolution, DeconvolutionInferDataSlice) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter Deconvolution InferDataSlice.");

  auto x_tensor = op.get_input_desc_x();
  auto w_tensor = op.get_input_desc_filter();
  auto x_format = x_tensor.GetOriginFormat();
  auto x_shape = x_tensor.GetOriginShape().GetDims();
  auto w_shape = w_tensor.GetOriginShape().GetDims();
  auto x_dtype = x_tensor.GetDataType();
  int32_t ih = -1;
  int32_t iw = -1;
  if (x_shape != DYNAMIC_DIM_ALL) {
    ih = x_shape[kHDimNCHWIdx];
    iw = x_shape[kWDimNCHWIdx];
  }
  int32_t kh = w_shape[kHDimNCHWIdx];
  int32_t kw = w_shape[kWDimNCHWIdx];
  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;

  if (false == GetAttrsDeconv(op, x_format, strh, strw, dilh, dilw)) {
    OP_LOGE(op_name.GetString(), "get attrs failed.");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "get attrs failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  if ((strh <= 0) || (strw <= 0)) {
    OP_LOGE(op_name.GetString(), "stride can not less than zero");
    ErrorManager::GetInstance().ATCReportErrMessage("E50029",
                                                    {"op_name", "param_name", "expected_value", "input_value"},
                                                    {op_name.GetString(), "strides", "positive",
                                                    std::to_string(strh) + ", " + std::to_string(strw)});
    return GRAPH_FAILED;
  }
  vector<int32_t> pad_list;
  op.GetAttr("pads", pad_list);
  CHECK_SIZE(pad_list.size() != kConv2dPadSizeLimit, return GRAPH_FAILED, "pad is invalid");
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
  GeTensorDescPtr tensor_desc_w = op_desc->MutableInputDesc("filter");
  CHECK_PTR_NULL(tensor_desc_y, "tensor y desc", return GRAPH_FAILED);
  CHECK_PTR_NULL(tensor_desc_x, "tensor x desc", return GRAPH_FAILED);
  CHECK_PTR_NULL(tensor_desc_w, "tensor filter desc", return GRAPH_FAILED);

  vector<vector<int64_t>> y_data_slice;
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}};
  vector<vector<int64_t>> w_data_slice = {{}, {}, {}, {}};

  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op_name.GetString(), "no data slice, not need infer input");
    return GRAPH_FAILED;
  }

  for (size_t i = 0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() > 0) {
      bool deconv_failed = i == 1 && x_dtype != DT_INT8 && y_data_slice[i].size() < kDataSliceLen;
      if (deconv_failed) {
        CUBE_INNER_ERR_REPORT(op_name.GetString(), "Shape dim check failed.");
        return GRAPH_FAILED;
      } else {
        std::vector<int32_t> y_position = {kNDimIdx, kCDimNCHWIdx, kHDimNCHWIdx, kWDimNCHWIdx};
        std::vector<int32_t> attr_param = {kh, kw, dilh, dilw, strh, strw, ih, iw};
        std::vector<int32_t> input_sizes;
        return data_slice_conv2d_backprop_input(op, y_data_slice, x_data_slice, w_data_slice,
                                                y_position, attr_param, input_sizes, pad_list,
                                                tensor_desc_w, tensor_desc_x, i);
      }
    }
  }

  OP_LOGI(op_name.GetString(), "no data slice, not need infer input");
  return GRAPH_FAILED;
}

static void set_deconvolution_out_shape_range(size_t idx,
                                              const vector<int32_t>& attrParams,
                                              const std::vector<std::pair<int64_t, int64_t>>& fm_range,
                                              std::vector<std::pair<int64_t, int64_t>>& out_range) {
  size_t attrIdx = 0;
  int32_t stride = attrParams[attrIdx++];
  int32_t kernel = attrParams[attrIdx++];
  int32_t pad = attrParams[attrIdx++];
  int64_t low = fm_range[idx].first;
  int64_t high = fm_range[idx].second;

  if (high == -1) {
    out_range[idx].first = std::max(low, kDynamicRangeLowerBound);
    out_range[idx].second = high;
  } else {
    out_range[idx].first = stride * (low - 1) + kernel - pad;
    out_range[idx].second = stride * (high - 1) + kernel - pad;
  }
  out_range[idx].first = std::max(out_range[idx].first, kDynamicRangeLowerBound);
  out_range[idx].second = std::min(out_range[idx].second, kDynamicRangeUpperBound);
}

IMPLEMT_INFERFUNC(Deconvolution, DeconvolutionInfer) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter DeconvolutionInfer.");
  auto opDesc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(opDesc, "op desc", return GRAPH_FAILED);
  auto xTensor = opDesc->MutableInputDesc("x");
  auto wTensor = opDesc->MutableInputDesc("filter");
  auto yTensor = opDesc->MutableOutputDesc("y");
  CHECK_PTR_NULL(yTensor, "tensor y desc", return GRAPH_FAILED);
  CHECK_PTR_NULL(xTensor, "tensor x desc", return GRAPH_FAILED);
  CHECK_PTR_NULL(wTensor, "tensor filter desc", return GRAPH_FAILED);

  auto xShape = xTensor->MutableShape().GetDims();
  auto wShape = wTensor->MutableShape().GetDims();
  auto xFormat = xTensor->GetFormat();
  auto wFormat = wTensor->GetFormat();
  CHECK_OP_FUNC(FORMAT_RESERVED == xFormat, return GRAPH_FAILED, "get format failed: %d", xFormat);
  CHECK_OP_FUNC(FORMAT_RESERVED == wFormat, return GRAPH_FAILED, "get format failed: %d", wFormat);
  bool isDynamic = false;
  bool unknownRank = IsUnknownRankShape(xShape);
  if (std::find(xShape.begin(), xShape.end(), -1) != xShape.end()) {
    isDynamic = true;
    reset_range(op, "x");
  }
  int32_t in = -1;
  int32_t ic = -1;
  int32_t ih = -1;
  int32_t iw = -1;
  int32_t kn = 0;
  int32_t kc = 0;
  int32_t kh = 0;
  int32_t kw = 0;
  if (xFormat != FORMAT_NCHW) {
    OP_LOGE(op_name.GetString(),
            "input x format should be NCHW"
            " actual is: %d",
            xFormat);
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["param_name"] = "xFormat";
    err_map["expected_format_list"] = "[NCHW]";
    err_map["format"] = ConcatString(xFormat);
    std::string report_error_code = "E50033";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  if (!unknownRank) {
    in = xShape[kNDimIdx];
    ic = xShape[kCDimNCHWIdx];
    ih = xShape[kHDimNCHWIdx];
    iw = xShape[kWDimNCHWIdx];
  }

  if (wFormat != FORMAT_NCHW) {
    OP_LOGE(op_name.GetString(), "wFormat != FORMAT_NCHW");
    return GRAPH_FAILED;
  }
  kn = wShape[kNDimIdx];
  kc = wShape[kCDimNCHWIdx];
  kh = wShape[kHDimNCHWIdx];
  kw = wShape[kWDimNCHWIdx];

  int64_t groups = 1;
  if (GRAPH_SUCCESS != op.GetAttr("groups", groups)) {
    OP_LOGI(op_name.GetString(), "no groups setting, use groups as 1");
  }
  OP_LOGI(op_name.GetString(), "groups is %lld", groups);

  bool invalid_channel = (!unknownRank) && (ic != -1) && (ic != kn);
  if (invalid_channel) {
    OP_LOGE(op_name.GetString(),
            "input x channel should be equal to filter. "
            "x format is: %d, filter format is: %d "
            "x shape is: [%s], filter shape is: [%s].",
            xFormat, wFormat, ops::to_string(xShape).c_str(), ops::to_string(wShape).c_str());
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["attr_name"] = "channel";
    err_map["param1_name"] = "x";
    err_map["param1_value"] = std::to_string(ic);
    err_map["param2_name"] = "filter";
    err_map["param2_value"] = std::to_string(kn);
    std::string report_error_code = "E50031";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;
  int32_t padt = 0;
  int32_t padb = 0;
  int32_t padl = 0;
  int32_t padr = 0;
  if (false == GetAttrsDeconv(op, xFormat, strh, strw, dilh, dilw)) {
    OP_LOGE(op_name.GetString(), "get attrs failed.");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "get attrs failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  if (false == GetPadConv2D(op_name, opDesc, ih, iw, kh, kw, strh, strw, dilh, dilw, padt, padb, padl, padr)) {
    OP_LOGE(op_name.GetString(), "get pads attrs failed.");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "get pads attrs failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  int khext = dilh * (kh - 1) + 1;
  int kwext = dilw * (kw - 1) + 1;
  int64_t oh = strh * (ih - 1) + khext - padt - padb;
  int64_t ow = strw * (iw - 1) + kwext - padl - padr;
  if (unknownRank) {
    oh = -1;
    ow = -1;
  }
  if (isDynamic) {
    size_t idxC = 1;
    size_t idxH = kHDimNCHWIdx;
    size_t idxW = kWDimNCHWIdx;
    OP_LOGD(op_name.GetString(), "dynamic shape set range");
    std::vector<std::pair<int64_t, int64_t>> x_range;
    xTensor->GetShapeRange(x_range);
    if (ih == -1) {
      oh = -1;
    }
    if (iw == -1) {
      ow = -1;
    }
    bool x_range_valid = !x_range.empty() && x_range.size() > idxW;
    if (x_range_valid) {
      std::vector<std::pair<int64_t, int64_t>> y_range(x_range);
      y_range[idxC].first = (int64_t)kc * groups;
      y_range[idxC].second = (int64_t)kc * groups;
      y_range[idxH].first = oh;
      y_range[idxH].second = oh;
      y_range[idxW].first = ow;
      y_range[idxW].second = ow;
      if (ih == -1) {
        vector<int32_t> attr_params_h = {strh, khext, padt + padb};
        set_deconvolution_out_shape_range(idxH, attr_params_h, x_range, y_range);
      }
      if (iw == -1) {
        vector<int32_t> attr_params_w = {strw, kwext, padl + padr};
        set_deconvolution_out_shape_range(idxW, attr_params_w, x_range, y_range);
      }
      yTensor->SetShapeRange(y_range);
    }
  }
  vector<int64_t> y_shape;
  auto yFormat = yTensor->GetFormat();
  CHECK_OP_FUNC(FORMAT_RESERVED == yFormat, return GRAPH_FAILED, "get format failed: %d", yFormat);
  if (yFormat != FORMAT_NCHW) {
    OP_LOGE(op_name.GetString(),
            "output y format should be NCHW."
            " actual is: %d",
            yFormat);
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["param_name"] = "yFormat";
    err_map["expected_format_list"] = "[NCHW]";
    err_map["format"] = ConcatString(yFormat);
    std::string report_error_code = "E50033";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  y_shape.push_back(in);
  y_shape.push_back(kc * groups);
  y_shape.push_back(oh);
  y_shape.push_back(ow);
  yTensor->SetShape(GeShape(y_shape));
  auto xDtype = xTensor->GetDataType();
  yTensor->SetDataType(xDtype);
  if (xDtype == DT_INT8) {
    yTensor->SetDataType(DT_INT32);
  }
  OP_LOGD(op_name.GetString(), "Leave DeconvolutionInfer.");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Deconvolution, DeconvolutionVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter DeconvolutionVerify.");
  auto x_tensor = op.get_input_desc_x();
  auto w_tensor = op.get_input_desc_filter();

  auto x_shape = x_tensor.GetShape().GetDims();
  auto w_shape = w_tensor.GetShape().GetDims();
  bool unknownRank = IsUnknownRankShape(x_shape);
  if ((!unknownRank) && (x_shape.size() != kConv2dDimSizeLimit)) {
    OP_LOGE(op_name.GetString(), "input x shape should be 4d.");
    string xvalue = ConcatString(x_shape.size());
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["param_name"] = "xShape";
    err_map["expected_value"] = "4d";
    err_map["input_value"] = xvalue;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  if (w_shape.size() != kConv2dDimSizeLimit) {
    OP_LOGE(op_name.GetString(), "input filter shape should be 4d.");
    string w_value = ConcatString(w_shape.size());
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["param_name"] = "wShape";
    err_map["expected_value"] = "4d";
    err_map["input_value"] = w_value;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  auto x_dtype = x_tensor.GetDataType();
  auto w_dtype = w_tensor.GetDataType();
  if (x_dtype != w_dtype) {
    OP_LOGE(op_name.GetString(),
            "input x dtype is differ from filter dtype."
            " actual x dtype is: %d filter dtype is: %d",
            (int)x_dtype, (int)w_dtype);
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["param1"] = "x";
    err_map["param1_data_type"] = std::to_string(x_dtype);
    err_map["param2"] = "filter";
    err_map["param2_data_type"] = std::to_string(w_dtype);
    err_map["rule"] = "input x dtype is same as filter dtype";
    std::string report_error_code = "E50004";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> stride_list;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stride_list)) {
    OP_LOGE(op_name.GetString(), "get strides list failed.");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "get strides list failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  std::vector<int32_t> dilation_list;
  if (GRAPH_SUCCESS != op.GetAttr("dilations", dilation_list)) {
    OP_LOGE(op_name.GetString(), "get dilations list failed.");
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "get dilations list failed.";
    std::string report_error_code = "E50058";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  OP_LOGD(op_name.GetString(), "Leave DeconvolutionVerify.");
  return GRAPH_SUCCESS;
}

INFER_DATA_SLICE_FUNC_REG(Deconvolution, DeconvolutionInferDataSlice);
INFER_FUNC_REG(Deconvolution, DeconvolutionInfer);
VERIFY_FUNC_REG(Deconvolution, DeconvolutionVerify);

// ---------------------------Conv3D---------------------------
template <typename T1>
static bool CheckVectorAnyNegative(const std::vector<T1>& list)
{
    for (const auto& iter : list) {
        if (iter < 0) {
            return false;
        }
    }
    return true;
}

static bool VerifyConv3dDilations(const ge::Operator& op, std::vector<int32_t>& dilation_list) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  // check dilations shape
  if (op.GetAttr("dilations", dilation_list) != GRAPH_SUCCESS) {
    dilation_list.clear();
    for (int32_t i = 0; i < kConv3dDimSizeLimit; i++) {
      dilation_list.push_back(1);
    }
    OP_LOGI(op_name.GetString(), "no dilations setting, use dilations as [1,1,1,1,1]");
  }
  auto d_size = dilation_list.size();
  if (d_size != kConv3dDimSizeLimit) {
    OP_LOGE(op_name.GetString(), "dilations list should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "dilation_list";
    err_map["op_name"] = "Conv3d or Conv3dbp or Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(kConv3dDimSizeLimit);
    err_map["input_value"] = std::to_string(d_size);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  return true;
}

static bool GetPadConv3D(ge::Operator& op, int32_t id, int32_t ih, int32_t iw,
                         int32_t kd, int32_t kh, int32_t kw, int32_t strd,
                         int32_t strh, int32_t strw, int32_t dild, const int32_t dilh,
                         int32_t dilw, int32_t& padf, int32_t& padba, int32_t& padt,
                         int32_t& padb, int32_t& padl, int32_t& padr) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  std::string pad_str;
  std::vector<int32_t> pad_list;
  if (GRAPH_SUCCESS == op.GetAttr("_padding", pad_list)) {
    op.SetAttr("pads", pad_list);
  } else if (GRAPH_SUCCESS == op.GetAttr("padding", pad_str)) {
    if (pad_str.compare("SAME") == 0) {
      int32_t tails_d = id % strd;
      int32_t tails_h = ih % strh;
      int32_t tails_w = iw % strw;
      int32_t dilate_kernel_d = dild * (kd - 1) + 1;
      int32_t dilate_kernel_h = dilh * (kh - 1) + 1;
      int32_t dilate_kernel_w = dilw * (kw - 1) + 1;
      int32_t pad_d = std::max((tails_d > 0 ? dilate_kernel_d - tails_d : dilate_kernel_d - strd), 0);
      int32_t pad_h = std::max((tails_h > 0 ? dilate_kernel_h - tails_h : dilate_kernel_h - strh), 0);
      int32_t pad_w = std::max((tails_w > 0 ? dilate_kernel_w - tails_w : dilate_kernel_w - strw), 0);
      pad_list.push_back((pad_d >> 1));
      pad_list.push_back((pad_d >> 1) + (pad_d & 1));
      pad_list.push_back((pad_h >> 1));
      pad_list.push_back((pad_h >> 1) + (pad_h & 1));
      pad_list.push_back((pad_w >> 1));
      pad_list.push_back((pad_w >> 1) + (pad_w & 1));
    } else if (pad_str.compare("VALID") == 0) {
      for (int32_t i = 0; i < kConv3dPadsSizeLimit; i++)
        pad_list.push_back(0);
    } else {
      OP_LOGE(op_name.GetString(), "padding should be SAME or VALID.");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "padding";
      err_map["op_name"] = "Conv3d";
      err_map["Expected_value"] = "SAME or VALID";
      err_map["input_value"] = pad_str;
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }
    op.SetAttr("pads", pad_list);
  }
  std::vector<int32_t> pad_vec;
  if (GRAPH_SUCCESS != op.GetAttr("pads", pad_vec)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "get pads failed");
    return false;
  }
  auto p_size = pad_vec.size();
  if (p_size != kConv3dPadsSizeLimit) {
    OP_LOGE(op_name.GetString(), "pads list should be 6d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "pads_list";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "6d";
    err_map["input_value"] = std::to_string(p_size);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  padf = pad_vec[kConv3dPadHeadIdx];
  padba = pad_vec[kConv3dPadTailIdx];
  padt = pad_vec[kConv3dPadUpIdx];
  padb = pad_vec[kConv3dPadDownIdx];
  padl = pad_vec[kConv3dPadLeftIdx];
  padr = pad_vec[kConv3dPadRightIdx];

  auto x_shape = op.GetInputDescByName("x").GetShape().GetDims();
  bool unknown_rank = IsUnknownRankShape(x_shape);
  bool unknown_shape = IsUnKnownShape(x_shape);
  bool negative_pad = std::any_of(pad_vec.begin(), pad_vec.end(), [](int32_t val){ return val < 0;});
  // dynamic shape pad maybe negative
  if ((!unknown_shape) && (!unknown_rank) && negative_pad) {
    OP_LOGE(op_name.GetString(), "pads should be positive");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "pads_list";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "positive";
    err_map["input_value"] = std::to_string(padf) + " " + \
                             std::to_string(padba) + " " + \
                             std::to_string(padt) + " " + \
                             std::to_string(padb) + " " + \
                             std::to_string(padl) + " " + \
                             std::to_string(padr);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  return true;
}

static bool GetAttrsConv3D(ge::Operator& op, Format refer, int32_t& strd,
                           int32_t& strh, int32_t& strw, int32_t& dild,
                           int32_t& dilh, int32_t& dilw) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  std::vector<int32_t> stride_list;
  if (op.GetAttr("strides", stride_list) != GRAPH_SUCCESS) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "get strides list failed.");
    return false;
  }
  auto s_size = stride_list.size();
  if (s_size != kConv3dStridesSizeLimit) {
    OP_LOGE(op_name.GetString(), "strides list should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "stride_list";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "5d";
    err_map["input_value"] = std::to_string(s_size);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::string data_format;
  if (op.GetAttr("data_format", data_format) != GRAPH_SUCCESS) {
    OP_LOGI(op_name.GetString(), "no data format setting, using NDHWC");
    data_format = FORMAT_NDHWC;
  }

  std::vector<int32_t> dilation_list;
  if (!VerifyConv3dDilations(op, dilation_list)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "get dilation attrs failed.");
    return false;
  }

  if (refer == FORMAT_NCDHW) {
    strd = stride_list[kDDimNCDHWIdx];
    strh = stride_list[kHDimNCDHWIdx];
    strw = stride_list[kWDimNCDHWIdx];
    dild = dilation_list[kDDimNCDHWIdx];
    dilh = dilation_list[kHDimNCDHWIdx];
    dilw = dilation_list[kWDimNCDHWIdx];
  } else if (refer == FORMAT_NDHWC) {
    strd = stride_list[kDDimNDHWCIdx];
    strh = stride_list[kHDimNDHWCIdx];
    strw = stride_list[kWDimNDHWCIdx];
    dild = dilation_list[kDDimNDHWCIdx];
    dilh = dilation_list[kHDimNDHWCIdx];
    dilw = dilation_list[kWDimNDHWCIdx];
  }
  if (strd <= 0 || strh <= 0 || strw <= 0) {
    OP_LOGE(op_name.GetString(), "strides should be positive.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "positive";
    err_map["input_value"] = std::to_string(strd) + " " + std::to_string(strh) + " " + std::to_string(strw);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  if (dild != 1) {
    OP_LOGE(op_name.GetString(), "dilations in the D dimension only supports 1 now.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "dilations";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "1";
    err_map["input_value"] = std::to_string(dild);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  return true;
}

static void SetConv3dOutShapeDimRange(const std::string& padding,
                                      size_t idx,
                                      map<std::string, int32_t>& attr_params,
                                      const std::vector<std::pair<int64_t, int64_t>>& fm_range,
                                      std::vector<std::pair<int64_t, int64_t>>& out_range) {
  int32_t stride = attr_params["stride"];
  int32_t dilation = attr_params["dilation"];
  int32_t pad = attr_params["pad"];
  int32_t kernel = attr_params["kernel"];
  int64_t low = fm_range[idx].first;
  int64_t high = fm_range[idx].second;
  if (padding == "SAME") {
    out_range[idx].first = (low + stride - 1) / stride;
    out_range[idx].second = (high + stride - 1) / stride;
  } else {
    out_range[idx].first = (low + pad - dilation * (kernel - 1) - 1) / stride + 1;
    out_range[idx].second = (high + pad - dilation * (kernel - 1) - 1) / stride + 1;
  }

  out_range[idx].first = std::max(out_range[idx].first, kDynamicRangeLowerBound);
  if (high == -1) {
    out_range[idx].second = high;
  } else {
    out_range[idx].second = std::min(out_range[idx].second, kDynamicRangeUpperBound);
  }
}

static bool IsDHWUnknown(const string& op_name, const string& tensor_name,
                         const vector<int64_t>& shape, Format format) {
  size_t idx_n = DIM_INDEX0;
  size_t idx_c = DIM_INDEX4;
  if (format == FORMAT_NCDHW) {
    idx_c = DIM_INDEX1;
  }

  vector<int64_t> shape_copy = shape;
  if (shape.size() > idx_n) {
    shape_copy[idx_n] = 1;
  }

  if ((shape.size() > idx_c) && (shape[idx_c] == -1)) {
    OP_LOGW(op_name.c_str(), "input %s channel is unknown", tensor_name.c_str());
    shape_copy[idx_c] = 1;
  }

  return IsUnKnownShape(shape_copy);
}

static bool SetConv3dOutShapeRange(op::Conv3D& op,
                                   map<std::string, int32_t>& attr_params,
                                   vector<int64_t>& y_shape,
                                   TensorDesc& y_tensor) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  auto x_tensor = op.get_input_desc_x();
  auto x_shape = x_tensor.GetShape().GetDims();
  bool unknown_shape = IsUnKnownShape(x_shape);

  // default format: NDHWC
  size_t idx_d = DIM_INDEX1;
  size_t idx_h = DIM_INDEX2;
  size_t idx_w = DIM_INDEX3;
  size_t idx_c = DIM_INDEX4;
  if (op.get_input_desc_x().GetFormat() == FORMAT_NCDHW) {
    idx_c = DIM_INDEX1;
    idx_d = DIM_INDEX2;
    idx_h = DIM_INDEX3;
    idx_w = DIM_INDEX4;
  }

  // update pads if padding is SAME
  std::string padding;
  // when rank is unknown, or D/H/W is unknown, set SAME padding as -1
  if (IsUnknownRankShape(x_shape) || IsDHWUnknown(string(op_name.GetString()), "x", x_shape, x_tensor.GetFormat())) {
    std::string pad_str;
    if (op.GetAttr("padding", pad_str) == GRAPH_SUCCESS) {
      std::vector<int32_t> pads(kConv3dPadsSizeLimit, 0);
      if (pad_str == "SAME") {
        pads.assign(kConv3dPadsSizeLimit, -1);
        OP_LOGD(op_name.GetString(), "set pads to {-1, -1, -1, -1, -1, -1} when padding is SAME in dynamic_shape");
      }
      op.SetAttr("pads", pads);
    }
  }

  if (!unknown_shape) {
    return true;
  }

  int32_t strd = attr_params["strd"];
  int32_t strh = attr_params["strh"];
  int32_t strw = attr_params["strw"];
  int32_t dild = attr_params["dild"];
  int32_t dilh = attr_params["dilh"];
  int32_t dilw = attr_params["dilw"];
  int32_t padf = attr_params["padf"];
  int32_t padba = attr_params["padba"];
  int32_t padt = attr_params["padt"];
  int32_t padb = attr_params["padb"];
  int32_t padl = attr_params["padl"];
  int32_t padr = attr_params["padr"];
  int32_t kn = attr_params["kn"];
  int32_t kd = attr_params["kd"];
  int32_t kh = attr_params["kh"];
  int32_t kw = attr_params["kw"];

  OP_LOGD(op_name.GetString(), "dynamic shape set range");
  std::vector<std::pair<int64_t, int64_t>> fm_range;
  x_tensor.GetShapeRange(fm_range);
  if (fm_range.empty()) {
    OP_LOGW(op_name.GetString(), "fm_range's shape is empty.");
    if (x_shape[idx_d] == -1) {
      y_shape[idx_d] = -1;
    }

    if (x_shape[idx_h] == -1) {
      y_shape[idx_h] = -1;
    }

    if (x_shape[idx_w] == -1) {
      y_shape[idx_w] = -1;
    }
    // op will check this invalid range
    return true;
  }

  if (fm_range.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op_name.GetString(), "fm_range's shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "fm_range";
    err_map["op_name"] = "Conv3DInfer";
    err_map["excepted_value"] = std::to_string(kConv3dInputSizeLimit);
    err_map["input_value"] = std::to_string(fm_range.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::vector<std::pair<int64_t, int64_t>> out_range(fm_range);
  out_range[idx_c] = std::make_pair(static_cast<int64_t>(kn), static_cast<int64_t>(kn));
  if (x_shape[idx_d] == -1) {
    y_shape[idx_d] = -1;
    map<std::string, int32_t> attr_params_d = {
      {"stride", strd}, {"dilation", dild},
      {"pad", padf + padba}, {"kernel", kd}
    };
    // attr_params_d data structure should keep same as SetConv3dOutShapeDimRange
    SetConv3dOutShapeDimRange(padding, idx_d, attr_params_d, fm_range, out_range);
  }

  if (x_shape[idx_h] == -1) {
    y_shape[idx_h] = -1;
    map<std::string, int32_t> attr_params_h = {
      {"stride", strh}, {"dilation", dilh},
      {"pad", padt + padb}, {"kernel", kh}
    };
    // attr_params_h data structure should keep same as SetConv3dOutShapeDimRange
    SetConv3dOutShapeDimRange(padding, idx_h, attr_params_h, fm_range, out_range);
  }

  if (x_shape[idx_w] == -1) {
    y_shape[idx_w] = -1;
    map<std::string, int32_t> attr_params_w = {
      {"stride", strw}, {"dilation", dilw},
      {"pad", padl + padr}, {"kernel", kw}
    };
    // attr_params_w data structure should keep same as SetConv3dOutShapeDimRange
    SetConv3dOutShapeDimRange(padding, idx_w, attr_params_w, fm_range, out_range);
  }

  y_tensor.SetShape(Shape(y_shape));
  y_tensor.SetShapeRange(out_range);

  return true;
}

static bool NormalizeConv3dShape(const op::Conv3D& op, vector<int64_t>& x_shape_new, vector<int64_t>& w_shape_new) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  auto x_tensor = op.get_input_desc_x();
  auto x_format = x_tensor.GetFormat();
  auto x_shape = x_tensor.GetShape().GetDims();
  bool unknown_rank = IsUnknownRankShape(x_shape);
  if (!((x_shape.size() == kConv3dInputSizeLimit) || unknown_rank)) {
    OP_LOGE(op_name.GetString(), "x_shape's shape should be 5d or -2.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "x_shape";
    err_map["op_name"] = "Conv3DInfer";
    err_map["excepted_value"] = std::to_string(kConv3dInputSizeLimit);
    err_map["input_value"] = std::to_string(x_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  int32_t in = -1;
  int32_t ic = -1;
  int32_t id = -1;
  int32_t ih = -1;
  int32_t iw = -1;
  if (x_format == FORMAT_NCDHW) {
    if (!unknown_rank) {
      in = x_shape[DIM_INDEX0];
      ic = x_shape[DIM_INDEX1];
      id = x_shape[DIM_INDEX2];
      ih = x_shape[DIM_INDEX3];
      iw = x_shape[DIM_INDEX4];
    }
  } else if (x_format == FORMAT_NDHWC) {
    if (!unknown_rank) {
      in = x_shape[DIM_INDEX0];
      ic = x_shape[DIM_INDEX4];
      id = x_shape[DIM_INDEX1];
      ih = x_shape[DIM_INDEX2];
      iw = x_shape[DIM_INDEX3];
    }
  } else {
    OP_LOGE(op_name.GetString(), "input x format should be NCDHW or NDHWC.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "x_format";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "NCDHW or NDHWC";
    err_map["input_value"] = x_format;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  auto w_tensor = op.get_input_desc_filter();
  auto w_shape = w_tensor.GetShape().GetDims();
  auto w_format = w_tensor.GetFormat();
  if (w_shape.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op_name.GetString(), "w_shape's shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "w_shape";
    err_map["op_name"] = "Conv3DInfer";
    err_map["excepted_value"] = std::to_string(kConv3dInputSizeLimit);
    err_map["input_value"] = std::to_string(w_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  int32_t kn = 0;
  int32_t kc = 0;
  int32_t kd = 0;
  int32_t kh = 0;
  int32_t kw = 0;
  if (w_format == FORMAT_NCDHW) {
    kn = w_shape[DIM_INDEX0];
    kc = w_shape[DIM_INDEX1];
    kd = w_shape[DIM_INDEX2];
    kh = w_shape[DIM_INDEX3];
    kw = w_shape[DIM_INDEX4];
  } else if (w_format == FORMAT_NDHWC) {
    kn = w_shape[DIM_INDEX0];
    kc = w_shape[DIM_INDEX4];
    kd = w_shape[DIM_INDEX1];
    kh = w_shape[DIM_INDEX2];
    kw = w_shape[DIM_INDEX3];
  } else if (w_format == FORMAT_DHWCN) {
    kn = w_shape[DIM_INDEX4];
    kc = w_shape[DIM_INDEX3];
    kd = w_shape[DIM_INDEX0];
    kh = w_shape[DIM_INDEX1];
    kw = w_shape[DIM_INDEX2];
  } else {
    OP_LOGE(op_name.GetString(), "input filter format should be NCDHW, NDHWC or DHWCN.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "wFormat";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "NCDHW or NDHWC or DHWCN";
    err_map["input_value"] = w_format;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  x_shape_new.clear();
  x_shape_new.push_back(in);
  x_shape_new.push_back(id);
  x_shape_new.push_back(ih);
  x_shape_new.push_back(iw);
  x_shape_new.push_back(ic);

  w_shape_new.clear();
  w_shape_new.push_back(kn);
  w_shape_new.push_back(kd);
  w_shape_new.push_back(kh);
  w_shape_new.push_back(kw);
  w_shape_new.push_back(kc);

  return true;
}


IMPLEMT_INFERFUNC(Conv3D, Conv3DInfer) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter Conv3DInfer.");

  auto x_tensor = op.get_input_desc_x();
  auto x_format = x_tensor.GetFormat();
  auto x_shape = x_tensor.GetShape().GetDims();
  auto w_tensor = op.get_input_desc_filter();

  bool unknown_rank = IsUnknownRankShape(x_shape);
  vector<int64_t> x_shape_new;
  vector<int64_t> w_shape_new;
  if (!NormalizeConv3dShape(op, x_shape_new, w_shape_new)) {
      return GRAPH_FAILED;
  }

  int32_t in = x_shape_new[DIM_INDEX0];
  int32_t id = x_shape_new[DIM_INDEX1];
  int32_t ih = x_shape_new[DIM_INDEX2];
  int32_t iw = x_shape_new[DIM_INDEX3];
  int32_t ic = x_shape_new[DIM_INDEX4];

  int32_t kn = w_shape_new[DIM_INDEX0];
  int32_t kd = w_shape_new[DIM_INDEX1];
  int32_t kh = w_shape_new[DIM_INDEX2];
  int32_t kw = w_shape_new[DIM_INDEX3];
  int32_t kc = w_shape_new[DIM_INDEX4];

  int64_t group = 1;
  if (GRAPH_SUCCESS != op.GetAttr("groups", group)) {
    OP_LOGI(op_name.GetString(), "no group setting, use group as 1");
  }

  if (ic == -1) {
    // print warn in IsDHWUnknown later
    ic = kc * group;
  }

  if ((!unknown_rank) && (ic != kc * group)) {
    OP_LOGE(op_name.GetString(), "input x channel should be equal to filter.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3d";
    err_map["channel_of_x"] = std::to_string(ic);
    err_map["channel_of_filter"] = std::to_string(kc * group);
    std::string report_error_code = "E50039";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  int32_t strd = 0;
  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dild = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;
  if (!GetAttrsConv3D(op, x_format, strd, strh, strw, dild, dilh, dilw)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "get attrs failed.");
    return GRAPH_FAILED;
  }

  int32_t padf = 0;
  int32_t padba = 0;
  int32_t padt = 0;
  int32_t padb = 0;
  int32_t padl = 0;
  int32_t padr = 0;
  if (!GetPadConv3D(op, id, ih, iw, kd, kh, kw, strd, strh, strw, dild, dilh, dilw, padf, padba, padt, padb,
                            padl, padr)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "get pads attrs failed.");
    return GRAPH_FAILED;
  }

  int64_t od = (id + padf + padba - dild * (kd - 1) - 1) / strd + 1;
  int64_t oh = (ih + padt + padb - dilh * (kh - 1) - 1) / strh + 1;
  int64_t ow = (iw + padl + padr - dilw * (kw - 1) - 1) / strw + 1;
  if (unknown_rank) {
      od = -1;
      oh = -1;
      ow = -1;
  }

  vector<int64_t> y_shape;
  auto y_tensor = op.get_output_desc_y();
  auto y_format = y_tensor.GetFormat();

  if (y_format == FORMAT_NCDHW) {
    y_shape.push_back(in);
    y_shape.push_back(kn);
    y_shape.push_back(od);
    y_shape.push_back(oh);
    y_shape.push_back(ow);
  } else if (y_format == FORMAT_NDHWC) {
    y_shape.push_back(in);
    y_shape.push_back(od);
    y_shape.push_back(oh);
    y_shape.push_back(ow);
    y_shape.push_back(kn);
  } else {
    OP_LOGE(op_name.GetString(), "output y format should be NCDHW or NDHWC.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "yFormat";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "NCDHW or NDHWC";
    err_map["input_value"] = y_format;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  y_tensor.SetShape(Shape(y_shape));
  auto x_dtype = x_tensor.GetDataType();
  y_tensor.SetDataType(x_dtype);

  // set dynamic out range
  map<std::string, int32_t> attr_params = {
    {"strd", strd}, {"strh", strh}, {"strw", strw},
    {"dild", dild}, {"dilh", dilh}, {"dilw", dilw},
    {"padf", padf}, {"padba", padba}, {"padt", padt},
    {"padb", padb}, {"padl", padl}, {"padr", padr},
    {"kn", kn}, {"kd", kd}, {"kh", kh}, {"kw", kw}
  };
  // attr_params data structure should keep same as SetConv3dOutShapeRange
  // GE will convert y_shape to only one -2 if shape contains -2, so can't get y_shape via y_tensor.GetShape
  if (!SetConv3dOutShapeRange(op, attr_params, y_shape, y_tensor)) {
    return GRAPH_FAILED;
  }

  CHECK_OP_FUNC(GRAPH_SUCCESS != op.update_output_desc_y(y_tensor),
                return GRAPH_FAILED, "update output desc failed.");

  OP_LOGD(op_name.GetString(), "Leave Conv3DInfer.");
  return GRAPH_SUCCESS;
}

static void InferHWConv3d(const SliceTuple& infer_dhw,
                          const vector<int64_t>& output,
                          vector<int64_t>& input,
                          vector<int32_t>& pad_list,
                          uint32_t pad_idx) {
  int32_t kernel = 0;
  int32_t dilation = 0;
  int32_t stride = 0;
  int32_t input_size = 0;
  std::tie(kernel, dilation, stride, input_size) = infer_dhw;
  int32_t kernel_size = (kernel - 1) * dilation + 1;
  int32_t pad_h = pad_list[pad_idx];
  if (input_size > 0) {
    input[0] = std::max(stride * output[0] - pad_h, 0L);
    input[1] = std::min(stride * output[1] - pad_h + kernel_size - 1,
                        static_cast<int64_t>(input_size - 1));

    pad_list[pad_idx] = std::max(static_cast<int32_t>(pad_h - stride * output[0]), 0);
    pad_list[pad_idx + 1] = std::max(static_cast<int32_t>(
                                        stride * output[1] - pad_h +
                                        kernel_size - input_size),
                                     0);
  } else {
    input[0] = -1;
    input[1] = -1;
  }
}

static graphStatus VerifyDataSlice(const ge::Operator& op, const vector<vector<int64_t>>& data_slice) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  // check data_slice attr
  if (data_slice.size() != kConv3dDataSlice) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "y_data_slice's size should be 6.");
    return GRAPH_FAILED;
  }

  // no support C0 axis
  if (data_slice[kConv3dDataSlice - 1].size() != 0) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "no support to cut C0 axis.");
    return NOT_SUPPORT_SLICE;
  }

  // check valid slice num in data slice
  int32_t valid_cnt = 0;
  for (uint32_t i = 0; i < data_slice.size(); ++i) {
    if (data_slice[i].size() == 0) {
      continue;
    }
    CHECK_OP_FUNC(data_slice[i].size() != kDataSliceLen, return GRAPH_FAILED,
        "data slice format input size should be 2.");
    valid_cnt ++;
  }
  if (valid_cnt == 0) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "data slice is empty.");
    return GRAPH_FAILED;
  }
  if (valid_cnt != 1) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "valid slice range num is more than 1.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFER_DATA_SLICE(Conv3D, Conv3DInferSliceData) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter Conv3DInferSliceData.");

  auto x_format = op.get_input_desc_x().GetFormat();
  int32_t strd = 0;
  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dild = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;
  CHECK_OP_FUNC(!GetAttrsConv3D(op, x_format, strd, strh, strw, dild, dilh, dilw), return GRAPH_FAILED,
                "get attrs failed.");

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
  GeTensorDescPtr tensor_desc_w = op_desc->MutableInputDesc("filter");
  CHECK_PTR_NULL(tensor_desc_y, "tensor y desc", return GRAPH_FAILED);
  CHECK_PTR_NULL(tensor_desc_x, "tensor x desc", return GRAPH_FAILED);
  CHECK_PTR_NULL(tensor_desc_w, "tensor filter desc", return GRAPH_FAILED);

  vector<vector<int64_t>> y_data_slice = {{}, {}, {}, {}, {}, {}};
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}, {}};
  vector<vector<int64_t>> w_data_slice = {{}, {}, {}, {}};
  vector<vector<int64_t>> bias_data_slice = {{}};

  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op_name.GetString(), "no data slice, need not infer input.");
    return GRAPH_FAILED;
  }

  graphStatus ret = VerifyDataSlice(op, y_data_slice);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }

  bool needUpdateX = false;
  bool needUpdateW = false;

  auto x_shape = op.get_input_desc_x().GetShape().GetDims();
  CHECK_KEY_IN_MAP(format2str, x_format, "x_format", return GRAPH_FAILED);
  std::string x_format_str = format2str[x_format];
  int32_t d_input_position = x_format_str.find("D");
  CHECK_OP_FUNC(d_input_position < 0, return GRAPH_FAILED, "get position failed: %d", d_input_position);
  int32_t h_input_position = x_format_str.find("H");
  CHECK_OP_FUNC(h_input_position < 0, return GRAPH_FAILED, "get position failed: %d", h_input_position);
  int32_t w_input_position = x_format_str.find("W");
  CHECK_OP_FUNC(w_input_position < 0, return GRAPH_FAILED, "get position failed: %d", w_input_position);
  int32_t n_input_position = x_format_str.find("N");
  CHECK_OP_FUNC(n_input_position < 0, return GRAPH_FAILED, "get position failed: %d", n_input_position);

  int32_t id = -1;
  int32_t ih = -1;
  int32_t iw = -1;
  int32_t in = -1;

  if (x_shape != DYNAMIC_DIM_ALL) {
    CHECK_OP_FUNC(x_shape.size() < kConv3dInputSizeLimit, return GRAPH_FAILED, "x_shape is error.");
    id = x_shape[d_input_position];
    ih = x_shape[h_input_position];
    iw = x_shape[w_input_position];
    in = x_shape[n_input_position];
  }

  auto filter_format = op.get_input_desc_filter().GetFormat();
  auto w_shape = op.get_input_desc_filter().GetShape().GetDims();
  CHECK_KEY_IN_MAP(format2str, filter_format, "filter_format", return GRAPH_FAILED);
  std::string filter_format_str = format2str[filter_format];
  int32_t d_filter_position = filter_format_str.find("D");
  CHECK_OP_FUNC(d_filter_position < 0, return GRAPH_FAILED, "get position failed: %d", d_filter_position);
  int32_t h_filter_position = filter_format_str.find("H");
  CHECK_OP_FUNC(h_filter_position < 0, return GRAPH_FAILED, "get position failed: %d", h_filter_position);
  int32_t w_filter_position = filter_format_str.find("W");
  CHECK_OP_FUNC(w_filter_position < 0, return GRAPH_FAILED, "get position failed: %d", w_filter_position);
  int32_t kd = w_shape[d_filter_position];
  int32_t kh = w_shape[h_filter_position];
  int32_t kw = w_shape[w_filter_position];
  vector<int32_t> pad_list;
  op.GetAttr("pads", pad_list);
  CHECK_OP_FUNC(pad_list.size() != kConv3dPadsSizeLimit, return GRAPH_FAILED, "the pad_list is illegal.");
  SliceTuple infer_dhw(kd, dild, strd, id);
  // cut N
  if (y_data_slice[0].size() != 0) {
    x_data_slice[0] = y_data_slice[0];
    if (in < 0) {
      x_data_slice[0] = {-1, -1};
    }
    needUpdateX = true;
  }

  // cut D
  if (y_data_slice[kDDimNDC1HWC0Idx].size() != 0) {
    x_data_slice[kDDimNDC1HWC0Idx].clear();
    x_data_slice[kDDimNDC1HWC0Idx].resize(kDataSliceLen);
    InferHWConv3d(infer_dhw,
                  y_data_slice[1], x_data_slice[1], pad_list, kConv3dPadHeadIdx);
    needUpdateX = true;
  }

  // cut Cout
  if (y_data_slice[kC1DimNDC1HWC0Idx].size() != 0) {
    w_data_slice[1] = y_data_slice[kC1DimNDC1HWC0Idx];
    bias_data_slice[0] = y_data_slice[kC1DimNDC1HWC0Idx];
    needUpdateW = true;
  }

  // cut H
  if (y_data_slice[kHDimNDC1HWC0Idx].size() != 0) {
    x_data_slice[kHDimNDC1HWC0Idx].clear();
    x_data_slice[kHDimNDC1HWC0Idx].resize(kDataSliceLen);
    infer_dhw = std::make_tuple(kh, dilh, strh, ih);
    InferHWConv3d(infer_dhw,
                  y_data_slice[kHDimNDC1HWC0Idx], x_data_slice[kHDimNDC1HWC0Idx], pad_list, kConv3dPadUpIdx);
    needUpdateX = true;
  }

  // cut W
  if (y_data_slice[kWDimNDC1HWC0Idx].size() != 0) {
    x_data_slice[kWDimNDC1HWC0Idx].clear();
    x_data_slice[kWDimNDC1HWC0Idx].resize(kDataSliceLen);
    infer_dhw = std::make_tuple(kw, dilw, strw, iw);
    InferHWConv3d(infer_dhw,
                  y_data_slice[kWDimNDC1HWC0Idx], x_data_slice[kWDimNDC1HWC0Idx], pad_list, kConv3dPadLeftIdx);
    needUpdateX = true;
  }

  // check update flag
  CHECK_OP_FUNC(!needUpdateX && !needUpdateW, return GRAPH_FAILED, "there's no update in desc.");

  // update data slice attr
  if (needUpdateX) {
    CHECK_OP_FUNC(!AttrUtils::SetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice),
                  return GRAPH_FAILED,
                  "set x data slice attr failed.");
  }
  if (needUpdateW) {
    CHECK_OP_FUNC(!AttrUtils::SetListListInt(tensor_desc_w, ge::ATTR_NAME_DATA_SLICE, w_data_slice),
                  return GRAPH_FAILED,
                  "set w data slice attr failed");
  }

  // update pads attr info
  op.SetAttr("pads", pad_list);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv3D, Conv3DVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter Conv3DVerify.");
  auto x_tensor = op.GetInputDescByName("x");
  auto w_tensor = op.GetInputDescByName("filter");

  auto x_shape = x_tensor.GetOriginShape().GetDims();
  auto w_shape = w_tensor.GetOriginShape().GetDims();
  bool unknown_rank = IsUnknownRankShape(x_shape);
  if (!((x_shape.size() == kConv3dInputSizeLimit) || unknown_rank)) {
    OP_LOGE(op_name.GetString(), "input x shape should be 5d or -2.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "xShape_size";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = std::to_string(kConv3dInputSizeLimit);
    err_map["output_value"] = std::to_string(x_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  if (w_shape.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op_name.GetString(), "input filter shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "w_shape_size";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = std::to_string(kConv3dInputSizeLimit);
    err_map["output_value"] = std::to_string(w_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  auto x_dtype = x_tensor.GetDataType();
  auto w_dtype = w_tensor.GetDataType();

  if (x_dtype != w_dtype) {
    OP_LOGE(op_name.GetString(), "input x dtype is differ from filter dtype.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3d";
    err_map["attr_name"] = "dtype";
    err_map["param1_name"] = "input x";
    err_map["param2_name"] = "weight";
    err_map["param1_value"] = std::to_string(x_dtype);
    err_map["param2_value"] = std::to_string(w_dtype);
    std::string report_error_code = "E50031";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> stride_list;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stride_list)) {
    OP_LOGE(op_name.GetString(), "get strides list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3d";
    err_map["op_name"] = "strides";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  if (stride_list.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op_name.GetString(), "strides list should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = std::to_string(kConv3dInputSizeLimit);
    err_map["input_value"] = std::to_string(stride_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  std::vector<int32_t> dilation_list;
  if (!VerifyConv3dDilations(op, dilation_list)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "get dilation attrs failed.");
    return GRAPH_FAILED;
  }

  OP_LOGD(op_name.GetString(), "Leave Conv3DVerify.");
  return GRAPH_SUCCESS;
}

INFER_DATA_SLICE_FUNC_REG(Conv3D, Conv3DInferSliceData);
INFER_FUNC_REG(Conv3D, Conv3DInfer);
VERIFY_FUNC_REG(Conv3D, Conv3DVerify);

// -----------------------------conv3dbp_common_check-----------------------------
template <typename T1, typename T2>
static bool SetPadListByPaddingConv3dbp(ge::Operator& op, const std::vector<T1>& input_sizes, Format input_format,
                                        const std::vector<T2>& filter_sizes, Format filter_format) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  if (filter_sizes.size() < kConv3dInputSizeLimit || input_sizes.size() < kConv3dInputSizeLimit) {
    OP_LOGE(op_name.GetString(), "filter_sizes or inputSizes is illegal");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "filter_size and inputsize";
    err_map["op_name"] = "Conv3dbp";
    err_map["excepted_value"] = std::to_string(kConv3dInputSizeLimit);
    err_map["input_value"] = std::to_string(filter_sizes.size()) + " " + std::to_string(input_sizes.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  CHECK_OP_FUNC(FORMAT_RESERVED == input_format, return false, "get format failed: %d", input_format);
  CHECK_OP_FUNC(FORMAT_RESERVED == filter_format, return false, "get format failed: %d", filter_format);

  std::vector<int32_t> stride_list;
  if (GRAPH_FAILED == op.GetAttr("strides", stride_list)) {
    OP_LOGE(op_name.GetString(), "op get strides failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbp";
    err_map["param_name"] = "strides";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (stride_list.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op_name.GetString(), "strides list should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = "Conv3dbp";
    err_map["excepted_value"] = std::to_string(kConv3dInputSizeLimit);
    err_map["input_value"] = std::to_string(stride_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::vector<int32_t> dilations_list;
  if (!VerifyConv3dDilations(op, dilations_list)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "get dilation attrs failed.");
    return false;
  }
  CHECK_KEY_IN_MAP(format2str, input_format, "input_format", return false);
  std::string input_format_str = format2str[input_format];
  int32_t h_input_position = input_format_str.find("H");
  CHECK_OP_FUNC(h_input_position < 0, return false, "get position failed: %d", h_input_position);
  int32_t w_input_position = input_format_str.find("W");
  CHECK_OP_FUNC(w_input_position < 0, return false, "get position failed: %d", w_input_position);
  int32_t d_input_position = input_format_str.find("D");
  CHECK_OP_FUNC(d_input_position < 0, return false, "get position failed: %d", d_input_position);
  int32_t dx_h = input_sizes[h_input_position];
  int32_t dx_w = input_sizes[w_input_position];
  int32_t dx_d = input_sizes[d_input_position];

  int32_t stride_h = stride_list[h_input_position];
  int32_t stride_w = stride_list[w_input_position];
  int32_t stride_d = stride_list[d_input_position];
  CHECK_KEY_IN_MAP(format2str, filter_format, "filter_format", return false);
  std::string filter_format_str = format2str[filter_format];
  int32_t h_filter_position = filter_format_str.find("H");
  CHECK_OP_FUNC(h_filter_position < 0, return false, "get position failed: %d", h_filter_position);
  int32_t w_filter_position = filter_format_str.find("W");
  CHECK_OP_FUNC(w_filter_position < 0, return false, "get position failed: %d", w_filter_position);
  int32_t d_filter_position = filter_format_str.find("D");
  CHECK_OP_FUNC(d_filter_position < 0, return false, "get position failed: %d", d_filter_position);

  int32_t filter_h = filter_sizes[h_filter_position];
  int32_t filter_w = filter_sizes[w_filter_position];
  int32_t filter_d = filter_sizes[d_filter_position];

  int32_t dilation_h = dilations_list[h_input_position];
  int32_t dilation_w = dilations_list[w_input_position];
  int32_t dilation_d = dilations_list[d_input_position];

  std::string padding;
  std::vector<int32_t> pads;
  if (GRAPH_SUCCESS == op.GetAttr("padding", padding)) {
    OP_LOGI(op_name.GetString(), "op get padding succ.");
    int pad_h = 0;
    int32_t pad_up = 0;
    int32_t pad_down = 0;
    int pad_w = 0;
    int32_t pad_left = 0;
    int32_t pad_right = 0;
    int pad_d = 0;
    int32_t pad_head = 0;
    int32_t pad_tail = 0;
    if (padding == "SAME") {
      pad_h = std::max(align_conv2dbp(dx_h, stride_h) - stride_h + (filter_h - 1) * dilation_h + 1 - dx_h, 0);
      pad_up = pad_h >> 1;
      pad_down = pad_h - pad_up;
      pad_w = std::max(align_conv2dbp(dx_w, stride_w) - stride_w + (filter_w - 1) * dilation_w + 1 - dx_w, 0);
      pad_left = pad_w >> 1;
      pad_right = pad_w - pad_left;
      pad_d = std::max(align_conv2dbp(dx_d, stride_d) - stride_d + (filter_d - 1) * dilation_d + 1 - dx_d, 0);
      pad_head = pad_d >> 1;
      pad_tail = pad_d - pad_head;
    }

    pads.push_back(pad_head);
    pads.push_back(pad_tail);
    pads.push_back(pad_up);
    pads.push_back(pad_down);
    pads.push_back(pad_left);
    pads.push_back(pad_right);

    op.SetAttr("pads", pads);
  }
  if (GRAPH_SUCCESS == op.GetAttr("pads", pads)) {
    if (pads.size() < kConv3dPadsSizeLimit) {
      OP_LOGE(op_name.GetString(), "op pads's size is illegal,pads.");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "pads";
      err_map["op_name"] = "Conv3dbp";
      err_map["excepted_value"] = std::to_string(kConv3dPadsSizeLimit);
      err_map["input_value"] = std::to_string(pads.size());
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }

    auto out_backprop_shape = op.GetInputDescByName("out_backprop").GetShape().GetDims();
    if (!IsUnknownRankShape(out_backprop_shape) && !CheckVectorAnyNegative(pads)) {
      OP_LOGE(op_name.GetString(), "op get pads is illegal");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "pads";
      err_map["op_name"] = "Conv3dbp";
      err_map["excepted_value"] = "Non-negative";
      err_map["input_value"] = ops::to_string(pads);
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return false;
    }
  } else {
    OP_LOGE(op_name.GetString(), "op get pads failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbp";
    err_map["param_name"] = "pads";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  OP_LOGI(op_name.GetString(), "op set pads succ.");
  return true;
}

static graphStatus VerifyConv3dbpInputCommon(const ge::Operator& op) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  auto filter_desc = op.GetInputDescByName("filter");
  auto filter_dtype = filter_desc.GetDataType();
  auto out_backprop_desc = op.GetInputDescByName("out_backprop");
  auto out_backprop_dtype = out_backprop_desc.GetDataType();
  // check input dtype
  if (filter_dtype != out_backprop_dtype) {
    OP_LOGE(op_name.GetString(), "filter's dtype should equal to outBackprop's dtype.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpInput";
    err_map["attr_name"] = "dtype";
    err_map["param1_name"] = "filter";
    err_map["param2_name"] = "outBackprop";
    err_map["param1_value"] = std::to_string(filter_dtype);
    err_map["param2_value"] = std::to_string(out_backprop_dtype);
    std::string report_error_code = "E50031";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check input tensor shape
  auto filter_shape = filter_desc.GetShape().GetDims();
  if (filter_shape.size() != kConv3dDimSizeLimit) {
    OP_LOGE(op_name.GetString(), "filter's shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "filterShape_size";
    err_map["op_name"] = "Conv3dbpInput";
    err_map["excepted_value"] = std::to_string(kConv3dDimSizeLimit);
    err_map["input_value"] = std::to_string(filter_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  auto out_backprop_shape = out_backprop_desc.GetShape().GetDims();
  if (!IsUnknownRankShape(out_backprop_shape) && out_backprop_shape.size() != kConv3dDimSizeLimit) {
    OP_LOGE(op_name.GetString(), "outBackprop's shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "outBackpropShape_size";
    err_map["op_name"] = "Conv3dbpInput";
    err_map["excepted_value"] = std::to_string(kConv3dDimSizeLimit);
    err_map["input_value"] = std::to_string(out_backprop_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check strides shape
  std::vector<int32_t> stride_list;
  if (GRAPH_SUCCESS == op.GetAttr("strides", stride_list)) {
    if (stride_list.size() != kConv3dDimSizeLimit) {
      OP_LOGE(op_name.GetString(), "strides should be 5d.");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "strides";
      err_map["op_name"] = "Conv3dbpInput";
      err_map["excepted_value"] = std::to_string(kConv3dDimSizeLimit);
      err_map["input_value"] = std::to_string(stride_list.size());
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op_name.GetString(), "get strides list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpInput";
    err_map["param_name"] = "strides";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check dilations shape
  std::vector<int32_t> dilations_list;
  if (!VerifyConv3dDilations(op, dilations_list)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "get dilation attrs failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

static graphStatus VerifyConv3dbpPads(const ge::Operator& op, bool is_dynamic = false) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  std::vector<int> pads;
  if (GRAPH_SUCCESS == op.GetAttr("pads", pads)) {
    if (pads.size() < kConv3dPadsSizeLimit) {
      OP_LOGE(op_name.GetString(), "op pads's size is illegal,pads.");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "pads";
      err_map["op_name"] = "Conv3dbpInput";
      err_map["excepted_value"] = std::to_string(kConv3dPadsSizeLimit);
      err_map["input_value"] = std::to_string(pads.size());
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }

    if (!is_dynamic && !CheckVectorAnyNegative(pads)) {
      OP_LOGE(op_name.GetString(), "op get pads is illegal");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "pads";
      err_map["op_name"] = "Conv3dbpInput";
      err_map["excepted_value"] = "positive";
      err_map["input_value"] = ops::to_string(pads);
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op_name.GetString(), "op get pads failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpInput";
    err_map["param_name"] = "pads";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

static void SetConv3dBpInputOutShapeDimRange(const std::string& pad_str,
                                             size_t idx,
                                             const vector<int64_t>& attrParams,
                                             const std::vector<std::pair<int64_t, int64_t>>& dy_range,
                                             std::vector<std::pair<int64_t, int64_t>>& dx_range) {
  size_t attrIdx = 0;
  int64_t stride = attrParams[attrIdx++];
  int64_t kernel = attrParams[attrIdx++];
  int64_t pad = attrParams[attrIdx++];
  int64_t low = dy_range[idx].first;
  int64_t high = dy_range[idx].second;

  if (pad_str == "SAME") {
    dx_range[idx].first = stride * (low - 1) + 1;
    dx_range[idx].second = stride * high;
  } else {
    dx_range[idx].first = stride * (low - 1) + kernel - pad;
    dx_range[idx].second = stride * (high - 1) + kernel - pad + stride - 1;
  }

  dx_range[idx].first = std::max(dx_range[idx].first, kDynamicRangeLowerBound);
  if (high == -1) {
    dx_range[idx].second = high;
  } else {
    dx_range[idx].second = std::min(dx_range[idx].second, kDynamicRangeUpperBound);
  }
}

static bool SetConv3dBpInputOutShapeRange(ge::Operator& op, bool unknown_rank,
                                          const std::vector<std::pair<int64_t, int64_t>>& dy_range,
                                          std::vector<std::pair<int64_t, int64_t>>& dx_range) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return false);
  auto filter_desc = op_desc->MutableInputDesc("filter");
  CHECK_PTR_NULL(filter_desc, "tensor filter desc", return false);
  auto y_desc = op_desc->MutableOutputDesc("y");
  CHECK_PTR_NULL(y_desc, "tensor y desc", return false);
  std::vector<int64_t> filter_sizes = filter_desc->MutableShape().GetDims();
  std::vector<int64_t> dx_sizes = y_desc->MutableShape().GetDims();
  bool shape_check_flag =
      filter_sizes.size() < kConv3dInputSizeLimit || (!dx_sizes.empty() && dx_sizes.size() < kConv3dInputSizeLimit);
  if (shape_check_flag) {
    OP_LOGE(op_name.GetString(), "filter_sizes or dx_sizes is illegal");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "filter_size and dx_sizes";
    err_map["op_name"] = "Conv3dbp";
    err_map["excepted_value"] = std::to_string(kConv3dInputSizeLimit);
    err_map["input_value"] = std::to_string(filter_sizes.size()) + " " + std::to_string(dx_sizes.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  Format filter_format = filter_desc->GetFormat();
  CHECK_OP_FUNC(FORMAT_RESERVED == filter_format, return false, "get format failed: %d", filter_format);
  CHECK_KEY_IN_MAP(format2str, filter_format, "filter_format", return false);
  std::string filter_format_str = format2str[filter_format];
  int32_t d_filter_position = filter_format_str.find("D");
  CHECK_OP_FUNC(d_filter_position < 0, return false, "get position failed: %d", d_filter_position);
  int32_t h_filter_position = filter_format_str.find("H");
  CHECK_OP_FUNC(h_filter_position < 0, return false, "get position failed: %d", h_filter_position);
  int32_t w_filter_position = filter_format_str.find("W");
  CHECK_OP_FUNC(w_filter_position < 0, return false, "get position failed: %d", w_filter_position);
  int32_t c_filter_position = filter_format_str.find("C");
  CHECK_OP_FUNC(c_filter_position < 0, return false, "get position failed: %d", c_filter_position);

  int64_t filter_d = filter_sizes[d_filter_position];
  int64_t filter_h = filter_sizes[h_filter_position];
  int64_t filter_w = filter_sizes[w_filter_position];
  int64_t filter_c = filter_sizes[c_filter_position];

  Format input_format = y_desc->GetFormat();
  CHECK_OP_FUNC(FORMAT_RESERVED == input_format, return false, "get format failed: %d", input_format);
  CHECK_KEY_IN_MAP(format2str, input_format, "input_format", return false);
  std::string input_format_str = format2str[input_format];
  int32_t n_input_position = input_format_str.find("N");
  CHECK_OP_FUNC(n_input_position < 0, return false, "get position failed: %d", n_input_position);
  int32_t d_input_position = input_format_str.find("D");
  CHECK_OP_FUNC(d_input_position < 0, return false, "get position failed: %d", d_input_position);
  int32_t h_input_position = input_format_str.find("H");
  CHECK_OP_FUNC(h_input_position < 0, return false, "get position failed: %d", h_input_position);
  int32_t w_input_position = input_format_str.find("W");
  CHECK_OP_FUNC(w_input_position < 0, return false, "get position failed: %d", w_input_position);
  int32_t c_input_position = input_format_str.find("C");
  CHECK_OP_FUNC(c_input_position < 0, return false, "get position failed: %d", c_input_position);

  int64_t groups = 1;
  if (op.GetAttr("groups", groups) != GRAPH_SUCCESS) {
    OP_LOGI(op_name.GetString(), "no groups setting, use groups as 1");
  }

  if (unknown_rank) {
    vector<int64_t> dx_shape;
    dx_shape.resize(kConv3dInputSizeLimit, -1);
    dx_shape[c_input_position] = groups * filter_c;
    y_desc->SetShape(GeShape(dx_shape));
    return true;
  }

  if (dx_sizes.empty()) {
    dx_sizes.resize(kConv3dInputSizeLimit, -1);
    dx_sizes[c_input_position] = groups * filter_c;
    y_desc->SetShape(GeShape(dx_sizes));
  }

  std::vector<int32_t> stride_list;
  if (op.GetAttr("strides", stride_list) == GRAPH_FAILED) {
    OP_LOGE(op_name.GetString(), "op get strides failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbp";
    err_map["param_name"] = "strides";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (stride_list.size() != kConv3dStridesSizeLimit) {
    OP_LOGE(op_name.GetString(), "strides list should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = "Conv3dbp";
    err_map["excepted_value"] = std::to_string(kConv3dStridesSizeLimit);
    err_map["input_value"] = std::to_string(stride_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::vector<int32_t> dilations_list;
  CHECK_OP_FUNC(!VerifyConv3dDilations(op, dilations_list), return false, "get dilation attrs failed.");

  int64_t dx_d = dx_sizes[d_input_position];
  int64_t dx_h = dx_sizes[h_input_position];
  int64_t dx_w = dx_sizes[w_input_position];

  int32_t stride_d = stride_list[d_input_position];
  int32_t stride_h = stride_list[h_input_position];
  int32_t stride_w = stride_list[w_input_position];

  int32_t dilation_d = dilations_list[d_input_position];
  int32_t dilation_h = dilations_list[h_input_position];
  int32_t dilation_w = dilations_list[w_input_position];

  std::vector<int32_t> pads_list;
  op.GetAttr("pads", pads_list);
  if (pads_list.size() < kConv3dPadsSizeLimit) {
    pads_list.assign(kConv3dPadsSizeLimit, 0);
  }

  int32_t pad_front = pads_list[kConv3dPadHeadIdx];
  int32_t pad_back = pads_list[kConv3dPadTailIdx];
  int32_t pad_up = pads_list[kConv3dPadUpIdx];
  int32_t pad_down = pads_list[kConv3dPadDownIdx];
  int32_t pad_left = pads_list[kConv3dPadLeftIdx];
  int32_t pad_right = pads_list[kConv3dPadRightIdx];

  int64_t kdext = (filter_d - 1) * dilation_d + 1;
  int64_t khext = (filter_h - 1) * dilation_h + 1;
  int64_t kwext = (filter_w - 1) * dilation_w + 1;

  int64_t dy_n = -1;
  int32_t n_dy_position = -1;
  AscendString op_type;
  CHECK(op.GetOpType(op_type) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_type"), return false);
  if (string(op_type.GetString()) == "Conv3DTranspose") {
    auto x_desc = op_desc->MutableInputDesc("x");
    CHECK_PTR_NULL(x_desc, "tensor x desc", return false);
    Format x_format = x_desc->GetFormat();
    CHECK_OP_FUNC(FORMAT_RESERVED == x_format, return false, "get format failed: %d", x_format);
    CHECK_KEY_IN_MAP(format2str, x_format, "x_format", return false);
    std::string x_format_str = format2str[x_format];
    n_dy_position = x_format_str.find("N");
    CHECK_OP_FUNC(n_dy_position < 0, return false, "get position failed: %d", n_dy_position);
    int32_t d_x_position = x_format_str.find("D");
    CHECK_OP_FUNC(d_x_position < 0, return false, "get position failed: %d", d_x_position);
    int32_t h_x_position = x_format_str.find("H");
    CHECK_OP_FUNC(h_x_position < 0, return false, "get position failed: %d", h_x_position);
    int32_t w_x_position = x_format_str.find("W");
    CHECK_OP_FUNC(w_x_position < 0, return false, "get position failed: %d", w_x_position);

    std::vector<int32_t> output_padding_list;
    op.GetAttr("output_padding", output_padding_list);
    int32_t outputpadding_d = output_padding_list[d_x_position];
    int32_t outputpadding_h = output_padding_list[h_x_position];
    int32_t outputpadding_w = output_padding_list[w_x_position];
    kdext = outputpadding_d + ((filter_d - 1) * dilation_d + 1);
    khext = outputpadding_h + ((filter_h - 1) * dilation_h + 1);
    kwext = outputpadding_w + ((filter_w - 1) * dilation_w + 1);
    dy_n = x_desc->MutableShape().GetDims()[n_dy_position];
  } else {
    auto dy_desc = op_desc->MutableInputDesc("out_backprop");
    CHECK_PTR_NULL(dy_desc, "tensor dy desc", return false);
    Format dy_format = dy_desc->GetFormat();
    CHECK_OP_FUNC(FORMAT_RESERVED == dy_format, return false, "get format failed: %d", dy_format);
    CHECK_KEY_IN_MAP(format2str, dy_format, "dy_format", return false);
    std::string dy_format_str = format2str[dy_format];
    n_dy_position = dy_format_str.find("N");
    CHECK_OP_FUNC(n_dy_position < 0, return false, "get position failed: %d", n_dy_position);
    dy_n = dy_desc->MutableShape().GetDims()[n_dy_position];
  }

  dx_range.resize(kConv3dInputSizeLimit);
  dx_range[n_input_position] = std::make_pair(dy_n, dy_n);
  dx_range[d_input_position] = std::make_pair(dx_d, dx_d);
  dx_range[h_input_position] = std::make_pair(dx_h, dx_h);
  dx_range[w_input_position] = std::make_pair(dx_w, dx_w);
  dx_range[c_input_position] = std::make_pair(filter_c * groups, filter_c * groups);
  if (dy_range.size() == kConv3dInputSizeLimit) {
    dx_range[n_input_position] = dy_range[n_dy_position];
    std::string pad_str;
    op.GetAttr("padding", pad_str);
    if (dx_d == -1) {
      vector<int64_t> attr_params_d = {static_cast<int64_t>(stride_d),
                                       kdext,
                                       static_cast<int64_t>(pad_front + pad_back)};
      SetConv3dBpInputOutShapeDimRange(pad_str, d_input_position, attr_params_d, dy_range, dx_range);
    }
    if (dx_h == -1) {
      vector<int64_t> attr_params_h = {static_cast<int64_t>(stride_h),
                                       khext,
                                       static_cast<int64_t>(pad_up + pad_down)};
      SetConv3dBpInputOutShapeDimRange(pad_str, h_input_position, attr_params_h, dy_range, dx_range);
    }
    if (dx_w == -1) {
      vector<int64_t> attr_params_w = {static_cast<int64_t>(stride_w),
                                       kwext,
                                       static_cast<int64_t>(pad_left + pad_right)};
      SetConv3dBpInputOutShapeDimRange(pad_str, w_input_position, attr_params_w, dy_range, dx_range);
    }
    y_desc->SetShapeRange(dx_range);
  }
  return true;
}

static void ResetConv3dBpInputOutShape(Format dy_format,
                                       const std::vector<int64_t>&dy_sizes,
                                       Format input_format,
                                       std::vector<int64_t>& input_sizes) {
  CHECK_KEY_IN_MAP(format2str, input_format, "input_format", return);
  std::string dx_format_str = format2str[input_format];
  int32_t n_input_position = dx_format_str.find("N");
  int32_t d_input_position = dx_format_str.find("D");
  int32_t h_input_position = dx_format_str.find("H");
  int32_t w_input_position = dx_format_str.find("W");
  CHECK_KEY_IN_MAP(format2str, dy_format, "dy_format", return);
  std::string dy_format_str = format2str[dy_format];
  int32_t n_dy_position = dy_format_str.find("N");
  int32_t d_dy_position = dy_format_str.find("D");
  int32_t h_dy_position = dy_format_str.find("H");
  int32_t w_dy_position = dy_format_str.find("W");

  if (dy_sizes[n_dy_position] == -1) {
    input_sizes[n_input_position] = -1;
  }

  if (dy_sizes[d_dy_position] == -1) {
    input_sizes[d_input_position] = -1;
  }

  if (dy_sizes[h_dy_position] == -1) {
    input_sizes[h_input_position] = -1;
  }

  if (dy_sizes[w_dy_position] == -1) {
    input_sizes[w_input_position] = -1;
  }
}

static bool InferConv3dBpInputOutShapeRange(ge::Operator& op, GeTensorDescPtr& input_sizes_desc,
                                            const GeTensorDescPtr& dy_desc, GeTensorDescPtr& y_desc,
                                            std::vector<int64_t>& input_sizes) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  bool unknown_rank = IsUnknownRankShape(dy_desc->MutableShape().GetDims());
  std::vector<std::pair<int64_t, int64_t>> dy_range;
  dy_desc->GetShapeRange(dy_range);
  std::vector<std::pair<int64_t, int64_t>> dx_range;
  input_sizes_desc->GetValueRange(dx_range);
  if ((dx_range.size() == kConv3dDimSizeLimit) && (dy_range.size() == kConv3dDimSizeLimit)) {
    y_desc->SetShapeRange(dx_range);
    OP_LOGD(op_name.GetString(), "get value_range success from GE.");
  } else {
    if (!SetConv3dBpInputOutShapeRange(op, unknown_rank, dy_range, dx_range)) {
      return false;
    }
  }

  input_sizes.assign(dx_range.size(), -1);
  for (size_t i = 0; i < dx_range.size(); i++) {
    if (dx_range[i].first == dx_range[i].second) {
      input_sizes[i] = dx_range[i].first;
    }

    OP_LOGD(op_name.GetString(), "dedx range[%u] is (%lld, %lld)", i, dx_range[i].first, dx_range[i].second);
  }

  if (!unknown_rank) {
    ResetConv3dBpInputOutShape(dy_desc->GetFormat(), dy_desc->GetShape().GetDims(), y_desc->GetFormat(),
                               input_sizes);
  }

  return true;
}

static void InferHWConv3dBackpropInput(const SliceTuple &infer_dhw, const vector<int64_t> &output,
                                       vector<int64_t> &input, vector<int32_t> &pad_list, uint32_t pad_idx) {
  int32_t input_size = 0;
  int32_t stride = 0;
  int32_t dilation = 0;
  int32_t kernel = 0;
  std::tie(kernel, dilation, stride, input_size) = infer_dhw;
  int32_t kernel_size = (kernel - 1) * dilation + 1;
  int32_t pad_out = kernel_size - pad_list[pad_idx] - 1;
  if (input_size > 0) {
    input[0] = std::min(std::max(static_cast<int64_t>(
                                 std::ceil(
                                     static_cast<float>(output[0] - pad_out) /
                                     static_cast<float>(stride))),
                                 0L),
                        static_cast<int64_t>(input_size - 1));
    input[1] = std::min((output[1] + kernel_size - 1 - pad_out) / static_cast<int64_t>(stride),
                        static_cast<int64_t>(input_size - 1));

    int32_t oh = static_cast<int32_t>(output[1] - output[0] + 1);
    int32_t ih = static_cast<int32_t>(input[1] - input[0] + 1);

    pad_list[pad_idx] = kernel_size - static_cast<int32_t>(
                                        input[0] * stride + pad_out - output[0]) - 1;
    pad_list[pad_idx + 1] = std::max(stride * (ih - 1) + kernel_size -
                                        oh - pad_list[pad_idx],
                                     0);
  } else {
    input[0] = -1;
    input[1] = -1;
  }
}

IMPLEMT_INFERFUNC(Conv3DBackpropInput, Conv3DBackpropInputInfer) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGI(op_name.GetString(), "Enter Conv3DBackpropInput inferfunction!");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);
  std::vector<std::string> input_infer_depends = {"input_size"};
  op_desc->SetOpInferDepends(input_infer_depends);
  auto dy_desc = op_desc->MutableInputDesc("out_backprop");
  CHECK_PTR_NULL(dy_desc, "tensor dy desc", return GRAPH_FAILED);
  auto y_desc = op_desc->MutableOutputDesc("y");
  CHECK_PTR_NULL(y_desc, "tensor y desc", return GRAPH_FAILED);

  std::vector<int64_t> dy_sizes = dy_desc->MutableShape().GetDims();
  for (size_t j = 0; j < dy_sizes.size(); j++) {
    OP_LOGD(op_name.GetString(), "dy_shape [%u] is %lld", j, dy_sizes[j]);
  }

  bool is_input_size_const = false; // means dynamic shape mode if false
  const GeShape &dy_shape = dy_desc->GetShape();
  bool is_dynamic = dy_shape.IsUnknownShape();  // dynamic fuzzy build mode is_input_size_const = true
  std::vector<int64_t> input_sizes;
  Tensor input_sizes_tensor;
  auto input_sizes_desc = op_desc->MutableInputDesc("input_size");
  CHECK_PTR_NULL(input_sizes_desc, "tensor input_size desc", return GRAPH_FAILED);
  if (op.GetInputConstData("input_size", input_sizes_tensor) == GRAPH_SUCCESS) {
    DataType dtype = input_sizes_desc->GetDataType();
    GetConstValue(input_sizes_tensor, dtype, input_sizes);
    is_input_size_const = true;
  }
  if ((!is_input_size_const || is_dynamic) &&
      !InferConv3dBpInputOutShapeRange(op, input_sizes_desc, dy_desc, y_desc, input_sizes)) {
    return GRAPH_FAILED;
  }

  // set shape of output desc, input_size should match the format of y
  if (input_sizes.size() == kConv3dDimSizeLimit) {
    y_desc->SetShape(GeShape(input_sizes));
  }

  // update pads list by padding[SAME,VALID]
  std::vector<int64_t> filter_sizes = op.GetInputDescByName("filter").GetShape().GetDims();
  Format filter_format = op.GetInputDescByName("filter").GetFormat();
  Format input_format = y_desc->GetFormat();
  // if only batch is -1, no need to set SAME padding as -1
  // dy_sizes maybe contains -1 in runtime compile, but can't set pads as -1
  if ((!is_input_size_const) && (IsUnknownRankShape(dy_sizes) ||
      IsDHWUnknown(string(op_name.GetString()), "y", y_desc->MutableShape().GetDims(), y_desc->GetFormat()))) {
    std::string pad_str;
    if (op.GetAttr("padding", pad_str) == GRAPH_SUCCESS) {
      std::vector<int32_t> pads(kConv3dPadsSizeLimit, 0);
      if (pad_str == "SAME") {
        pads.assign(kConv3dPadsSizeLimit, -1);
        OP_LOGD(op_name.GetString(), "set pads to {-1, -1, -1, -1, -1, -1} when padding is SAME in dynamic_shape");
      }

      op.SetAttr("pads", pads);
    }
  } else if (!SetPadListByPaddingConv3dbp(op, y_desc->MutableShape().GetDims(),
             input_format, filter_sizes, filter_format)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "update pads list by padding failed.");
    return GRAPH_FAILED;
  }

  // set dtype of output desc
  auto out_backprop_dtype = op.GetInputDescByName("out_backprop").GetDataType();
  y_desc->SetDataType(out_backprop_dtype);

  OP_LOGI(op_name.GetString(), "Leaving Conv3DBackpropInput inferfunction!");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv3DBackpropInput, Conv3DBackpropInputVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);
  OpDescPtr op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);

  OP_LOGI(op_name.GetString(), "Enter Conv3DBackpropInput verifyfunction!");
  if (GRAPH_SUCCESS != VerifyConv3dbpInputCommon(op)) {
    return GRAPH_FAILED;
  }
  // check padding value
  auto out_backprop_shape = op.GetInputDescByName("out_backprop").GetShape().GetDims();
  bool unknown_rank = IsUnknownRankShape(out_backprop_shape);
  bool unknown_shape = IsUnKnownShape(out_backprop_shape);
  bool is_dynamic = unknown_rank || unknown_shape;
  if (VerifyConvPadding(op_name, op_desc) || GRAPH_SUCCESS == VerifyConv3dbpPads(op, is_dynamic)) {
    OP_LOGI(op_name.GetString(), "Leaving Conv3DBackpropInput verifyfunction!");
    return GRAPH_SUCCESS;
  } else {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "Leaving Conv3DBackpropInput verifyfunction!");
    return GRAPH_FAILED;
  }
}

INFER_FUNC_REG(Conv3DBackpropInput, Conv3DBackpropInputInfer);
VERIFY_FUNC_REG(Conv3DBackpropInput, Conv3DBackpropInputVerify);

// ----------------Conv3DBackpropInputD-------------------
IMPLEMT_INFERFUNC(Conv3DBackpropInputD, Conv3DBackpropInputDInfer) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGI(op_name.GetString(), "Enter Conv3DBackpropInputD inferfunction!");
  // get shape for output from input_size
  std::vector<int32_t> input_sizes;
  if (GRAPH_SUCCESS == op.GetAttr("input_size", input_sizes)) {
    if (input_sizes.size() != kConv3dDimSizeLimit) {
      OP_LOGE(op_name.GetString(), "input_size list should be 5d.");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "input_size";
      err_map["op_name"] = "Conv3dbpInput";
      err_map["excepted_value"] = std::to_string(kConv3dDimSizeLimit);
      err_map["input_value"] = std::to_string(input_sizes.size());
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op_name.GetString(), "get input_size list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpInput";
    err_map["param_name"] = "input_size";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  auto out_backprop_desc = op.GetInputDescByName("out_backprop");
  // get dtype for output from out_backprop
  auto out_backprop_dtype = out_backprop_desc.GetDataType();
  // set dtype of output desc
  auto y_desc = op.GetOutputDescByName("y");
  y_desc.SetDataType(out_backprop_dtype);
  // set shape of output desc, input_size should match the format of y
  std::vector<int64_t> out_shape;
  for (auto i : input_sizes) {
    out_shape.push_back(i);
  }
  y_desc.SetShape(ge::Shape(out_shape));

  // update output desc
  if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", y_desc)) {
    OP_LOGE(op_name.GetString(), "update output desc failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpInput";
    err_map["param_name"] = "output y";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> filter_sizes = op.GetInputDescByName("filter").GetShape().GetDims();
  Format filter_format = op.GetInputDescByName("filter").GetFormat();
  Format input_format = y_desc.GetFormat();
  // update pads list by padding[SAME,VALID]
  if (!SetPadListByPaddingConv3dbp(op, input_sizes, input_format, filter_sizes, filter_format)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "Conv3DBackpropInputD update pads list by padding failed.");
    return GRAPH_FAILED;
  }
  OP_LOGI(op_name.GetString(), "Leaving Conv3DBackpropInputD inferfunction!");
  return GRAPH_SUCCESS;
}

IMPLEMT_INFER_DATA_SLICE(Conv3DBackpropInputD, Conv3DBackpropInputDInfereDataSlice) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter Conv3DBackpropInputDInfereDataSlice.");

  auto x_format = op.get_input_desc_out_backprop().GetFormat();
  int32_t strd = 0;
  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dild = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;

  CHECK_OP_FUNC(!GetAttrsConv3D(op, x_format, strd, strh, strw, dild, dilh, dilw),
                return GRAPH_FAILED, "get attrs failed.");

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_dedy = op_desc->MutableInputDesc("out_backprop");
  GeTensorDescPtr tensor_desc_w = op_desc->MutableInputDesc("filter");
  CHECK_PTR_NULL(tensor_desc_y, "tensor y desc", return GRAPH_FAILED);
  CHECK_PTR_NULL(tensor_desc_dedy, "tensor dedy desc", return GRAPH_FAILED);
  CHECK_PTR_NULL(tensor_desc_w, "tensor filter desc", return GRAPH_FAILED);

  vector<vector<int64_t>> y_data_slice;
  vector<vector<int64_t>> dedy_data_slice(kConv3dDataSlice, vector<int64_t>(0));
  vector<vector<int64_t>> w_data_slice(kFilterDataSlice, vector<int64_t>(0));

  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGW(op_name.GetString(), "no data slice, not need infer input");
    return GRAPH_FAILED;
  }

  graphStatus ret = VerifyDataSlice(op, y_data_slice);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }

  auto x_shape = op.get_input_desc_out_backprop().GetShape().GetDims();
  CHECK_KEY_IN_MAP(format2str, x_format, "x_format", return GRAPH_FAILED);
  std::string x_format_str = format2str[x_format];
  int32_t d_input_position = x_format_str.find("D");
  CHECK_OP_FUNC(d_input_position < 0, return GRAPH_FAILED, "get position failed: %d", d_input_position);
  int32_t h_input_position = x_format_str.find("H");
  CHECK_OP_FUNC(h_input_position < 0, return GRAPH_FAILED, "get position failed: %d", h_input_position);
  int32_t w_input_position = x_format_str.find("W");
  CHECK_OP_FUNC(w_input_position < 0, return GRAPH_FAILED, "get position failed: %d", w_input_position);
  int32_t id = x_shape[d_input_position];
  int32_t ih = x_shape[h_input_position];
  int32_t iw = x_shape[w_input_position];

  auto filter_format = op.get_input_desc_filter().GetFormat();
  auto w_shape = op.get_input_desc_filter().GetShape().GetDims();
  CHECK_KEY_IN_MAP(format2str, filter_format, "filter_format", return GRAPH_FAILED);
  std::string filter_format_str = format2str[filter_format];
  int32_t d_filter_position = filter_format_str.find("D");
  CHECK_OP_FUNC(d_filter_position < 0, return GRAPH_FAILED, "get position failed: %d", d_filter_position);
  int32_t h_filter_position = filter_format_str.find("H");
  CHECK_OP_FUNC(h_filter_position < 0, return GRAPH_FAILED, "get position failed: %d", h_filter_position);
  int32_t w_filter_position = filter_format_str.find("W");
  CHECK_OP_FUNC(w_filter_position < 0, return GRAPH_FAILED, "get position failed: %d", w_filter_position);
  int32_t kd = w_shape[d_filter_position];
  int32_t kh = w_shape[h_filter_position];
  int32_t kw = w_shape[w_filter_position];

  bool needUpdateX = false;
  bool needUpdateW = false;
  vector<int32_t> pad_list;
  op.GetAttr("pads", pad_list);
  CHECK_OP_FUNC(pad_list.size() != kConv3dPadsSizeLimit, return GRAPH_FAILED, "the pad_list is illegal.");
  SliceTuple infer_dhw = std::make_tuple(kd, dild, strd, id);
  // cut N
  if (y_data_slice[0].size() != 0) {
    dedy_data_slice[0] = y_data_slice[0];
    needUpdateX = true;
  }

  // cut D
  if (y_data_slice[kDDimNDC1HWC0Idx].size() != 0) {
    dedy_data_slice[kDDimNDC1HWC0Idx].clear();
    dedy_data_slice[kDDimNDC1HWC0Idx].resize(kDataSliceLen);
    InferHWConv3dBackpropInput(infer_dhw,
                               y_data_slice[kDDimNDC1HWC0Idx], dedy_data_slice[kDDimNDC1HWC0Idx],
                               pad_list, kConv3dPadHeadIdx);
    needUpdateX = true;
  }

  // cut H
  if (y_data_slice[kHDimNDC1HWC0Idx].size() != 0) {
    dedy_data_slice[kHDimNDC1HWC0Idx].clear();
    dedy_data_slice[kHDimNDC1HWC0Idx].resize(kDataSliceLen);
    infer_dhw = std::make_tuple(kh, dilh, strh, ih);
    InferHWConv3dBackpropInput(infer_dhw,
                               y_data_slice[kHDimNDC1HWC0Idx], dedy_data_slice[kHDimNDC1HWC0Idx],
                               pad_list, kConv3dPadUpIdx);
    needUpdateX = true;
  }

  // cut W
  if (y_data_slice[kWDimNDC1HWC0Idx].size() != 0) {
    dedy_data_slice[kWDimNDC1HWC0Idx].clear();
    dedy_data_slice[kWDimNDC1HWC0Idx].resize(kDataSliceLen);
    infer_dhw = std::make_tuple(kw, dilw, strw, iw);
    InferHWConv3dBackpropInput(infer_dhw,
                               y_data_slice[kWDimNDC1HWC0Idx], dedy_data_slice[kWDimNDC1HWC0Idx],
                               pad_list, kConv3dPadLeftIdx);
    needUpdateX = true;
  }

  // cut Cout
  if (y_data_slice[kC1DimNDC1HWC0Idx].size() != 0) {
    w_data_slice[0].clear();
    w_data_slice[0].resize(kDataSliceLen);
    w_data_slice[0][0] = y_data_slice[kC1DimNDC1HWC0Idx][0] * static_cast<int64_t>(kh * kw);
    w_data_slice[0][1] = y_data_slice[kC1DimNDC1HWC0Idx][1] * static_cast<int64_t>(kh * kw);
    needUpdateW = true;
  }

  // check update flag
  CHECK_OP_FUNC(!needUpdateX && !needUpdateW, return GRAPH_FAILED, "there's no update in desc.");

  // update data slice attr
  if (needUpdateX) {
    CHECK_OP_FUNC(!AttrUtils::SetListListInt(tensor_desc_dedy, ge::ATTR_NAME_DATA_SLICE, dedy_data_slice),
                  return GRAPH_FAILED,
                  "set outbackprop data slice attr failed.");
  }
  if (needUpdateW) {
    CHECK_OP_FUNC(!AttrUtils::SetListListInt(tensor_desc_w, ge::ATTR_NAME_DATA_SLICE, w_data_slice),
                  return GRAPH_FAILED,
                  "set w data slice attr failed");
  }

  // update pads attr info
  op.SetAttr("pads", pad_list);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv3DBackpropInputD, Conv3DBackpropInputDVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGI(op_name.GetString(), "Enter Conv3DBackpropInputD verifyfunction!");
  if (GRAPH_SUCCESS != VerifyConv3dbpInputCommon(op)) {
    return GRAPH_FAILED;
  }
  // check padding value
  if (GRAPH_SUCCESS != VerifyConv3dbpPads(op)) {
    return GRAPH_FAILED;
  }
  OP_LOGI(op_name.GetString(), "Leaving Conv3DBackpropInputD verifyfunction!");
  return GRAPH_SUCCESS;
}

INFER_DATA_SLICE_FUNC_REG(Conv3DBackpropInputD, Conv3DBackpropInputDInfereDataSlice);
INFER_FUNC_REG(Conv3DBackpropInputD, Conv3DBackpropInputDInfer);
VERIFY_FUNC_REG(Conv3DBackpropInputD, Conv3DBackpropInputDVerify);

// ----------------Conv3DBackpropFilter-------------------
static graphStatus VerifyConv3dbpFilterCommon(const ge::Operator& op) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  auto x_desc = op.GetInputDescByName("x");
  auto out_backprop_desc = op.GetInputDescByName("out_backprop");

  // check input dtype
  auto x_dtype = x_desc.GetDataType();
  auto out_backprop_dtype = out_backprop_desc.GetDataType();
  if (x_dtype != out_backprop_dtype) {
    OP_LOGE(op_name.GetString(), "x's dtype should equal to out_backprop's dtype.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpFilter";
    err_map["attr_name"] = "dtype";
    err_map["param1_name"] = "input x";
    err_map["param2_name"] = "outBackprop";
    err_map["param1_value"] = std::to_string(x_dtype);
    err_map["param2_value"] = std::to_string(out_backprop_dtype);
    std::string report_error_code = "E50031";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check input tensor shape
  auto x_shape = x_desc.GetShape().GetDims();
  if (!IsUnknownRankShape(x_shape) && x_shape.size() != kConv3dDimSizeLimit) {
    OP_LOGE(op_name.GetString(), "x's shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "x_shape";
    err_map["op_name"] = "Conv3dbpFilter";
    err_map["excepted_value"] = std::to_string(kConv3dDimSizeLimit);
    err_map["input_value"] = std::to_string(x_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  auto out_backprop_shape = out_backprop_desc.GetShape().GetDims();
  if (!IsUnknownRankShape(out_backprop_shape) && out_backprop_shape.size() != kConv3dDimSizeLimit) {
    OP_LOGE(op_name.GetString(), "out_backprop's shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "out_backprop_shape_size";
    err_map["op_name"] = "Conv3dbpFilter";
    err_map["excepted_value"] = std::to_string(kConv3dDimSizeLimit);
    err_map["input_value"] = std::to_string(out_backprop_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check strides shape
  std::vector<int32_t> stride_list;
  if (GRAPH_SUCCESS == op.GetAttr("strides", stride_list)) {
    if (stride_list.size() != kConv3dStridesSizeLimit) {
      OP_LOGE(op_name.GetString(), "strides should be 5d.");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "stride_list";
      err_map["op_name"] = "Conv3dbpFilter";
      err_map["excepted_value"] = std::to_string(kConv3dStridesSizeLimit);
      err_map["input_value"] = std::to_string(stride_list.size());
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op_name.GetString(), "get strides list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpFilter";
    err_map["param_name"] = "strides";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check dilations shape
  std::vector<int32_t> dilations_list;
  CHECK_OP_FUNC(!VerifyConv3dDilations(op, dilations_list), return GRAPH_FAILED, "get dilation attrs failed.");
  return GRAPH_SUCCESS;
}
IMPLEMT_INFERFUNC(Conv3DBackpropFilter, Conv3DBackpropFilterInfer) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGI(op_name.GetString(), "Enter Conv3DBackpropFilter Infer Function!");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);
  std::vector<std::string> input_infer_depends = {"filter_size"};
  op_desc->SetOpInferDepends(input_infer_depends);

  auto y_desc = op_desc->MutableOutputDesc("y");
  Format filter_format = y_desc->GetFormat();
  CHECK_OP_FUNC(FORMAT_RESERVED == filter_format, return GRAPH_FAILED, "get format failed: %d", filter_format);

  auto x_desc = op_desc->MutableInputDesc("x");
  Format x_format = x_desc->GetFormat();
  CHECK_OP_FUNC(FORMAT_RESERVED == x_format, return GRAPH_FAILED, "get format failed: %d", x_format);
  int32_t x_c = -1;
  std::vector<int64_t> x_sizes = x_desc->MutableShape().GetDims();
  bool x_unknown_rank = IsUnknownRankShape(x_sizes);
  if (!x_unknown_rank) {
    CHECK_KEY_IN_MAP(format2str, x_format, "x_format", return GRAPH_FAILED);
    std::string x_format_str = format2str[x_format];
    int32_t c_position = x_format_str.find('C');
    x_c = x_sizes[c_position];
  }

  std::vector<int64_t> filter_sizes;
  auto filter_size_desc = op_desc->MutableInputDesc("filter_size");
  CHECK_KEY_IN_MAP(format2str, filter_format, "filter_format", return GRAPH_FAILED);
  std::string filter_format_str = format2str[filter_format];
  int32_t filter_co_position = filter_format_str.find('N');
  int32_t filter_ci_position = filter_format_str.find('C');
  auto out_backprop_desc = op_desc->MutableInputDesc("out_backprop");
  bool is_filter_size_const = true;
  Tensor filter_sizes_tensor;
  if (GRAPH_SUCCESS != op.GetInputConstData("filter_size", filter_sizes_tensor)) {
    filter_sizes.assign(kConv3dInputSizeLimit, -1);
    is_filter_size_const = false;
    std::vector<int64_t> out_backprop_sizes = out_backprop_desc->MutableShape().GetDims();
    if (!IsUnknownRankShape(out_backprop_sizes)) {
      Format out_backprop_format = out_backprop_desc->GetFormat();
      CHECK_OP_FUNC(FORMAT_RESERVED == out_backprop_format, return GRAPH_FAILED,
                    "get format failed: %d", out_backprop_format);
      CHECK_KEY_IN_MAP(format2str, out_backprop_format, "out_backprop_format", return GRAPH_FAILED);
      std::string format_str = format2str[out_backprop_format];
      int32_t out_c_position = format_str.find('C');
      filter_sizes[filter_co_position] = out_backprop_sizes[out_c_position];
    }
    int32_t groups_ori = 1;
    op.GetAttr("groups", groups_ori);
    filter_sizes[filter_ci_position] = (x_c >= 1 ? x_c / groups_ori : x_c);
  } else {
    // get shape for output from filter_size
    DataType dtype = filter_size_desc->GetDataType();
    GetConstValue(filter_sizes_tensor, dtype, filter_sizes);
  }
  if (filter_sizes.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op_name.GetString(), "filter_sizes's shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "filter_sizes";
    err_map["op_name"] = "Conv3dbpFilter";
    err_map["excepted_value"] = std::to_string(kConv3dInputSizeLimit);
    err_map["input_value"] = std::to_string(filter_sizes.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // set shape of output desc, filter_size should match the format of y
  std::vector<int64_t> y_shape;
  y_shape.push_back(filter_sizes[0]);
  y_shape.push_back(filter_sizes[1]);
  y_shape.push_back(filter_sizes[kHDimNDHWCIdx]);
  y_shape.push_back(filter_sizes[kWDimNDHWCIdx]);
  y_shape.push_back(filter_sizes[kCDimNDHWCIdx]);
  y_desc->SetShape(ge::GeShape(y_shape));

  bool is_dynamic = (!is_filter_size_const || IsUnKnownShape(x_sizes) || x_unknown_rank);
  bool unset_group = (x_unknown_rank || filter_sizes[filter_ci_position] < 1 || x_c < 1);
  if (unset_group) {
    OP_LOGD(op_name.GetString(), "ignore set groups.");
  } else if (!SetGroupsConv(op, x_sizes, x_format, filter_sizes, filter_format)) {
    OP_LOGD(op_name.GetString(), "Set groups for Conv3DBackpropFilter failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = string(op_name.GetString());
    err_map["description"] = "Set groups for Conv3DBackpropFilter failed.";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (!is_dynamic) {
    // update pads list by padding[SAME,VALID]
    CHECK_OP_FUNC(!SetPadListByPaddingConv3dbp(op, x_sizes, x_format, filter_sizes, filter_format),
                  return GRAPH_FAILED,
                  "update pads list by padding failed.");
  } else if (IsUnKnownShape(x_sizes) || x_unknown_rank) {
    std::string pad_str;
    if (GRAPH_SUCCESS == op.GetAttr("padding", pad_str)) {
      if (pad_str == "SAME") {
        op.SetAttr("pads", {-1, -1, -1, -1, -1, -1});
        OP_LOGD(op_name.GetString(), "set pads to {-1, -1, -1, -1, -1, -1} when padding is SAME in dynamic_shape");
      } else if (pad_str == "VALID") {
        op.SetAttr("pads", {0, 0, 0, 0, 0, 0});
        OP_LOGD(op_name.GetString(), "set pads to {0, 0, 0, 0, 0, 0} when padding is VALID in dynamic_shape");
      }
    }
  }
  OP_LOGI(op_name.GetString(), "Leaving Conv3DBackpropFilter infer function!");
  return GRAPH_SUCCESS;
}

static graphStatus VerifyConv3dbpFilterPads(const ge::Operator& op) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  std::vector<int> pads;
  if (GRAPH_SUCCESS == op.GetAttr("pads", pads)) {
    if (pads.size() < kConv3dPadsSizeLimit) {
      OP_LOGE(op_name.GetString(), "op pads's size is illegal.");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "pads";
      err_map["op_name"] = "Conv3dbpFilter";
      err_map["excepted_value"] = std::to_string(kConv3dPadsSizeLimit);
      err_map["input_value"] = std::to_string(pads.size());
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }

    if (!CheckVectorAnyNegative(pads)) {
      OP_LOGE(op_name.GetString(), "op get pads is illegal");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "pads";
      err_map["op_name"] = "Conv3dbpFilter";
      err_map["excepted_value"] = "positive";
      err_map["input_value"] = ops::to_string(pads);
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op_name.GetString(), "op get pads failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpFilter";
    err_map["param_name"] = "pads";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv3DBackpropFilter, Conv3DBackpropFilterVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);
  OpDescPtr op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);

  OP_LOGI(op_name.GetString(), "Enter Conv3DBackpropFilter verify function!");
  if (GRAPH_SUCCESS != VerifyConv3dbpFilterCommon(op)) {
    return GRAPH_FAILED;
  }
  // check padding value
  if (VerifyConvPadding(op_name, op_desc) || GRAPH_SUCCESS == VerifyConv3dbpPads(op)) {
    OP_LOGI(op_name.GetString(), "Leaving Conv3DBackpropFilter verify function!");
    return GRAPH_SUCCESS;
  } else {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "Leaving Conv3DBackpropFilter verify function!");
    return GRAPH_FAILED;
  }
}

INFER_FUNC_REG(Conv3DBackpropFilter, Conv3DBackpropFilterInfer);
VERIFY_FUNC_REG(Conv3DBackpropFilter, Conv3DBackpropFilterVerify);

// ----------------Conv3DBackpropFilterD-------------------
IMPLEMT_INFERFUNC(Conv3DBackpropFilterD, Conv3DBackpropFilterDInfer) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGI(op_name.GetString(), "Enter Conv3DBackpropFilterD inferfunction!");

  // get shape for output from filter_size
  std::vector<int32_t> filter_sizes;
  if (GRAPH_SUCCESS == op.GetAttr("filter_size", filter_sizes)) {
    if (filter_sizes.size() != kConv3dDimSizeLimit) {
      OP_LOGE(op_name.GetString(), "filter_size list should be 5d.");
      map<std::string, std::string> err_map;
      err_map["param_name"] = "filter_sizes";
      err_map["op_name"] = "Conv3dbpFilter";
      err_map["excepted_value"] = std::to_string(kConv3dDimSizeLimit);
      err_map["input_value"] = std::to_string(filter_sizes.size());
      std::string report_error_code = "E50029";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
    }
  } else {
    OP_LOGE(op_name.GetString(), "get filter_size list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpFilter";
    err_map["param_name"] = "filter_size";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // set dtype of output desc
  auto y_desc = op.GetOutputDescByName("y");
  y_desc.SetDataType(DT_FLOAT);

  // set shape of output desc, filter_size should match the format of y
  std::vector<int64_t> y_shape;
  y_shape.push_back(filter_sizes[0]);
  y_shape.push_back(filter_sizes[1]);
  y_shape.push_back(filter_sizes[kHDimNDHWCIdx]);
  y_shape.push_back(filter_sizes[kWDimNDHWCIdx]);
  y_shape.push_back(filter_sizes[kCDimNDHWCIdx]);
  y_desc.SetShape(ge::Shape(y_shape));

  // update output desc
  if (GRAPH_SUCCESS != op.UpdateOutputDesc("y", y_desc)) {
    OP_LOGE(op_name.GetString(), "update output desc failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dbpFilter";
    err_map["param_name"] = "output y";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> x_sizes = op.GetInputDescByName("x").GetShape().GetDims();
  Format x_format = op.GetInputDescByName("x").GetFormat();
  Format filter_format = y_desc.GetFormat();
  // update pads list by padding[SAME,VALID]
  CHECK_OP_FUNC(!SetPadListByPaddingConv3dbp(op, x_sizes, x_format, filter_sizes, filter_format), return GRAPH_FAILED,
                "update pads list by padding failed.");

  OP_LOGI(op_name.GetString(), "Leaving Conv3DBackpropFilterD inferfunction!");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv3DBackpropFilterD, Conv3DBackpropFilterDVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGI(op_name.GetString(), "Enter Conv3DBackpropFilterD verifyfunction!");
  if (GRAPH_SUCCESS != VerifyConv3dbpFilterCommon(op)) {
    return GRAPH_FAILED;
  }
  // check padding value
  if (GRAPH_SUCCESS != VerifyConv3dbpFilterPads(op)) {
    return GRAPH_FAILED;
  }
  OP_LOGI(op_name.GetString(), "Leaving Conv3DBackpropFilterD verifyfunction!");
  return GRAPH_SUCCESS;
}

IMPLEMT_INFER_DATA_SLICE(Conv3DBackpropFilterD, Conv3DBackpropFilterDInfereDataSlice) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter Conv3DBackpropFilterDInfereDataSlice.");

  auto y_tensor = op.GetOutputDescByName("y");
  auto filter_format = y_tensor.GetOriginFormat();
  CHECK_KEY_IN_MAP(format2str, filter_format, "filter_format", return GRAPH_FAILED);
  std::string filter_format_str = format2str[filter_format];
  int32_t n_filter_position = filter_format_str.find("N");
  // get shape for output from filter_size
  std::vector<int32_t> filter_sizes;
  if (op.GetAttr("filter_size", filter_sizes) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "get filter_size list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3DBackpropFilter";
    err_map["param_name"] = "filter_size";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_dedy = op_desc->MutableInputDesc("out_backprop");
  CHECK_PTR_NULL(tensor_desc_y, "tensor y desc", return GRAPH_FAILED);
  CHECK_PTR_NULL(tensor_desc_dedy, "tensor dedy desc", return GRAPH_FAILED);

  vector<vector<int64_t>> y_data_slice;
  vector<vector<int64_t>> dedy_data_slice(6, vector<int64_t>(0));

  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op_name.GetString(), "no data slice, not need infer input");
    return GRAPH_FAILED;
  }

  for (size_t i = 0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() > 1) {
      int32_t y_extend = y_data_slice[i][1] - y_data_slice[i][0] + 1;
      if (i == 1) {
        dedy_data_slice[i + 1] = y_data_slice[i];
        CHECK_OP_FUNC(!AttrUtils::SetListListInt(tensor_desc_dedy, ge::ATTR_NAME_DATA_SLICE, dedy_data_slice),
                      return GRAPH_FAILED, "set dedy data slice failed.");
        filter_sizes[n_filter_position] = y_extend * kBlockBaseSize;
        op.SetAttr("filter_size", filter_sizes);
        OP_LOGI(op_name.GetString(), "infer input in Cout success");
        return GRAPH_SUCCESS;
      }
      OP_LOGI(op_name.GetString(), "can not supported split in Cin, H and W");
      return GRAPH_FAILED;
    }
  }
  OP_LOGI(op_name.GetString(), "no data slice, not need infer input");
  return GRAPH_FAILED;
}

INFER_DATA_SLICE_FUNC_REG(Conv3DBackpropFilterD, Conv3DBackpropFilterDInfereDataSlice);
INFER_FUNC_REG(Conv3DBackpropFilterD, Conv3DBackpropFilterDInfer);
VERIFY_FUNC_REG(Conv3DBackpropFilterD, Conv3DBackpropFilterDVerify);

template <typename T1, typename T2, typename T3>
static bool SetInputsizeListConv3dtranspose(ge::Operator& op, const std::vector<T1>& x_sizes, Format x_format,
                                            const std::vector<T2>& filter_sizes, Format filter_format,
                                            const std::vector<T3>& input_sizes, Format input_format) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  CHECK_OP_FUNC(FORMAT_RESERVED == x_format, return false, "get format failed: %d", x_format);
  CHECK_OP_FUNC(FORMAT_RESERVED == filter_format, return false, "get format failed: %d", filter_format);
  CHECK_OP_FUNC(FORMAT_RESERVED == input_format, return false, "get format failed: %d", input_format);
  if (x_sizes.size() != kConv3dInputSizeLimit) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "x_sizes is illegal");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "x_sizes";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(kConv3dInputSizeLimit);
    err_map["input_value"] = std::to_string(x_sizes.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (input_sizes.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op_name.GetString(), "input_sizes is illegal");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "input_sizes";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(kConv3dInputSizeLimit);
    err_map["input_value"] = std::to_string(input_sizes.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (filter_sizes.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op_name.GetString(), "filter_sizes is illegal");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "filter_size";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(kConv3dInputSizeLimit);
    err_map["input_value"] = std::to_string(filter_sizes.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::vector<int32_t> stride_list;
  if (op.GetAttr("strides", stride_list) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "op get strides failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dTranspose";
    err_map["param_name"] = "strides";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (stride_list.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op_name.GetString(), "op get strides failed.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(kConv3dInputSizeLimit);
    err_map["input_value"] = std::to_string(stride_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::vector<int32_t> dilations_list;
    if (!VerifyConv3dDilations(op, dilations_list)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "get dilation attrs failed.");
    return false;
  }

  std::vector<int32_t> output_padding_list;
  if (op.GetAttr("output_padding", output_padding_list) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "op get outputpadding failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dTranspose";
    err_map["param_name"] = "output_padding";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (output_padding_list.size() != kConv3dInputSizeLimit) {
    OP_LOGE(op_name.GetString(), "op get outputpadding failed.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "output_padding";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(kConv3dInputSizeLimit);
    err_map["input_value"] = std::to_string(output_padding_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  CHECK_KEY_IN_MAP(format2str, x_format, "x_format", return false);
  std::string x_format_str = format2str[x_format];
  int32_t h_input_position = x_format_str.find("H");
  CHECK_OP_FUNC(h_input_position < 0, return false, "get position failed: %d", h_input_position);
  int32_t w_input_position = x_format_str.find("W");
  CHECK_OP_FUNC(w_input_position < 0, return false, "get position failed: %d", w_input_position);
  int32_t d_input_position = x_format_str.find("D");
  CHECK_OP_FUNC(d_input_position < 0, return false, "get position failed: %d", d_input_position);
  int32_t c_input_position = x_format_str.find("C");
  CHECK_OP_FUNC(c_input_position < 0, return false, "get position failed: %d", c_input_position);
  int32_t n_input_position = x_format_str.find("N");
  CHECK_OP_FUNC(n_input_position < 0, return false, "get position failed: %d", n_input_position);
  int32_t dy_h = x_sizes[h_input_position];
  int32_t dy_w = x_sizes[w_input_position];
  int32_t dy_d = x_sizes[d_input_position];
  int32_t dy_n = x_sizes[n_input_position];

  int32_t stride_h = stride_list[h_input_position];
  int32_t stride_w = stride_list[w_input_position];
  int32_t stride_d = stride_list[d_input_position];

  int32_t dilation_h = dilations_list[h_input_position];
  int32_t dilation_w = dilations_list[w_input_position];
  int32_t dilation_d = dilations_list[d_input_position];

  int32_t outputpadding_h = output_padding_list[h_input_position];
  int32_t outputpadding_w = output_padding_list[w_input_position];
  int32_t outputpadding_d = output_padding_list[d_input_position];
  CHECK_KEY_IN_MAP(format2str, filter_format, "filter_format", return false);
  std::string filter_format_str = format2str[filter_format];
  int32_t h_filter_position = filter_format_str.find("H");
  CHECK_OP_FUNC(h_filter_position < 0, return false, "get position failed: %d", h_filter_position);
  int32_t w_filter_position = filter_format_str.find("W");
  CHECK_OP_FUNC(w_filter_position < 0, return false, "get position failed: %d", w_filter_position);
  int32_t d_filter_position = filter_format_str.find("D");
  CHECK_OP_FUNC(d_filter_position < 0, return false, "get position failed: %d", d_filter_position);
  int32_t c_filter_position = filter_format_str.find("C");
  CHECK_OP_FUNC(c_filter_position < 0, return false, "get position failed: %d", c_filter_position);

  int32_t filter_h = filter_sizes[h_filter_position];
  int32_t filter_w = filter_sizes[w_filter_position];
  int32_t filter_d = filter_sizes[d_filter_position];
  int32_t filter_c = filter_sizes[c_filter_position];

  std::vector<int32_t> pads_list;
  if (op.GetAttr("pads", pads_list) == GRAPH_FAILED) {
    OP_LOGE(op_name.GetString(), "op get pads failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dTranspose";
    err_map["param_name"] = "pads";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (pads_list.size() != kConv3dPadsSizeLimit) {
    OP_LOGE(op_name.GetString(), "op get pads failed.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "pads";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(kConv3dPadsSizeLimit);
    err_map["input_value"] = std::to_string(pads_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  int32_t pad_head = pads_list[kConv3dPadHeadIdx];
  int32_t pad_tail = pads_list[kConv3dPadTailIdx];
  int32_t pad_up = pads_list[kConv3dPadUpIdx];
  int32_t pad_down = pads_list[kConv3dPadDownIdx];
  int32_t pad_left = pads_list[kConv3dPadLeftIdx];
  int32_t pad_right = pads_list[kConv3dPadRightIdx];

  std::vector<int64_t> output;
  int32_t output_h = 0;
  int32_t output_w = 0;
  int32_t output_d = 0;
  int32_t output_n = 0;
  int32_t output_c = 0;
  if (!CheckVectorAllZero(input_sizes)) {
    output_h = input_sizes[h_input_position];
    output_w = input_sizes[w_input_position];
    output_d = input_sizes[d_input_position];
    output_n = input_sizes[n_input_position];
    output_c = input_sizes[c_input_position];
  } else {
    output_d = stride_d * (dy_d - 1) + outputpadding_d + ((filter_d - 1) * dilation_d + 1) - pad_head - pad_tail;
    output_h = stride_h * (dy_h - 1) + outputpadding_h + ((filter_h - 1) * dilation_h + 1) - pad_up - pad_down;
    output_w = stride_w * (dy_w - 1) + outputpadding_w + ((filter_w - 1) * dilation_w + 1) - pad_left - pad_right;
    output_n = dy_n;
    output_c = filter_c;
  }

  if (x_format == FORMAT_NCDHW) {
    output.push_back(output_n);
    output.push_back(output_c);
    output.push_back(output_d);
    output.push_back(output_h);
    output.push_back(output_w);
  } else if (x_format == FORMAT_NDHWC) {
    output.push_back(output_n);
    output.push_back(output_d);
    output.push_back(output_h);
    output.push_back(output_w);
    output.push_back(output_c);
  } else {
    OP_LOGE(op_name.GetString(), "input_size format should be NCDHW or NDHWC.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "input_size";
    err_map["op_name"] = "Conv3d";
    err_map["excepted_value"] = "NCDHW or NDHWC";
    err_map["input_value"] = x_format;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // set input_size shape to dedx
  op.SetAttr("dedx", output);

  return true;
}

static bool GetAttrsConv3DTranspose(ge::Operator& op, Format refer, int32_t& strd,
                                    int32_t& strh, int32_t& strw, int32_t& dild,
                                    int32_t& dilh, int32_t& dilw) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  std::vector<int32_t> stride_list;
  CHECK_OP_FUNC(GRAPH_SUCCESS != op.GetAttr("strides", stride_list), return false, "get strides list failed.");
  auto s_size = stride_list.size();
  if (s_size != kConv3dStridesSizeLimit) {
    OP_LOGE(op_name.GetString(), "strides list should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "stride_list";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = "5d";
    err_map["input_value"] = std::to_string(s_size);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::string data_format;
  if (op.GetAttr("data_format", data_format) != GRAPH_SUCCESS) {
    OP_LOGI(op_name.GetString(), "no data format setting, using NDHWC");
    data_format = FORMAT_NDHWC;
  }

  std::vector<int32_t> dilation_list;
  CHECK_OP_FUNC(!VerifyConv3dDilations(op, dilation_list), return false, "get dilation attrs failed.");

  if (refer == FORMAT_NCDHW) {
    strd = stride_list[kDDimNCDHWIdx];
    strh = stride_list[kHDimNCDHWIdx];
    strw = stride_list[kWDimNCDHWIdx];
    dild = dilation_list[kDDimNCDHWIdx];
    dilh = dilation_list[kHDimNCDHWIdx];
    dilw = dilation_list[kWDimNCDHWIdx];
  } else if (refer == FORMAT_NDHWC) {
    strd = stride_list[kDDimNDHWCIdx];
    strh = stride_list[kHDimNDHWCIdx];
    strw = stride_list[kWDimNDHWCIdx];
    dild = dilation_list[kDDimNDHWCIdx];
    dilh = dilation_list[kHDimNDHWCIdx];
    dilw = dilation_list[kWDimNDHWCIdx];
  }
  if (strd <= 0 || strh <= 0 || strw <= 0) {
    OP_LOGE(op_name.GetString(), "strides should be positive.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = "positive";
    err_map["input_value"] = std::to_string(strd) + " " + std::to_string(strh) + " " + std::to_string(strw);
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  return true;
}

static graphStatus VerifyConv3dTransposeInput(const ge::Operator& op) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  auto filter_desc = op.GetInputDescByName("filter");
  auto filter_dtype = filter_desc.GetDataType();
  auto x_desc = op.GetInputDescByName("x");
  auto x_dtype = x_desc.GetDataType();
  // check input dtype
  if (filter_dtype != x_dtype) {
    OP_LOGE(op_name.GetString(), "filter's dtype should equal to x's dtype.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dTranspose";
    err_map["attr_name"] = "dtype";
    err_map["param1_name"] = "filter";
    err_map["param2_name"] = "x";
    err_map["param1_value"] = std::to_string(filter_dtype);
    err_map["param2_value"] = std::to_string(x_dtype);
    std::string report_error_code = "E50031";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check input tensor shape
  auto filter_shape = filter_desc.GetShape().GetDims();
  if (filter_shape.size() != kConv3dDimSizeLimit) {
    OP_LOGE(op_name.GetString(), "filter's shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "filterShape_size";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(kConv3dDimSizeLimit);
    err_map["input_value"] = std::to_string(filter_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  auto x_shape = x_desc.GetShape().GetDims();
  if ((!IsUnknownRankShape(x_shape)) &&  x_shape.size() != kConv3dDimSizeLimit) {
    OP_LOGE(op_name.GetString(), "x's shape should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "xShape_size";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(kConv3dDimSizeLimit);
    err_map["input_value"] = std::to_string(x_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check strides shape
  std::vector<int32_t> stride_list;
  if (op.GetAttr("strides", stride_list) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "get strides list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dTranspose";
    err_map["param_name"] = "strides";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (stride_list.size() != kConv3dDimSizeLimit) {
    OP_LOGE(op_name.GetString(), "strides should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(kConv3dDimSizeLimit);
    err_map["input_value"] = std::to_string(stride_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  // check dilations shape
  std::vector<int32_t> dilations_list;
  if (!VerifyConv3dDilations(op, dilations_list)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "get dilation attrs failed.");
    return GRAPH_FAILED;
  }

  std::vector<int32_t> output_padding_list;
  if (op.GetAttr("output_padding", output_padding_list) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "get output_padding list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dTranspose";
    err_map["param_name"] = "output_padding";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (output_padding_list.size() != kConv3dDimSizeLimit) {
    OP_LOGE(op_name.GetString(), "output_paddingList list should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "output_padding";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(kConv3dDimSizeLimit);
    err_map["input_value"] = std::to_string(output_padding_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

graphStatus InferConv3dTransposeDataSlice(ge::Operator& op) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  auto x_format = op.GetInputDescByName("x").GetOriginFormat();
  int32_t strd = 0;
  int32_t strh = 0;
  int32_t strw = 0;
  int32_t dild = 0;
  int32_t dilh = 0;
  int32_t dilw = 0;

  if (!GetAttrsConv3DTranspose(op, x_format, strd, strh, strw, dild, dilh, dilw)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "get attrs failed.");
    return GRAPH_FAILED;
  }

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
  GeTensorDescPtr tensor_desc_w = op_desc->MutableInputDesc("filter");

  CHECK_PTR_NULL(tensor_desc_y, "tensor x desc", return GRAPH_FAILED);
  CHECK_PTR_NULL(tensor_desc_x, "tensor y desc", return GRAPH_FAILED);
  CHECK_PTR_NULL(tensor_desc_w, "tensor w desc", return GRAPH_FAILED);

  vector<vector<int64_t>> y_data_slice(kConv3dDataSlice, vector<int64_t>(0));
  vector<vector<int64_t>> x_data_slice(kConv3dDataSlice, vector<int64_t>(0));
  vector<vector<int64_t>> w_data_slice(kFilterDataSlice, vector<int64_t>(0));
  vector<vector<int64_t>> bias_data_slice(1, vector<int64_t>(0));

  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op_name.GetString(), "no data slice, not need infer input");
    return GRAPH_FAILED;
  }

  graphStatus ret = VerifyDataSlice(op, y_data_slice);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }

  auto x_shape = op.GetInputDescByName("x").GetOriginShape().GetDims();
  CHECK_KEY_IN_MAP(format2str, x_format, "x_format", return GRAPH_FAILED);
  std::string x_format_str = format2str[x_format];
  int32_t d_input_position = x_format_str.find("D");
  CHECK_OP_FUNC(d_input_position < 0, return GRAPH_FAILED, "get position failed: %d", d_input_position);
  int32_t h_input_position = x_format_str.find("H");
  CHECK_OP_FUNC(h_input_position < 0, return GRAPH_FAILED, "get position failed: %d", h_input_position);
  int32_t w_input_position = x_format_str.find("W");
  CHECK_OP_FUNC(w_input_position < 0, return GRAPH_FAILED, "get position failed: %d", w_input_position);
  int32_t n_input_position = x_format_str.find("N");
  CHECK_OP_FUNC(n_input_position < 0, return GRAPH_FAILED, "get position failed: %d", n_input_position);

  int32_t in = -1;
  int32_t id = -1;
  int32_t ih = -1;
  int32_t iw = -1;
  if (x_shape != DYNAMIC_DIM_ALL) {
    id = x_shape[d_input_position];
    ih = x_shape[h_input_position];
    iw = x_shape[w_input_position];
    in = x_shape[n_input_position];
  }

  auto filter_format = op.GetInputDescByName("filter").GetOriginFormat();
  auto w_shape = op.GetInputDescByName("filter").GetOriginShape().GetDims();
  CHECK_KEY_IN_MAP(format2str, filter_format, "filter_format", return GRAPH_FAILED);
  std::string filter_format_str = format2str[filter_format];
  int32_t d_filter_position = filter_format_str.find("D");
  CHECK_OP_FUNC(d_filter_position < 0, return GRAPH_FAILED, "get position failed: %d", d_filter_position);
  int32_t h_filter_position = filter_format_str.find("H");
  CHECK_OP_FUNC(h_filter_position < 0, return GRAPH_FAILED, "get position failed: %d", h_filter_position);
  int32_t w_filter_position = filter_format_str.find("W");
  CHECK_OP_FUNC(w_filter_position < 0, return GRAPH_FAILED, "get position failed: %d", w_filter_position);
  int32_t kd = w_shape[d_filter_position];
  int32_t kh = w_shape[h_filter_position];
  int32_t kw = w_shape[w_filter_position];

  bool needUpdateX = false;
  bool needUpdateW = false;
  SliceTuple infer_dhw = std::make_tuple(kd, dild, strd, id);
  // cut N
  if (y_data_slice[0].size() != 0) {
    x_data_slice[0] = y_data_slice[0];
    if (in < 0) {
      x_data_slice[0] = {-1, -1};
    }
    needUpdateX = true;
  }

  vector<int32_t> pad_list;
  op.GetAttr("pads", pad_list);
  CHECK_OP_FUNC(pad_list.size() != kConv3dPadsSizeLimit, return GRAPH_FAILED, "the pad_list is illegal.");
  // cut D
  if (y_data_slice[kDDimNDC1HWC0Idx].size() != 0) {
    x_data_slice[kDDimNDC1HWC0Idx].clear();
    x_data_slice[kDDimNDC1HWC0Idx].resize(kDataSliceLen);
    InferHWConv3dBackpropInput(infer_dhw,
                               y_data_slice[kDDimNDC1HWC0Idx], x_data_slice[kDDimNDC1HWC0Idx],
                               pad_list, kConv3dPadHeadIdx);
    needUpdateX = true;
  }

  // cut H
  if (y_data_slice[kHDimNDC1HWC0Idx].size() != 0) {
    x_data_slice[kHDimNDC1HWC0Idx].clear();
    x_data_slice[kHDimNDC1HWC0Idx].resize(kDataSliceLen);
    infer_dhw = std::make_tuple(kh, dilh, strh, ih);
    InferHWConv3dBackpropInput(infer_dhw,
                               y_data_slice[kHDimNDC1HWC0Idx], x_data_slice[kHDimNDC1HWC0Idx],
                               pad_list, kConv3dPadUpIdx);
    needUpdateX = true;
  }

  // cut W
  if (y_data_slice[kWDimNDC1HWC0Idx].size() != 0 && pad_list.size() == kConv3dPadsSizeLimit) {
    x_data_slice[kWDimNDC1HWC0Idx].clear();
    x_data_slice[kWDimNDC1HWC0Idx].resize(kDataSliceLen);
    infer_dhw = std::make_tuple(kw, dilw, strw, iw);
    InferHWConv3dBackpropInput(infer_dhw,
                               y_data_slice[kWDimNDC1HWC0Idx], x_data_slice[kWDimNDC1HWC0Idx],
                               pad_list, kConv3dPadLeftIdx);
    needUpdateX = true;
  }

  // cut Cout
  if (y_data_slice[kC1DimNDC1HWC0Idx].size() != 0) {
    w_data_slice[0].clear();
    w_data_slice[0].resize(kDataSliceLen);
    w_data_slice[0][0] = y_data_slice[kC1DimNDC1HWC0Idx][0] * static_cast<int64_t>(kh * kw);
    w_data_slice[0][1] = y_data_slice[kC1DimNDC1HWC0Idx][1] * static_cast<int64_t>(kh * kw);
    bias_data_slice[0] = y_data_slice[kC1DimNDC1HWC0Idx];
    needUpdateW = true;
  }

  // check update flag
  CHECK_OP_FUNC(!needUpdateX && !needUpdateW, return GRAPH_FAILED, "there's no update in desc.");

  // update data slice attr
  if (needUpdateX) {
    CHECK_OP_FUNC(!AttrUtils::SetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice),
                  return GRAPH_FAILED,
                  "set x data slice attr failed.");
  }
  if (needUpdateW) {
    CHECK_OP_FUNC(!AttrUtils::SetListListInt(tensor_desc_w, ge::ATTR_NAME_DATA_SLICE, w_data_slice),
                  return GRAPH_FAILED,
                  "set w data slice attr failed");
  }

  // update pads attr info
  op.SetAttr("pads", pad_list);
  return GRAPH_SUCCESS;
}

// ----------------Conv3DTranspose-------------------
IMPLEMT_INFERFUNC(Conv3DTranspose, Conv3DTransposeInfer) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGI(op_name.GetString(), "Enter Conv3DTranspose inferfunction!");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);
  std::vector<std::string> input_infer_depends = {"input_size"};
  op_desc->SetOpInferDepends(input_infer_depends);
  auto x_desc = op_desc->MutableInputDesc("x");
  auto y_desc = op_desc->MutableOutputDesc("y");
  CHECK_PTR_NULL(x_desc, "tensor x desc", return GRAPH_FAILED);
  CHECK_PTR_NULL(y_desc, "tensor y desc", return GRAPH_FAILED);

  std::vector<int64_t> x_sizes = x_desc->MutableShape().GetDims();
  for (size_t j = 0; j < x_sizes.size(); j++) {
    OP_LOGD(op_name.GetString(), "dy_sizes [%u] is %lld", j, x_sizes[j]);
  }

  bool is_input_size_const = false;
  bool unknown_rank = IsUnknownRankShape(x_sizes);
  std::vector<int64_t> input_sizes;
  Tensor input_sizes_tensor;
  auto input_sizes_desc = op_desc->MutableInputDesc("input_size");
  if (op.GetInputConstData("input_size", input_sizes_tensor) == GRAPH_SUCCESS) {
    DataType dtype = input_sizes_desc->GetDataType();
    GetConstValue(input_sizes_tensor, dtype, input_sizes);
    is_input_size_const = true;
  }

  // when static op or dynamic op phase running, is_dynamic == false
  bool unknown_shape = IsUnKnownShape(x_sizes) && (!is_input_size_const);
  if (unknown_shape || (!is_input_size_const && unknown_rank)) {
    if (!InferConv3dBpInputOutShapeRange(op, input_sizes_desc, x_desc, y_desc, input_sizes)) {
      return GRAPH_FAILED;
    }
  }

  // set dtype of x
  auto x_dtype = op.GetInputDescByName("x").GetDataType();
  y_desc->SetDataType(x_dtype);
  if (!unknown_rank) {
    std::vector<int64_t> filter_sizes = op.GetInputDescByName("filter").GetShape().GetDims();
    Format filter_format = op.GetInputDescByName("filter").GetFormat();
    Format input_format = y_desc->GetFormat();
    Format x_format = x_desc->GetFormat();
    if (!SetInputsizeListConv3dtranspose(op, x_sizes, x_format, filter_sizes, filter_format,
                                         input_sizes, input_format)) {
      CUBE_INNER_ERR_REPORT(op_name.GetString(),
        "Conv3DTranspose update pads list by padding failed or calculate input sizes failed.");
      return GRAPH_FAILED;
    }
  }

  // set shape of output desc, input_sizes should match the format of y
  std::vector<int64_t> dedx;
  bool get_dedx_invalid = (!unknown_rank) && (op.GetAttr("dedx", dedx) != GRAPH_SUCCESS);
  if (get_dedx_invalid) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "Get dedx shape which is used to set output shape failed");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dedx_shape = !is_input_size_const ? input_sizes : dedx;
  // set shape of output desc, input_size should match the format of y
  if (input_sizes.size() == kConv3dDimSizeLimit) {
    y_desc->SetShape(GeShape(dedx_shape));
  }

  // update pads list by padding[SAME,VALID]
  // if only batch is -1, no need to set SAME padding as -1
  // x_size maybe contains -1 in runtime compile, but can't set pads as -1
  if ((!is_input_size_const) && (unknown_rank ||
      IsDHWUnknown(string(op_name.GetString()), "y", y_desc->MutableShape().GetDims(), y_desc->GetFormat()))) {
    std::string pad_str;
    if (op.GetAttr("padding", pad_str) == GRAPH_SUCCESS) {
      std::vector<int32_t> pads(kConv3dPadsSizeLimit, 0);
      if (pad_str == "SAME") {
        pads.assign(kConv3dPadsSizeLimit, -1);
        OP_LOGD(op_name.GetString(), "set pads to {-1, -1, -1, -1, -1, -1} when padding is SAME in dynamic_shape");
      }

      op.SetAttr("pads", pads);
    }
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv3DTranspose, Conv3DTransposeVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  if (VerifyConv3dTransposeInput(op) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  // check padding value
  if (VerifyConv3dbpPads(op) == GRAPH_SUCCESS) {
    return GRAPH_SUCCESS;
  } else {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "Leaving Conv3DTranspose verifyfunction!");
    return GRAPH_FAILED;
  }
}

IMPLEMT_INFER_DATA_SLICE(Conv3DTranspose, Conv3DTransposeInfereDataSlice) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter Conv3DTransposeInfereDataSlice.");
  graphStatus ret = InferConv3dTransposeDataSlice(op);
  return ret;
}

INFER_DATA_SLICE_FUNC_REG(Conv3DTranspose, Conv3DTransposeInfereDataSlice);
INFER_FUNC_REG(Conv3DTranspose, Conv3DTransposeInfer);
VERIFY_FUNC_REG(Conv3DTranspose, Conv3DTransposeVerify);
// ----------------Conv3DTransposeD-------------------
IMPLEMT_INFERFUNC(Conv3DTransposeD, Conv3DTransposeDInfer) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  // get shape for output from input_size
  std::vector<int32_t> input_sizes;
  if (op.GetAttr("input_size", input_sizes) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "get input_size list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dTranspose";
    err_map["param_name"] = "input_size";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  if (input_sizes.size() != kConv3dDimSizeLimit) {
    OP_LOGE(op_name.GetString(), "input_size list should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "input_size";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(kConv3dDimSizeLimit);
    err_map["input_value"] = std::to_string(input_sizes.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  Format filter_format = op.GetInputDescByName("filter").GetFormat();
  auto y_desc = op.GetOutputDescByName("y");
  Format input_format = y_desc.GetFormat();
  Format x_format = op.GetInputDescByName("x").GetFormat();
  // update pads list by padding[SAME,VALID] and calculate input_size
  std::vector<int64_t> filter_sizes = op.GetInputDescByName("filter").GetShape().GetDims();
  std::vector<int64_t> x_sizes = op.GetInputDescByName("x").GetShape().GetDims();
  if (SetInputsizeListConv3dtranspose(op, x_sizes, x_format, filter_sizes, filter_format, input_sizes, input_format) ==
      false) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(),
      "Conv3DTransposeD update pads list by padding failed or calculate input sizes failed.");
    return GRAPH_FAILED;
  }
  // get dtype for output from x
  auto x_desc = op.GetInputDescByName("x");
  auto x_dtype = x_desc.GetDataType();
  // set dtype of output desc
  y_desc.SetDataType(x_dtype);
  // set shape of output desc, input_size should match the format of y
  std::vector<int32_t> dedx;
  if (op.GetAttr("dedx", dedx) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "get dedx list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dTranspose";
    err_map["param_name"] = "dedx";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (dedx.size() != kConv3dDimSizeLimit) {
    OP_LOGE(op_name.GetString(), "dedx list should be 5d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "dedx";
    err_map["op_name"] = "Conv3dTranspose";
    err_map["excepted_value"] = std::to_string(kConv3dDimSizeLimit);
    err_map["input_value"] = std::to_string(dedx.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> out_shape;
  for (auto i : dedx) {
    out_shape.push_back(i);
  }

  y_desc.SetShape(ge::Shape(out_shape));
  // update input_size shape
  op.SetAttr("input_size", dedx);

  // update output desc
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "update output desc failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv3dTranspose";
    err_map["param_name"] = "output y";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_INFER_DATA_SLICE(Conv3DTransposeD, Conv3DTransposeDInfereDataSlice) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter Conv3DTransposeDInfereDataSlice.");
  graphStatus ret = InferConv3dTransposeDataSlice(op);
  return ret;
}

IMPLEMT_VERIFIER(Conv3DTransposeD, Conv3DTransposeDVerify) {
  if (VerifyConv3dTransposeInput(op) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  // check padding value
  if (VerifyConv3dbpPads(op) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_DATA_SLICE_FUNC_REG(Conv3DTransposeD, Conv3DTransposeDInfereDataSlice);
INFER_FUNC_REG(Conv3DTransposeD, Conv3DTransposeDInfer);
VERIFY_FUNC_REG(Conv3DTransposeD, Conv3DTransposeDVerify);

// ----------------Conv2DTransposeD-------------------
template <typename T1, typename T2, typename T3>
static bool SetInputsizeListConv2DTranspose(ge::Operator& op, const std::vector<T1>& x_sizes, Format x_format,
                                            const std::vector<T2>& filter_sizes, Format filter_format,
                                            const std::vector<T3>& input_sizes, bool& isRun) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  if (filter_sizes.size() != kConv2dInputSizeLimit) {
    OP_LOGE(op_name.GetString(), "filter_sizes is illegal.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "filter_size";
    err_map["op_name"] = "Conv2DTranspose";
    err_map["excepted_value"] = std::to_string(kConv2dInputSizeLimit);
    err_map["input_value"] = std::to_string(filter_sizes.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::vector<int32_t> stride_list;
  if (op.GetAttr("strides", stride_list) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "op get strides failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTranspose";
    err_map["param_name"] = "strides";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (stride_list.size() != kConv2dInputSizeLimit) {
    OP_LOGE(op_name.GetString(), "stride_list size is illegal.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "strides";
    err_map["op_name"] = "Conv2DTranspose";
    err_map["excepted_value"] = std::to_string(kConv2dInputSizeLimit);
    err_map["input_value"] = std::to_string(stride_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::vector<int32_t> dilations_list;
  if (op.GetAttr("dilations", dilations_list) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "op get dilation failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTranspose";
    err_map["param_name"] = "dilations";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (dilations_list.size() != kConv2dInputSizeLimit) {
    OP_LOGE(op_name.GetString(), "dilations_list size is illegal.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "dilations";
    err_map["op_name"] = "Conv2DTranspose";
    err_map["excepted_value"] = std::to_string(kConv2dInputSizeLimit);
    err_map["input_value"] = std::to_string(stride_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  std::vector<int32_t> output_padding_list;
  if (op.GetAttr("output_padding", output_padding_list) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "op get outputpadding failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTranspose";
    err_map["param_name"] = "output_padding";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  if (output_padding_list.size() != kConv2dInputSizeLimit) {
    OP_LOGE(op_name.GetString(), "outputpadding size is illegal.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "output_padding";
    err_map["op_name"] = "Conv2DTranspose";
    err_map["excepted_value"] = std::to_string(kConv2dInputSizeLimit);
    err_map["input_value"] = std::to_string(output_padding_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  CHECK_KEY_IN_MAP(format2str, x_format, "x_format", return false);
  std::string x_format_str = format2str[x_format];

  int32_t h_input_position = x_format_str.find("H");
  CHECK_OP_FUNC(h_input_position < 0, return false, "get position failed: %d", h_input_position);
  int32_t w_input_position = x_format_str.find("W");
  CHECK_OP_FUNC(w_input_position < 0, return false, "get position failed: %d", w_input_position);
  int32_t c_input_position = x_format_str.find("C");
  CHECK_OP_FUNC(c_input_position < 0, return false, "get position failed: %d", c_input_position);
  int32_t n_input_position = x_format_str.find("N");
  CHECK_OP_FUNC(n_input_position < 0, return false, "get position failed: %d", n_input_position);
  int32_t dy_h = x_sizes[h_input_position];
  int32_t dy_w = x_sizes[w_input_position];
  int32_t dy_n = x_sizes[n_input_position];

  int32_t stride_h = stride_list[h_input_position];
  int32_t stride_w = stride_list[w_input_position];

  int32_t dilation_h = dilations_list[h_input_position];
  int32_t dilation_w = dilations_list[w_input_position];

  int32_t outputpadding_h = output_padding_list[h_input_position];
  int32_t outputpadding_w = output_padding_list[w_input_position];
  CHECK_KEY_IN_MAP(format2str, filter_format, "filter_format", return false);
  std::string filter_format_str = format2str[filter_format];

  int32_t h_filter_position = filter_format_str.find("H");
  CHECK_OP_FUNC(h_filter_position < 0, return false, "get position failed: %d", h_filter_position);
  int32_t w_filter_position = filter_format_str.find("W");
  CHECK_OP_FUNC(w_filter_position < 0, return false, "get position failed: %d", w_filter_position);
  int32_t c_filter_position = filter_format_str.find("C");
  CHECK_OP_FUNC(c_filter_position < 0, return false, "get position failed: %d", c_filter_position);

  int32_t filter_h = filter_sizes[h_filter_position];
  int32_t filter_w = filter_sizes[w_filter_position];
  int32_t filter_c = filter_sizes[c_filter_position];

  std::vector<int32_t> pads_list;
  if (op.GetAttr("pads", pads_list) == GRAPH_FAILED) {
    OP_LOGE(op_name.GetString(), "op get pads failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTranspose";
    err_map["param_name"] = "pads";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  int32_t groups = 1;
  if (op.GetAttr("groups", groups) == GRAPH_FAILED) {
    OP_LOGE(op_name.GetString(), "op get groups failed.");
    ErrorManager::GetInstance().ReportErrMessage("E50030", {{"op_name", "Conv2DTranspose"}, {"param_name", "groups"}});
    return false;
  }
  if (pads_list.size() != kConv2dPadSizeLimit) {
    OP_LOGE(op_name.GetString(), "op get pads_list failed.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "output_padding";
    err_map["op_name"] = "Conv2DTranspose";
    err_map["excepted_value"] = std::to_string(kConv2dPadSizeLimit);
    err_map["input_value"] = std::to_string(pads_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }

  int32_t pad_up = pads_list[kConv2dPadUpIdx];
  int32_t pad_down = pads_list[kConv2dPadDownIdx];
  int32_t pad_left = pads_list[kConv2dPadLeftIdx];
  int32_t pad_right = pads_list[kConv2dPadRightIdx];

  std::vector<int64_t> output;
  int32_t output_h = 0;
  int32_t output_w = 0;
  int32_t output_n = 0;
  int32_t output_c = 0;
  if (!CheckVectorAllZero(input_sizes) && input_sizes.size() == kConv2dInputSizeLimit) {
    output_h = input_sizes[h_input_position];
    output_w = input_sizes[w_input_position];
    output_n = input_sizes[n_input_position];
    output_c = input_sizes[c_input_position];
  } else {
    output_h = stride_h * (dy_h - 1) + outputpadding_h + ((filter_h - 1) * dilation_h + 1) - pad_up - pad_down;
    output_w = stride_w * (dy_w - 1) + outputpadding_w + ((filter_w - 1) * dilation_w + 1) - pad_left - pad_right;
    output_n = dy_n;
    output_c = filter_c * groups;
  }

  if (x_format == FORMAT_NCHW) {
    output.push_back(output_n);
    output.push_back(output_c);
    output.push_back(output_h);
    output.push_back(output_w);
  } else if (x_format == FORMAT_NHWC) {
    output.push_back(output_n);
    output.push_back(output_h);
    output.push_back(output_w);
    output.push_back(output_c);
  } else {
    OP_LOGE(op_name.GetString(), "inputSize format should be NCHW or NHWC.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "inputSize";
    err_map["op_name"] = "Conv2DTranspose";
    err_map["excepted_value"] = "NCHW or NHWC";
    err_map["input_value"] = x_format;
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  if (isRun) {
    stride_h = static_cast<int64_t>(stride_h);
    stride_w = static_cast<int64_t>(stride_w);
    dilation_h = static_cast<int64_t>(dilation_h);
    dilation_w = static_cast<int64_t>(dilation_w);
    vector<int64_t> attr_param = {stride_h, stride_w, dilation_h, dilation_w};
    vector<int64_t> output_new = {static_cast<int64_t>(output[0]), static_cast<int64_t>(output[kCDimNCHWIdx]),
                                  static_cast<int64_t>(output[kHDimNCHWIdx]),
                                  static_cast<int64_t>(output[kWDimNCHWIdx])};
    if (!check_conv2d_backprop_input_pads(op, x_sizes, x_format, filter_sizes, filter_format,
                                          output_new, x_format, attr_param)) {
      return false;
    }
  }
  // set input_size shape to dedx
  op.SetAttr("dedx", output);

  return true;
}

bool InferConv2DTransposeDataSlice(ge::Operator& op) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return false);

  auto x_tensor = op.GetInputDescByName("x");
  auto w_tensor = op.GetInputDescByName("filter");
  auto y_tensor = op.GetOutputDescByName("y");
  auto x_format = x_tensor.GetOriginFormat();
  auto w_format = w_tensor.GetOriginFormat();
  auto y_format = y_tensor.GetOriginFormat();
  auto x_shape = x_tensor.GetOriginShape().GetDims();
  auto w_shape = w_tensor.GetOriginShape().GetDims();

  vector<int32_t> stride_list;
  if (op.GetAttr("strides", stride_list) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "op get strides failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTransposeD";
    err_map["param_name"] = "strides";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  vector<int32_t> dilations_list;
  if (op.GetAttr("dilations", dilations_list) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "op get dilation failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTransposeD";
    err_map["param_name"] = "dilations";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  vector<int32_t> input_sizes;
  if (op.GetAttr("input_size", input_sizes) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "op get input_size failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTransposeD";
    err_map["param_name"] = "input_size";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return false;
  }
  CHECK_OP_FUNC(input_sizes.size() != kConv2dInputSizeLimit, return false, "Dim of input_size not 4");
  CHECK_KEY_IN_MAP(format2str, x_format, "x_format", return false);
  CHECK_KEY_IN_MAP(format2str, w_format, "w_format", return false);
  CHECK_KEY_IN_MAP(format2str, y_format, "y_format", return false);
  std::string x_format_str = format2str[x_format];
  std::string w_format_str = format2str[w_format];
  std::string y_format_str = format2str[y_format];

  int32_t h_input_position = x_format_str.find("H");
  int32_t w_input_position = x_format_str.find("W");
  int32_t h_filter_position = w_format_str.find("H");
  int32_t w_filter_position = w_format_str.find("W");
  int32_t n_y_position = y_format_str.find("N");
  int32_t c_y_position = y_format_str.find("C");
  int32_t h_y_position = y_format_str.find("H");
  int32_t w_y_position = y_format_str.find("W");
  int32_t ih = -1;
  int32_t iw = -1;
  if (x_shape != DYNAMIC_DIM_ALL) {
    ih = x_shape[h_input_position];
    iw = x_shape[w_input_position];
  }
  int32_t strh = stride_list[h_input_position];
  int32_t strw = stride_list[h_input_position];
  int32_t dilh = dilations_list[h_input_position];
  int32_t dilw = dilations_list[w_input_position];
  int32_t kh = w_shape[h_filter_position];
  int32_t kw = w_shape[w_filter_position];

  bool invalid_stride_hw = (strh <= 0) || (strw <= 0);
  if (invalid_stride_hw) {
    OP_LOGE(op_name.GetString(), "stride can not less than zero");
    ErrorManager::GetInstance().ATCReportErrMessage("E50029",
                                                    {"op_name", "param_name", "expected_value", "input_value"},
                                                    {op_name.GetString(), "strides", "positive",
                                                    std::to_string(strh) + ", " + std::to_string(strw)});
    return false;
  }
  vector<int32_t> pad_list;
  op.GetAttr("pads", pad_list);
  CHECK_SIZE(pad_list.size() != kConv2dPadSizeLimit, return false, "pad is invalid");

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return false);
  GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
  GeTensorDescPtr tensor_desc_w = op_desc->MutableInputDesc("filter");
  CHECK_PTR_NULL(tensor_desc_y, "tensor y desc", return false);
  CHECK_PTR_NULL(tensor_desc_x, "tensor x desc", return false);
  CHECK_PTR_NULL(tensor_desc_w, "tensor w desc", return false);

  vector<vector<int64_t>> y_data_slice;
  vector<vector<int64_t>> x_data_slice = {{}, {}, {}, {}, {}};
  vector<vector<int64_t>> w_data_slice = {{}, {}, {}, {}};

  if (!AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice)) {
    OP_LOGI(op_name.GetString(), "no data slice, not need infer input");
    return false;
  }
  for (size_t i = 0; i < y_data_slice.size(); i++) {
    if (y_data_slice[i].size() > 0) {
      std::vector<int32_t> y_position = {n_y_position, c_y_position, h_y_position, w_y_position};
      std::vector<int32_t> attr_param = {kh, kw, dilh, dilw, strh, strw, ih, iw};
      return data_slice_conv2d_backprop_input(op, y_data_slice, x_data_slice, w_data_slice,
                                              y_position, attr_param, input_sizes, pad_list,
                                              tensor_desc_w, tensor_desc_x, i) == GRAPH_SUCCESS;
    }
  }
  OP_LOGI(op_name.GetString(), "no data slice, not need infer input");
  return false;
}

// ----------------Conv2DTranspose-------------------
static bool CheckShapeAllZero(const GeShape& shape) {
  for (size_t i = 0; i < shape.GetDimNum(); i++) {
    if (shape.GetDim(i) != 0) {
      return false;
    }
  }
  return true;
}

static bool SetDeDxAttrForConv2DTranspose(const OpDetailInfo& op_info, const GeShape& input_shape,
                                          const ShapeValInfo& filter_value,
                                          const PosInfo& y_format_pos, GeShape& y_shape) {
  if (input_shape.IsUnknownDimNum()) {
    OP_LOGD(op_info.op_name.GetString(), "input_shape is -2, do not need check pads.");
    return true;
  }

  AttrInfo attr_paras;
  CHECK(!GetConv2dBpAttrInfo(op_info, y_format_pos, filter_value, attr_paras),
        OP_LOGE(op_info.op_name.GetString(), "failed to get attr params"), return false);

  int64_t dy_h = input_shape.GetDim(y_format_pos.h_position);
  int64_t dy_w = input_shape.GetDim(y_format_pos.w_position);
  if (CheckShapeAllZero(y_shape) || y_shape.GetDimNum() != kConv2dInputSizeLimit) {
    int32_t groups = 1;
    if (!AttrUtils::GetInt(op_info.op_desc, "groups", groups)) {
      OP_LOGW(op_info.op_name.GetString(), "op get groups failed, use default 1.");
    }
    int64_t output_h = attr_paras.h_attr.stride * (dy_h - 1) + attr_paras.h_attr.kernel - attr_paras.h_attr.pads;
    int64_t output_w = attr_paras.w_attr.stride * (dy_w - 1) + attr_paras.w_attr.kernel - attr_paras.w_attr.pads;
    int64_t output_n = input_shape.GetDim(y_format_pos.n_position);
    int64_t output_c = filter_value.c_value * groups;
    y_shape.SetDimNum(kConv2dInputSizeLimit);
    y_shape.SetDim(y_format_pos.h_position, output_h);
    y_shape.SetDim(y_format_pos.w_position, output_w);
    y_shape.SetDim(y_format_pos.n_position, output_n);
    y_shape.SetDim(y_format_pos.c_position, output_c);
  } else {
    int64_t dx_h = y_shape.GetDim(y_format_pos.h_position);
    int64_t dx_w = y_shape.GetDim(y_format_pos.w_position);
    int64_t dy_h_new = (dx_h + attr_paras.h_attr.pads - attr_paras.h_attr.kernel) / attr_paras.h_attr.stride + 1;
    int64_t dy_w_new = (dx_w + attr_paras.w_attr.pads - attr_paras.w_attr.kernel) / attr_paras.w_attr.stride + 1;
    CHECK(dy_h_new != dy_h || dy_w_new != dy_w, OP_LOGE(op_info.op_name.GetString(), "check pads attrs failed."),
          return false);
  }
  return true;
}

IMPLEMT_INFERFUNC(Conv2DTranspose, Conv2DTransposeInfer) {
  OpDetailInfo op_info;
  CHECK(op.GetName(op_info.op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);
  op_info.op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_info.op_desc, "op desc", return GRAPH_FAILED);
  CHECK(op.GetOpType(op_info.op_type) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_type"), return GRAPH_FAILED);

  OP_LOGI(op_info.op_name.GetString(), "Enter Conv2DTranspose inferfunction!");
  std::vector<std::string> inputInferDepends = {"input_size"};
  op_info.op_desc->SetOpInferDepends(inputInferDepends);

  GeTensorDescPtr y_desc = op_info.op_desc->MutableOutputDesc(0);
  CHECK_PTR_NULL(y_desc, "tensor y desc", return GRAPH_FAILED);
  GeTensorDescPtr input_sizes_desc = op_info.op_desc->MutableInputDesc(0);
  CHECK_PTR_NULL(input_sizes_desc, "tensor input_sizes desc", return GRAPH_FAILED);
  // the index of filter desc tensor is 2
  GeTensorDescPtr filter_desc = op_info.op_desc->MutableInputDesc(2);
  CHECK_PTR_NULL(filter_desc, "tensor filter desc", return GRAPH_FAILED);
  GeTensorDescPtr x_desc = op_info.op_desc->MutableInputDesc(1);
  CHECK_PTR_NULL(x_desc, "tensor x desc", return GRAPH_FAILED);

  Format y_format = y_desc->GetFormat();
  CHECK_OP_FUNC(FORMAT_RESERVED == y_format, return GRAPH_FAILED, "get format failed: %d", y_format);
  CHECK_KEY_IN_MAP(format2str, y_format, "y_format", return GRAPH_FAILED);
  PosInfo y_format_pos;
  CHECK_OP_FUNC(!GetPosInfoByFormat(y_format, y_format_pos), return GRAPH_FAILED, "get position info failed.");
  const GeShape &x_shape = x_desc->GetShape();

  bool is_dynamic = x_shape.IsUnknownShape();
  DataType input_sizes_dtype = input_sizes_desc->GetDataType();
  bool is_input_size_const = false;
  CHECK_OP_FUNC(!CheckAndSetInputSizes(op, input_sizes_dtype, "input_size", is_input_size_const, y_desc),
                return GRAPH_FAILED, "get input_size const fail.");

  ShapeValInfo filter_value;
  CHECK_OP_FUNC(!GetFilterValues(filter_desc, filter_value), return GRAPH_FAILED, "get filter_value fail.");
  OutputTuple y_infer_info(y_desc, is_input_size_const);
  // dynamic shape scene
  if (!is_input_size_const || is_dynamic) {
    CHECK_OP_FUNC(!InferConv2dBpInputOutShapeRange(op_info, input_sizes_desc, filter_value, x_desc, y_infer_info),
                  return GRAPH_FAILED, "infer out shape and range fail.");
    GeShape &out_shape = y_desc->MutableShape();
    bool is_force_dynamic =
        (out_shape.GetDimNum() == kConv2dDimSizeLimit && !is_dynamic && !out_shape.IsUnknownShape());
    if (is_force_dynamic) {
      OP_LOGD(op_info.op_name.GetString(),
              "input_sizes is Data but all tensors have no -1, so force to set HW in y to -1.");
      out_shape.SetDim(y_format_pos.h_position, -1);
      out_shape.SetDim(y_format_pos.w_position, -1);
    }
  }

  // update pads list by padding[SAME,VALID] and calculate y shape
  bool is_strict_static = is_input_size_const && !is_dynamic;
  if (is_strict_static) {
    GeShape &y_shape = y_desc->MutableShape();
    CHECK_OP_FUNC(!SetConv2dbpPadsByPadding(op_info, y_shape, y_format_pos, filter_value), return GRAPH_FAILED,
                  "update pads list by padding failed.");
    CHECK_OP_FUNC(!SetDeDxAttrForConv2DTranspose(op_info, x_shape, filter_value, y_format_pos, y_shape),
                  return GRAPH_FAILED, "set dedx or check pads failed.");
  }

  // set dtype of output desc
  DataType x_dtype = x_desc->GetDataType();
  y_desc->SetDataType(x_dtype);
  if (x_dtype == DT_INT8) {
    y_desc->SetDataType(DT_INT32);
  }
  OP_LOGI(op_info.op_name.GetString(), "Leave Conv2DTranspose Infer.");
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv2DTranspose, Conv2DTransposeVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);
  OpDescPtr op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);

  // the filter tensor index is 2
  ConstGeTensorDescPtr filter_desc = op_desc->GetInputDescPtr(2);
  CHECK_PTR_NULL(filter_desc, "filter desc", return GRAPH_FAILED);
  // the x tensor index is 1
  ConstGeTensorDescPtr x_desc = op_desc->GetInputDescPtr(1);
  CHECK_PTR_NULL(x_desc, "x desc", return GRAPH_FAILED);

  OP_LOGI(op_name.GetString(), "Enter Conv2DTranspose verifyfunction!");
  if (GRAPH_SUCCESS != VerifyConv2dbpCommon(op_name, op_desc, filter_desc, x_desc)) {
    return GRAPH_FAILED;
  }
  if (!VerifyConv2dbpOutputPadding(op_name, op_desc)) {
    return GRAPH_FAILED;
  }
  OP_LOGI(op_name.GetString(), "Leave Conv2DTranspose verifyfunction!");
  return GRAPH_SUCCESS;
}

IMPLEMT_INFER_DATA_SLICE(Conv2DTranspose, Conv2DTransposeInferDataSlice) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter Conv2DTranspose InferDataSlice.");
  if (!InferConv2DTransposeDataSlice(op)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_DATA_SLICE_FUNC_REG(Conv2DTranspose, Conv2DTransposeInferDataSlice);
INFER_FUNC_REG(Conv2DTranspose, Conv2DTransposeInfer);
VERIFY_FUNC_REG(Conv2DTranspose, Conv2DTransposeVerify);
// ----------------Conv2DTransposeD-------------------
template <typename T1>
static bool CheckVectorLessZero(const std::vector<T1>& input_list) {
  for (uint32_t i = 0; i < input_list.size(); i++) {
    if (input_list[i] < 0) {
      return true;
    }
  }
  return false;
}

template <typename T1, typename T2>
static void ProcessOnnxAttr(ge::Operator& op, const std::vector<T1>& x_sizes, Format x_format,
                            const std::vector<T2>& filter_sizes, Format filter_format) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return);

  OP_LOGD(op_name.GetString(), "Enter ProcessOnnxAttr, process auto_pad and output_shape.");

  std::string auto_pad;
  if (op.GetAttr("auto_pad", auto_pad) != GRAPH_SUCCESS) {
    auto_pad = "NOTSET";
  }
  OP_LOGD(op_name.GetString(), "get default auto_pad[%s].", auto_pad.c_str());

  std::vector<int32_t> output_shape_list;
  std::vector<int32_t> stride_list;
  std::vector<int32_t> dilations_list;
  bool get_attr_invalid = op.GetAttr("output_shape", output_shape_list) != GRAPH_SUCCESS ||
                          op.GetAttr("strides", stride_list) != GRAPH_SUCCESS ||
                          op.GetAttr("dilations", dilations_list) != GRAPH_SUCCESS;
  if (get_attr_invalid) {
    return;
  }

  // dynamic shape scenario, no need process
  bool dynamic_scene = CheckVectorLessZero(x_sizes) || CheckVectorLessZero(filter_sizes) ||
                       CheckVectorLessZero(output_shape_list) || CheckVectorLessZero(stride_list) ||
                       CheckVectorLessZero(dilations_list);
  if (dynamic_scene) {
    OP_LOGD(op_name.GetString(), "no need process with dynamic shape scenario.");
    return;
  }
  CHECK_KEY_IN_MAP(format2str, x_format, "x_format", return);
  std::string x_format_str = format2str[x_format];
  int32_t h_input_position = x_format_str.find("H");
  int32_t w_input_position = x_format_str.find("W");
  int32_t input_h = x_sizes[h_input_position];
  int32_t input_w = x_sizes[w_input_position];
  int32_t stride_h = stride_list[h_input_position];
  int32_t stride_w = stride_list[w_input_position];
  int32_t dilation_h = dilations_list[h_input_position];
  int32_t dilation_w = dilations_list[w_input_position];

  CHECK_KEY_IN_MAP(format2str, filter_format, "filter_format", return);
  std::string filter_format_str = format2str[filter_format];
  int32_t h_filter_position = filter_format_str.find("H");
  int32_t w_filter_position = filter_format_str.find("W");
  int32_t filter_h = filter_sizes[h_filter_position];
  int32_t filter_w = filter_sizes[w_filter_position];

  int32_t output_shape_h = output_shape_list[0];
  int32_t output_shape_w = output_shape_list[1];

  if (auto_pad == "NOTSET") {
    std::vector<int32_t> pads_list;
    if (op.GetAttr("pads", pads_list) != GRAPH_SUCCESS) {
      OP_LOGD(op_name.GetString(), "can't get pads.");
      return;
    }

    if (pads_list.size() != kConv2dPadSizeLimit) {
      OP_LOGD(op_name.GetString(), "pads_list size error.");
      return;
    }

    int32_t pad_up = pads_list[kConv2dPadUpIdx];
    int32_t pad_down = pads_list[kConv2dPadDownIdx];
    int32_t pad_left = pads_list[kConv2dPadLeftIdx];
    int32_t pad_right = pads_list[kConv2dPadRightIdx];

    int32_t standard_h = stride_h * (input_h - 1) + ((filter_h - 1) * dilation_h + 1) - pad_up - pad_down;
    int32_t output_padding_h = output_shape_h > standard_h ? output_shape_h - standard_h : 0;
    int32_t standard_w = stride_w * (input_w - 1) + ((filter_w - 1) * dilation_w + 1) - pad_left - pad_right;
    int32_t output_padding_w = output_shape_w > standard_w ? output_shape_w - standard_w : 0;
    OP_LOGD(op_name.GetString(), "result: oPadH[%d], oPadW[%d]", output_padding_h, output_padding_w);

    std::vector<int32_t> output_padding_list = {0, 0, 0, 0};
    output_padding_list[h_input_position] = output_padding_h;
    output_padding_list[w_input_position] = output_padding_w;
    op.SetAttr("output_padding", output_padding_list);
  } else {
    int32_t pad_h = 0;
    int32_t pad_w = 0;
    int32_t output_padding_h = 0;
    int32_t output_padding_w = 0;

    int32_t standard_h = stride_h * (input_h - 1) + ((filter_h - 1) * dilation_h + 1);
    if (output_shape_h > standard_h) {
      pad_h = 0;
      output_padding_h = output_shape_h - standard_h;
    } else {
      pad_h = standard_h - output_shape_h;
      output_padding_h = 0;
    }

    int32_t standard_w = stride_w * (input_w - 1) + ((filter_w - 1) * dilation_w + 1);
    if (output_shape_w > standard_w) {
      pad_w = 0;
      output_padding_w = output_shape_w - standard_w;
    } else {
      pad_w = standard_w - output_shape_w;
      output_padding_w = 0;
    }

    OP_LOGD(op_name.GetString(), "result: oPadH[%d], oPadW[%d]", output_padding_h, output_padding_w);

    std::vector<int32_t> pads_list = {0, 0, 0, 0};
    std::vector<int32_t> output_padding_list = {0, 0, 0, 0};
    output_padding_list[h_input_position] = output_padding_h;
    output_padding_list[w_input_position] = output_padding_w;
    op.SetAttr("output_padding", output_padding_list);
    bool not_same_upper_padding = auto_pad == "VALID" || auto_pad == "SAME_LOWER";
    if (not_same_upper_padding) {
      // up, down, left, right
      pads_list[kConv2dPadUpIdx] = pad_h >> 1;
      pads_list[kConv2dPadDownIdx] = pad_h - (pad_h >> 1);
      pads_list[kConv2dPadLeftIdx] = pad_w >> 1;
      pads_list[kConv2dPadRightIdx] = pad_w - (pad_w >> 1);
      op.SetAttr("pads", pads_list);
    } else if (auto_pad == "SAME_UPPER") {
      pads_list[kConv2dPadUpIdx] = pad_h - (pad_h >> 1);
      pads_list[kConv2dPadDownIdx] = pad_h >> 1;
      pads_list[kConv2dPadLeftIdx] = pad_w - (pad_w >> 1);
      pads_list[kConv2dPadRightIdx] = pad_w >> 1;
      op.SetAttr("pads", pads_list);
    }

    OP_LOGD(op_name.GetString(), "the pads: [top, bottom, left, right] is %s", ops::to_string(pads_list).c_str());
  }

  return;
}

IMPLEMT_INFERFUNC(Conv2DTransposeD, Conv2DTransposeDInfer) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter Conv2DTransposeD InferShape.");

  // get shape for output from input_size
  std::vector<int32_t> input_sizes;
  if (op.GetAttr("input_size", input_sizes) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "get input_size list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTranspose";
    err_map["param_name"] = "input_size";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (input_sizes.size() != kConv2dInputSizeLimit) {
    OP_LOGE(op_name.GetString(), "input_size list should be 4d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "input_size";
    err_map["op_name"] = "Conv2DTranspose";
    err_map["excepted_value"] = std::to_string(kConv2dInputSizeLimit);
    err_map["input_value"] = std::to_string(input_sizes.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> filter_sizes = op.GetInputDescByName("filter").GetShape().GetDims();
  std::vector<int64_t> x_sizes = op.GetInputDescByName("x").GetShape().GetDims();
  Format filter_format = op.GetInputDescByName("filter").GetFormat();
  auto x_desc = op.GetInputDescByName("x");
  // get dtype for output from x
  auto x_dtype = x_desc.GetDataType();
  auto y_desc = op.GetOutputDescByName("y");
  Format input_format = y_desc.GetFormat();
  Format x_format = op.GetInputDescByName("x").GetFormat();
  CHECK_OP_FUNC(FORMAT_RESERVED == filter_format, return GRAPH_FAILED, "get format failed: %d", filter_format);
  CHECK_OP_FUNC(FORMAT_RESERVED == input_format, return GRAPH_FAILED, "get format failed: %d", input_format);
  CHECK_OP_FUNC(FORMAT_RESERVED == x_format, return GRAPH_FAILED, "get format failed: %d", x_format);

  // process ONNX attr
  if (CheckVectorAllZero(input_sizes)) {
    ProcessOnnxAttr(op, x_sizes, x_format, filter_sizes, filter_format);
  }

  // update pads list by padding[SAME,VALID] and calculate input_size
  bool isRun = false;
  if (!SetInputsizeListConv2DTranspose(op, x_sizes, x_format, filter_sizes, filter_format, input_sizes, isRun)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "Set Conv2DTranspose InputsizeList failed.");
    return GRAPH_FAILED;
  }
  // set dtype of output desc
  if (x_dtype == DT_INT8) {
    y_desc.SetDataType(DT_INT32);
  } else {
    y_desc.SetDataType(x_dtype);
  }
  // set shape of output desc, input_size should match the format of y
  std::vector<int32_t> dedx;
  if (op.GetAttr("dedx", dedx) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "get dedx list failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTranspose";
    err_map["param_name"] = "dedx";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  if (dedx.size() != kConv2dInputSizeLimit) {
    OP_LOGE(op_name.GetString(), "dedx list should be 4d.");
    map<std::string, std::string> err_map;
    err_map["param_name"] = "dedx";
    err_map["op_name"] = "Conv2DTranspose";
    err_map["excepted_value"] = std::to_string(kConv2dInputSizeLimit);
    err_map["input_value"] = std::to_string(dedx.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int64_t> out_shape;
  for (auto i : dedx) {
    out_shape.push_back(i);
  }

  y_desc.SetShape(ge::Shape(out_shape));
  // update input_size shape
  op.SetAttr("input_size", dedx);

  // update output desc
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "update output desc failed.");
    map<std::string, std::string> err_map;
    err_map["op_name"] = "Conv2DTranspose";
    err_map["param_name"] = "output y";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Conv2DTransposeD, Conv2DTransposeDVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);
  OpDescPtr op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);

  // the filter tensor index is 1
  ConstGeTensorDescPtr filter_desc = op_desc->GetInputDescPtr(1);
  CHECK_PTR_NULL(filter_desc, "filter desc", return GRAPH_FAILED);
  // the x tensor index is 0
  ConstGeTensorDescPtr x_desc = op_desc->GetInputDescPtr(0);
  CHECK_PTR_NULL(x_desc, "x desc", return GRAPH_FAILED);

  OP_LOGI(op_name.GetString(), "Enter Conv2DTransposeD verifyfunction!");
  if (GRAPH_SUCCESS != VerifyConv2dbpCommon(op_name, op_desc, filter_desc, x_desc)) {
    return GRAPH_FAILED;
  }
  if (!VerifyConv2dbpOutputPadding(op_name, op_desc)) {
    return GRAPH_FAILED;
  }
  OP_LOGI(op_name.GetString(), "Leave Conv2DTransposeD verifyfunction!");
  return GRAPH_SUCCESS;
}

IMPLEMT_INFER_DATA_SLICE(Conv2DTransposeD, Conv2DTransposeDInferDataSlice) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  OP_LOGD(op_name.GetString(), "Enter Conv2DTransposeD InferDataSlice.");
  // process ONNX attr
  auto x_tensor = op.GetInputDescByName("x");
  auto w_tensor = op.GetInputDescByName("filter");
  auto x_format = x_tensor.GetOriginFormat();
  auto w_format = w_tensor.GetOriginFormat();
  auto x_shape = x_tensor.GetOriginShape().GetDims();
  auto w_shape = w_tensor.GetOriginShape().GetDims();
  ProcessOnnxAttr(op, x_shape, x_format, w_shape, w_format);

  if (!InferConv2DTransposeDataSlice(op)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_DATA_SLICE_FUNC_REG(Conv2DTransposeD, Conv2DTransposeDInferDataSlice);
INFER_FUNC_REG(Conv2DTransposeD, Conv2DTransposeDInfer);
VERIFY_FUNC_REG(Conv2DTransposeD, Conv2DTransposeDVerify);

// ----------------DeformableOffsets-------------------
static graphStatus VerifyDeformableOffsetsInput(const ge::Operator& op) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  auto x_desc = op.GetInputDescByName("x");
  auto x_shape = x_desc.GetShape().GetDims();
  if (x_shape.size() != kDeformDimSizeLimit) {
    OP_LOGE(op_name.GetString(), "Input x should be 4d.");
    map<string, string> err_map;
    err_map["op_name"] = "DeformableOffsets";
    err_map["param_name"] = "x_shape";
    err_map["expected_value"] = std::to_string(kDeformDimSizeLimit);
    err_map["input_value"] = std::to_string(x_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  auto offsets_desc = op.GetInputDescByName("offsets");
  auto offsets_shape = offsets_desc.GetShape().GetDims();
  Format offset_format = offsets_desc.GetFormat();
  CHECK_OP_FUNC(FORMAT_RESERVED == offset_format, return GRAPH_FAILED, "get format failed: %d", offset_format);
  CHECK_OP_FUNC(offset_format != FORMAT_NCHW && offset_format != FORMAT_NHWC, return GRAPH_FAILED,
                "Input offset's format should be NCHW or NHWC, actual is [%s]",
                TypeUtils::FormatToSerialString(offset_format).c_str())
  if (offsets_shape.size() != kDeformDimSizeLimit) {
    OP_LOGE(op_name.GetString(), "Offsets shape should be 4d, actual is [%zu].",
            offsets_shape.size());
    map<string, string> err_map;
    err_map["op_name"] = "DeformableOffsets";
    err_map["param_name"] = "offsets_shape";
    err_map["expected_value"] = std::to_string(kDeformDimSizeLimit);
    err_map["input_value"] = std::to_string(offsets_shape.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::string offset_format_str = format2str[offset_format];
  int32_t offset_pos_c = offset_format_str.find("C");
  CHECK_OP_FUNC(offset_pos_c < 0, return GRAPH_FAILED, "get position failed: %d", offset_pos_c);
  int32_t offset_pos_n = offset_format_str.find("N");
  CHECK_OP_FUNC(offset_pos_n < 0, return GRAPH_FAILED, "get position failed: %d", offset_pos_n);
  int32_t offset_c = offsets_shape[offset_pos_c];
  int32_t offset_n = offsets_shape[offset_pos_n];

  Format x_format = x_desc.GetFormat();
  CHECK_OP_FUNC(FORMAT_RESERVED == x_format, return GRAPH_FAILED, "get format failed: %d", x_format);
  std::string x_format_str = format2str[x_format];
  if (offset_format_str != x_format_str) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(),
                          "Offset_format should be same as x_format, x [%s], offset [%s].",
                          x_format_str.c_str(), offset_format_str.c_str());
    return GRAPH_FAILED;
  }
  int32_t x_pos_c = x_format_str.find("C");
  CHECK_OP_FUNC(x_pos_c < 0, return GRAPH_FAILED, "get position failed: %d", x_pos_c);
  int32_t x_pos_n = x_format_str.find("N");
  CHECK_OP_FUNC(x_pos_n < 0, return GRAPH_FAILED, "get position failed: %d", x_pos_n);
  int32_t x_c = x_shape[x_pos_c];
  int32_t x_n = x_shape[x_pos_n];

  CHECK_OP_FUNC(x_n != offset_n, return GRAPH_FAILED,
                "Offset batch should be equal to x batch, x_batch: %d, offset_batch: %d",
                x_n, offset_n);

  int32_t dg;
  if (op.GetAttr("deformable_groups", dg) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "op get deformable_groups failed.");
    map<string, string> err_map;
    err_map["op_name"] = "DeformableOffsets";
    err_map["param_name"] = "deformable_groups";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  CHECK_OP_FUNC(dg <= 0, return GRAPH_FAILED, "Deformable_groups less than 0, deformable_groups [%d].", dg);

  if (x_c % dg != 0) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "Input x's channel can not divide deformable_groups");
    return GRAPH_FAILED;
  }

  std::vector<int32_t> ksize_list;
  if (op.GetAttr("ksize", ksize_list) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "Op get ksize failed.");
    map<string, string> err_map;
    err_map["op_name"] = "DeformableOffsets";
    err_map["param_name"] = "ksize";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  if (ksize_list.size() != kDeformKsizeLimit) {
    OP_LOGE(op_name.GetString(), "Input ksize should be 2d.");
    map<string, string> err_map;
    err_map["op_name"] = "DeformableOffsets";
    err_map["param_name"] = "ksize";
    err_map["expected_value"] = std::to_string(kDeformKsizeLimit);
    err_map["input_value"] = std::to_string(ksize_list.size());
    std::string report_error_code = "E50029";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  int32_t ksize_h = ksize_list[0];
  int32_t ksize_w = ksize_list[1];

  bool modulated;
  if (op.GetAttr("modulated", modulated) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "Op get modulated failed.");
    map<string, string> err_map;
    err_map["op_name"] = "DeformableOffsets";
    err_map["param_name"] = "modulated";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  if (!modulated) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "Currently modulated must be true.");
    return GRAPH_FAILED;
  }

  int multiple = modulated ? 3 : 2;
  if (offset_c != dg * ksize_h * ksize_w * multiple) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(),
                          "Check offset_c failed, offsets_c [%d], should be [%d].",
                          offset_c, dg * ksize_h * ksize_w * multiple);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(DeformableOffsets, DeformableOffsetsInfer) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  auto x_desc = op.GetInputDescByName("x");
  auto offsets_desc = op.GetInputDescByName("offsets");
  auto y_desc = op.GetOutputDescByName("y");

  std::vector<int32_t> ksize_list;
  if (op.GetAttr("ksize", ksize_list) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "Op get ksize failed.");
    map<string, string> err_map;
    err_map["op_name"] = "DeformableOffsets";
    err_map["param_name"] = "ksize";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> stride_list;
  if (op.GetAttr("strides", stride_list) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "Get strides list failed.");
    map<string, string> err_map;
    err_map["op_name"] = "DeformableOffsets";
    err_map["param_name"] = "strides";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> pads_list;
  if (op.GetAttr("pads", pads_list) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "Get pads list failed.");
    map<string, string> err_map;
    err_map["op_name"] = "DeformableOffsets";
    err_map["param_name"] = "pads";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> dilations_list;
  if (op.GetAttr("dilations", dilations_list) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "Get dilations list failed.");
    map<string, string> err_map;
    err_map["op_name"] = "DeformableOffsets";
    err_map["param_name"] = "dilations";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  int32_t dilations_h = 0;
  int32_t dilations_w = 0;
  int32_t stride_h = 0;
  int32_t stride_w = 0;

  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_SUCCESS) {
    if (data_format == "NCHW") {
      dilations_h = dilations_list[2];
      dilations_w = dilations_list[3];
      stride_h = stride_list[2];
      stride_w = stride_list[3];
    } else if (data_format == "NHWC") {
      dilations_h = dilations_list[1];
      dilations_w = dilations_list[2];
      stride_h = stride_list[1];
      stride_w = stride_list[2];
    } else {
      CUBE_INNER_ERR_REPORT(op_name.GetString(),
                            "Data_format should be 'NCHW' or 'NHWC', actual is [%s].",
                            data_format.c_str());
        return GRAPH_FAILED;
    }
  }
  if ((stride_h <= 0) || (stride_w <= 0)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(),
                          "Stride should be greater than 0, stride_h [%d], stride_w [%d].",
                            stride_h, stride_w);
    return GRAPH_FAILED;
  }

  auto x_format = x_desc.GetFormat();
  auto offsets_format = offsets_desc.GetFormat();
  auto y_format = y_desc.GetFormat();
  CHECK_OP_FUNC(FORMAT_RESERVED == x_format, return GRAPH_FAILED, "get format failed: %d", x_format);
  CHECK_OP_FUNC(FORMAT_RESERVED == offsets_format, return GRAPH_FAILED, "get format failed: %d", offsets_format);
  CHECK_OP_FUNC(FORMAT_RESERVED == y_format, return GRAPH_FAILED, "get format failed: %d", y_format);

  std::string x_format_str = format2str[x_format];
  std::string offsets_format_str = format2str[offsets_format];
  std::string y_format_str = format2str[y_format];
  if (x_format_str != data_format || y_format_str != data_format ||
      offsets_format_str != data_format) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(),
          "Check x, offsets, y format failed, x [%s], offsets [%s] y [%s], data_format [%s].",
          x_format_str.c_str(), offsets_format_str.c_str(), y_format_str.c_str(), data_format.c_str());
    return GRAPH_FAILED;
  }

  auto x_shape = x_desc.GetShape().GetDims();
  auto offsets_shape = offsets_desc.GetShape().GetDims();
  int32_t ksize_h = ksize_list[0];
  int32_t ksize_w = ksize_list[1];
  // formula : (ksize_w - 1) * dilations_w + 1
  const int32_t dil_ksize_h = (ksize_h - 1) * dilations_h + 1;
  const int32_t dil_ksize_w = (ksize_w - 1) * dilations_w + 1;

  int32_t pos_h = data_format.find("H");
  CHECK_OP_FUNC(pos_h < 0, return GRAPH_FAILED, "get position failed: %d", pos_h);
  int32_t pos_w = data_format.find("W");
  CHECK_OP_FUNC(pos_w < 0, return GRAPH_FAILED, "get position failed: %d", pos_w);
  int32_t x_h = x_shape[pos_h];
  int32_t x_w = x_shape[pos_w];
  int32_t offsets_h = offsets_shape[pos_h];
  int32_t offsets_w = offsets_shape[pos_w];

  int32_t pad_u = pads_list[0];
  int32_t pad_d = pads_list[1];
  int32_t pad_l = pads_list[2];
  int32_t pad_r = pads_list[3];

  // formula : (width + pad_l + pad_r - ksize_dil_w) / stride_w + 1
  int32_t conv_out_h = -1;
  int32_t conv_out_w = -1;
  if (x_h != -1) {
    conv_out_h = (x_h + pad_u + pad_d - dil_ksize_h) / stride_h + 1;
  }
  if (x_w != -1) {
    conv_out_w = (x_w + pad_l + pad_r - dil_ksize_w) / stride_w + 1;
  }

  CHECK_OP_FUNC(conv_out_h != offsets_h || conv_out_w != offsets_w, return GRAPH_FAILED,
                "Input_offsets h/w should be same as h/w after convolution, \
                offsets: [h:%d, w:%d], conv_out: [h:%d, w:%d].",
                offsets_h, offsets_w, conv_out_h, conv_out_w);

  std::vector<int64_t> y_shape(x_shape);
  y_shape[pos_h] = offsets_h * ksize_h;
  y_shape[pos_w] = offsets_w * ksize_w;
  y_desc.SetShape(ge::Shape(y_shape));

  auto x_dtype = x_desc.GetDataType();
  y_desc.SetDataType(x_dtype);

  std::vector<std::pair<int64_t, int64_t>> x_range;
  std::vector<std::pair<int64_t, int64_t>> offsets_range;
  CHECK_OP_FUNC((x_desc.GetShapeRange(x_range) != GRAPH_SUCCESS) ||
                (offsets_desc.GetShapeRange(offsets_range) != GRAPH_SUCCESS),
                return GRAPH_FAILED, "Fail to get input_x or input_offsets range");
  std::vector<std::pair<int64_t, int64_t>> y_range(x_range);
  if ((!x_range.empty()) && (!offsets_range.empty())) {
    y_range[pos_h].first = offsets_range[pos_h].first * ksize_h;
    y_range[pos_h].second = offsets_range[pos_h].second * ksize_h;
    y_range[pos_w].first = offsets_range[pos_w].first * ksize_w;
    y_range[pos_w].second = offsets_range[pos_w].second * ksize_w;
  }
  y_desc.SetShapeRange(y_range);

  // update output desc
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    OP_LOGE(op_name.GetString(), "Update output desc failed.");
    map<string, string> err_map;
    err_map["op_name"] = "DeformableOffsets";
    err_map["param_name"] = "y";
    std::string report_error_code = "E50030";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(DeformableOffsets, DeformableOffsetsVerify) {
  if (VerifyDeformableOffsetsInput(op) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DeformableOffsets, DeformableOffsetsInfer);
VERIFY_FUNC_REG(DeformableOffsets, DeformableOffsetsVerify);

IMPLEMT_INFERFUNC(DeformableOffsetsGrad, DeformableOffsetsGradInfer) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  auto x_desc = op.GetInputDescByName("x");
  auto offsets_desc = op.GetInputDescByName("offsets");
  auto grad_x_desc = op.GetOutputDescByName("grad_x");
  auto grad_offsets_desc = op.GetOutputDescByName("grad_offsets");

  auto x_format = x_desc.GetFormat();
  auto offsets_format = offsets_desc.GetFormat();
  auto grad_x_format = grad_x_desc.GetFormat();
  auto grad_offsets_format = grad_offsets_desc.GetFormat();
  CHECK_OP_FUNC(FORMAT_RESERVED == x_format, return GRAPH_FAILED, "get format failed: %d", x_format);
  CHECK_OP_FUNC(FORMAT_RESERVED == offsets_format, return GRAPH_FAILED, "get format failed: %d", offsets_format);
  CHECK_OP_FUNC(FORMAT_RESERVED == grad_x_format, return GRAPH_FAILED, "get format failed: %d", grad_x_format);
  CHECK_OP_FUNC(FORMAT_RESERVED == grad_offsets_format, return GRAPH_FAILED,
                "get format failed: %d", grad_offsets_format);
  std::string x_format_str = format2str[x_format];
  std::string offsets_format_str = format2str[offsets_format];
  std::string grad_x_format_str = format2str[grad_x_format];
  std::string grad_offsets_format_str = format2str[grad_offsets_format];
  if (x_format_str != grad_x_format_str) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "Grad_x format should be same as input x format");
    return GRAPH_FAILED;
  }
  if (offsets_format_str != grad_offsets_format_str) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(),
                          "Grad_offsets format should be same as offsets format, \
                          actually grad_offsets: [%s], offsets: [%s]",
                          grad_offsets_format_str.c_str(), offsets_format_str.c_str());
    return GRAPH_FAILED;
  }

  auto x_shape = x_desc.GetShape();
  auto x_dtype = x_desc.GetDataType();
  auto offsets_shape = offsets_desc.GetShape();
  auto offsets_dtype = offsets_desc.GetDataType();
  grad_x_desc.SetShape(x_shape);
  grad_x_desc.SetDataType(x_dtype);
  grad_offsets_desc.SetShape(offsets_shape);
  grad_offsets_desc.SetDataType(offsets_dtype);

  std::vector<std::pair<int64_t, int64_t>> x_range;
  std::vector<std::pair<int64_t, int64_t>> offsets_range;
  if ((x_desc.GetShapeRange(x_range) != GRAPH_SUCCESS) ||
      (offsets_desc.GetShapeRange(offsets_range) != GRAPH_SUCCESS)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "Fail to get input_x or input_offsets range");
    return GRAPH_FAILED;
  }
  grad_x_desc.SetShapeRange(x_range);
  grad_offsets_desc.SetShapeRange(offsets_range);

  if ((op.UpdateOutputDesc("grad_x", grad_x_desc) != GRAPH_SUCCESS) ||
      (op.UpdateOutputDesc("grad_offsets", grad_offsets_desc) != GRAPH_SUCCESS)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "Fail to update output desc");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(DeformableOffsetsGrad, DeformableOffsetsGradVerify) {
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  auto grad_desc = op.GetInputDescByName("grad");
  auto x_desc = op.GetInputDescByName("x");
  auto offsets_desc = op.GetInputDescByName("offsets");

  auto grad_dims = grad_desc.GetShape().GetDims();
  auto x_dims = x_desc.GetShape().GetDims();
  auto offsets_dims = offsets_desc.GetShape().GetDims();

  if (grad_dims.size() != 4 || x_dims.size() != 4 || offsets_dims.size() != 4) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(),
                          "Grad, x and offsets dimension should all be 4, actually [%zu, %zu, %zu]",
                          grad_dims.size(), x_dims.size(), offsets_dims.size());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DeformableOffsetsGrad, DeformableOffsetsGradInfer);
VERIFY_FUNC_REG(DeformableOffsetsGrad, DeformableOffsetsGradVerify);

IMPLEMT_COMMON_INFERFUNC(DilationInferShape)
{
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);
  int64_t h_position = 0;
  int64_t w_position = 0;
  auto x_desc = op.GetInputDescByName("x");
  auto x_shape = x_desc.GetShape().GetDims();
  std::vector<int64_t> dilations;
  dilations = GetAttrValue(op, "dilations");
  std::vector<int64_t> out_shape;
  std::vector<int64_t> pads;
  pads = GetAttrValue(op, "pads");
  auto x_format = x_desc.GetFormat();
  std::string x_format_str = format2str[x_format];
  if (!GetDimInFormat(string(op_name.GetString()), x_format_str, "H", h_position) ||
      !GetDimInFormat(string(op_name.GetString()), x_format_str, "W", w_position)) {
    return GRAPH_FAILED;
  }
  for (size_t i = 0; i < x_shape.size(); i++) {
    int shape_value = (x_shape[i] - 1) * dilations[i] + 1;
    if (!pads.empty()) {
      if (i == static_cast<size_t>(h_position)) {
        shape_value += (pads[0] + pads[1]);
      }
      else if (i == static_cast<size_t>(w_position)) {
        shape_value += (pads[2] + pads[3]);
      }
    }
    out_shape.push_back(shape_value);
  }
  auto y_desc = op.GetOutputDescByName("y");
  y_desc.SetShape(ge::Shape(out_shape));
  (void)op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Dilation, DilationVerify)
{
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return GRAPH_FAILED);

  auto x_desc = op.GetInputDescByName("x");
  auto x_shape = x_desc.GetShape().GetDims();
  std::vector<int64_t> dilations;
  dilations = GetAttrValue(op, "dilations");
  std::vector<int64_t> out_shape;
  if (x_shape.size() != dilations.size()) {
      map<string, string> err_map;
      err_map["op_name"] = op_name.GetString();
      err_map["attr_name"] = "dim";
      err_map["param1_name"] = "x";
      err_map["param1_value"] = std::to_string(x_shape.size());
      err_map["param2_name"] = "dilations";
      err_map["param2_value"] = std::to_string(dilations.size());
      std::string report_error_code = "E50031";
      ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
      return GRAPH_FAILED;
  }
  for (size_t i = 0; i < dilations.size(); i++) {
      if (dilations[i] <= 0) {
          map<string, string> err_map;
          err_map["op_name"] = op_name.GetString();
          err_map["description"] = "Elements of dilations should be positive integers";
          std::string report_error_code = "E50060";
          ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
          return GRAPH_FAILED;
      }
  }
  std::vector<int64_t> pads;
  pads = GetAttrValue(op, "pads");
  if (!pads.empty() && pads.size() != kConv2dPadSizeLimit) {
    map<string, string> err_map;
    err_map["op_name"] = op_name.GetString();
    err_map["description"] = "pads must be empty or size is 4";
    std::string report_error_code = "E50060";
    ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Dilation, DilationInferShape);
VERIFY_FUNC_REG(Dilation, DilationVerify);
} // namespace ge
