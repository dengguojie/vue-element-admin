/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file tbe_aipp_fusion_rule.cpp
 * \brief
 */
#include "tbe_aipp_fusion_rule.h"
#include <math.h>
#include <vector>
#include <nlohmann/json.hpp>
#include "common/util/platform_info.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph/utils/attr_utils.h"

namespace fe {
/***************************************************************
check! strideh optim case, aipp can not fusion with conv.
***************************************************************/
bool TbeAippFusionRule::CheckAippConvStridehValidation(const ge::NodePtr conv_node) {
  ge::Format first_format = conv_node->GetOpDesc()->GetInputDesc(0).GetOriginFormat();
  vector<int64_t> first_dims(4);
  first_dims = conv_node->GetOpDesc()->GetInputDesc(0).GetOriginShape().GetDims();
  FUSION_PASS_CHECK(first_dims.size() != 4,
                    OP_LOGI(conv_node->GetType().c_str(),
                            "Get node[%s]'s first input shape not success.",
                            conv_node->GetName().c_str()),
                    return false);
  vector<int64_t> pads(4);
  ge::AttrUtils::GetListInt(conv_node->GetOpDesc(), "pads", pads);
  FUSION_PASS_CHECK(pads.size() != 4,
                    OP_LOGI(conv_node->GetType().c_str(),
                            "Get node[%s]'s pads attr not success.",
                            conv_node->GetName().c_str()),
                    return false);
  vector<int64_t> strides;
  ge::AttrUtils::GetListInt(conv_node->GetOpDesc(), "strides", strides);
  FUSION_PASS_CHECK(strides.size() != 4,
                    OP_LOGI(conv_node->GetType().c_str(),
                            "Get node[%s]'s strides attr not success.",
                            conv_node->GetName().c_str()),
                    return false);
  int64_t pad_up = pads[0];
  int64_t pad_down = pads[1];
  int64_t pad_left = pads[2];
  int64_t pad_right = pads[3];

  int64_t stride_h = 0;
  if (first_format == ge::FORMAT_NCHW) {
    stride_h = strides[2];
  } else if (first_format == ge::FORMAT_NHWC) {
    stride_h = strides[1];
  }
  ge::Format second_format = conv_node->GetOpDesc()->GetInputDesc(1).GetOriginFormat();
  std::vector<int64_t> second_dims(4);
  second_dims = conv_node->GetOpDesc()->GetInputDesc(1).GetOriginShape().GetDims();
  FUSION_PASS_CHECK(second_dims.size() != 4,
                    OP_LOGI(conv_node->GetType().c_str(),
                            "Get node[%s]'s second input shape not success.",
                            conv_node->GetName().c_str()),
                    return false);
  int64_t height_ft = 0;
  if (second_format == ge::FORMAT_NCHW) {
    height_ft = second_dims[2];
  } else if (second_format == ge::FORMAT_NHWC) {
    height_ft = second_dims[1];
  } else if (second_format == ge::FORMAT_HWCN) {
    height_ft = second_dims[0];
  }
  ge::Format filter_format = conv_node->GetOpDesc()->GetInputDesc(1).GetFormat();
  int64_t pad_sum = pad_up + pad_down + pad_left + pad_right;
  bool strideh_opti_flag =
      (height_ft == 1 && stride_h > 1 && pad_sum == 0 && filter_format != ge::FORMAT_FRACTAL_Z_C04);
  FUSION_PASS_CHECK(strideh_opti_flag == true, OP_LOGI(conv_node->GetType().c_str(),
                                                       "node[%s]'s is the strideh optim case"
                                                       "can not fusion.",
                                                       conv_node->GetName().c_str()),
                    return false);
  return true;
}
/***************************************************************
check! load2d case, aipp can not fusion with conv.
***************************************************************/
bool TbeAippFusionRule::CheckConvload2dNodeValidation(const ge::NodePtr conv_node) {
  ge::Format first_format = conv_node->GetOpDesc()->GetInputDesc(0).GetOriginFormat();
  vector<int64_t> strides;
  ge::AttrUtils::GetListInt(conv_node->GetOpDesc(), "strides", strides);
  FUSION_PASS_CHECK(strides.size() != 4,
                    OP_LOGI(conv_node->GetType().c_str(),
                            "Get node[%s]'s strides attr not success.",
                            conv_node->GetName().c_str()),
                    return false);
  int64_t stride_h = 1;
  int64_t stride_w = 1;
  if (first_format == ge::FORMAT_NCHW) {
    stride_h = strides[2];
    stride_w = strides[3];
  } else if (first_format == ge::FORMAT_NHWC) {
    stride_h = strides[1];
    stride_w = strides[2];
  }
  bool stride_flg = (stride_h == 1 && stride_w == 1);

  ge::Format second_format = conv_node->GetOpDesc()->GetInputDesc(1).GetOriginFormat();
  std::vector<int64_t> second_dims = conv_node->GetOpDesc()->GetInputDesc(1).GetOriginShape().GetDims();
  FUSION_PASS_CHECK(second_dims.size() != 4,
                    OP_LOGI(conv_node->GetType().c_str(),
                            "Get node[%s]'s second input shape not success.",
                            conv_node->GetName().c_str()),
                    return false);
  bool filter_flg = false;
  if (second_format == ge::FORMAT_NCHW) {
    if (second_dims[2] == 1 && second_dims[3] == 1) {
      filter_flg = true;
    }

  } else if (second_format == ge::FORMAT_NHWC) {
    if (second_dims[1] == 1 && second_dims[2] == 1) {
      filter_flg = true;
    }
  } else {
    if (second_dims[0] == 1 && second_dims[1] == 1) {
      filter_flg = true;
    }
  }

  vector<int64_t> pads;
  ge::AttrUtils::GetListInt(conv_node->GetOpDesc(), "pads", pads);
  FUSION_PASS_CHECK(pads.size() != 4,
                    OP_LOGI(conv_node->GetType().c_str(),
                            "Get node[%s]'s pads attr not success.",
                            conv_node->GetName().c_str()),
                    return false);
  int64_t pad_up = pads[0];
  int64_t pad_down = pads[1];
  int64_t pad_left = pads[2];
  int64_t pad_right = pads[3];
  bool pad_flg = (pad_up == 0 && pad_down == 0 && pad_left == 0 && pad_right == 0);
  ge::DataType second_data_type = conv_node->GetOpDesc()->GetInputDesc(0).GetDataType();
  bool load2d_flg = filter_flg && stride_flg && pad_flg && second_data_type == ge::DT_FLOAT16;
  FUSION_PASS_CHECK(load2d_flg == true, OP_LOGI(conv_node->GetType().c_str(),
                                                "node[%s]'s is the load2d case"
                                                "can not fusion.",
                                                conv_node->GetName().c_str()),
                    return false);
  return true;
}
/***************************************************************
if the minimal l1 buffer is exceed the L1 Buffer Size,
the aipp can not fusion with the conv.
***************************************************************/
bool TbeAippFusionRule::CheckAippConvEltwiseFusionValidation(const ge::NodePtr conv_node, const string& input_format) {
  ge::Format first_format = conv_node->GetOpDesc()->GetInputDesc(0).GetOriginFormat();
  vector<int64_t> first_dims;
  first_dims = conv_node->GetOpDesc()->GetInputDesc(0).GetOriginShape().GetDims();
  FUSION_PASS_CHECK(first_dims.size() != 4,
                    OP_LOGI(conv_node->GetType().c_str(),
                            "Get node[%s]'s first input shape not success.",
                            conv_node->GetName().c_str()),
                    return false);
  vector<int64_t> dilations;
  ge::AttrUtils::GetListInt(conv_node->GetOpDesc(), "dilations", dilations);
  FUSION_PASS_CHECK(dilations.size() != 4,
                    OP_LOGI(conv_node->GetType().c_str(),
                            "Get node[%s]'s dilations attr not success.",
                            conv_node->GetName().c_str()),
                    return false);
  vector<int64_t> strides;
  ge::AttrUtils::GetListInt(conv_node->GetOpDesc(), "strides", strides);
  FUSION_PASS_CHECK(strides.size() != 4,
                    OP_LOGI(conv_node->GetType().c_str(),
                            "Get node[%s]'s strides attr not success.",
                            conv_node->GetName().c_str()),
                    return false);
  int64_t width_fm = 0;
  int64_t dilate_h = 0;
  int64_t dilate_w = 0;
  int64_t stride_h = 0;
  int64_t stride_w = 1;
  if (first_format == ge::FORMAT_NCHW) {
    width_fm = first_dims[3];
    dilate_h = dilations[2];
    dilate_w = dilations[3];
    stride_h = strides[2];
    stride_w = strides[3];
  } else if (first_format == ge::FORMAT_NHWC) {
    width_fm = first_dims[2];
    dilate_h = dilations[1];
    dilate_w = dilations[2];
    stride_h = strides[1];
    stride_w = strides[2];
  }
  ge::Format second_format = conv_node->GetOpDesc()->GetInputDesc(1).GetOriginFormat();
  std::vector<int64_t> second_dims;
  second_dims = conv_node->GetOpDesc()->GetInputDesc(1).GetOriginShape().GetDims();
  FUSION_PASS_CHECK(second_dims.size() != 4,
                    OP_LOGI(conv_node->GetType().c_str(),
                            "Get node[%s]'s second input shape not success.",
                            conv_node->GetName().c_str()),
                    return false);
  int64_t height_ft = 0;
  int64_t width_ft = 0;
  if (second_format == ge::FORMAT_NCHW) {
    height_ft = second_dims[2];
    width_ft = second_dims[3];
  } else if (second_format == ge::FORMAT_NHWC) {
    height_ft = second_dims[1];
    width_ft = second_dims[2];
  } else if (second_format == ge::FORMAT_HWCN) {
    height_ft = second_dims[0];
    width_ft = second_dims[1];
  }
  vector<int64_t> pads;
  ge::AttrUtils::GetListInt(conv_node->GetOpDesc(), "pads", pads);
  FUSION_PASS_CHECK(pads.size() != 4,
                    OP_LOGI(conv_node->GetType().c_str(),
                            "Get node[%s]'s pads attr not success.",
                            conv_node->GetName().c_str()),
                    return false);
  int64_t pad_left = pads[2];
  int64_t pad_right = pads[3];
  int64_t wk_dilation = (width_ft - 1) * dilate_w + 1;
  int64_t hk_dilation = (height_ft - 1) * dilate_h + 1;
  FUSION_PASS_CHECK(stride_w == 0, OP_LOGI(conv_node->GetType().c_str(), "Stride width is zero."), return false);
  int64_t width_out = floor((width_fm - wk_dilation + pad_left + pad_right) / stride_w) + 1;
  FUSION_PASS_CHECK(width_out == 0, OP_LOGI(conv_node->GetType().c_str(), "Out width is zero."), return false);
  int64_t width_in = floor(16 / width_out) + 2;
  int64_t tmp = ((width_in - 1) * stride_h + hk_dilation) * width_fm;
  if (input_format == "YUV420SP_U8") {
    tmp = tmp + 2*width_fm;
  }
  int64_t m_bit_ratio = 2;
  int64_t ci0 = 16;
  int64_t max_feature_map_l1 = ci0 * tmp * m_bit_ratio;
  int64_t max_l1 = 0;
  PlatformInfo platform_info;
  OptionalInfo opti_compilation_info;
  FUSION_PASS_CHECK(PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(
                        platform_info, opti_compilation_info) != fe::SUCCESS,
                    OP_LOGI(conv_node->GetType().c_str(), "Get platform_info failed."), return false);
  max_l1 = platform_info.ai_core_spec.l1_size;
  FUSION_PASS_CHECK(max_feature_map_l1 > max_l1, OP_LOGI(conv_node->GetType().c_str(),
                                                     "node[%s]'s minimal l1 buffer is exceed the L1 Buffer Size"
                                                     "can not fusion.",
                                                     conv_node->GetName().c_str()),
                    return false);
  return true;
}

}  // namespace fe
