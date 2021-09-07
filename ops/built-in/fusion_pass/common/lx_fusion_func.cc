/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file lx_fusion_func.cpp
 * \brief
 */
#include "lx_fusion_func.h"

namespace fe {
void DelSplitInfoByOutputAxis(std::vector<AxisSplitMap>& split_maps, int axis) {
  std::vector<AxisSplitMap> temp_maps;
  for(auto it = split_maps.begin(); it != split_maps.end(); ++it) {
    bool del_axis = false;
    auto output_split_infos = (*it).GetOutputSplitInfoVec();
    for (auto output_split_info : output_split_infos) {
      if (!output_split_info.GetAxis().empty()) {
        if (output_split_info.GetAxis()[0] == axis) {
          del_axis = true;
        }
      }
    }
    if (!del_axis) {
      temp_maps.push_back(*it);
    }
  }
  split_maps = temp_maps;
}

void DelSplitInfoByInputAxis(std::vector<AxisSplitMap>& split_maps, int axis) {
  std::vector<AxisSplitMap> temp_maps;
  for(auto it = split_maps.begin(); it != split_maps.end(); ++it) {
    bool del_axis = false;
    auto input_split_infos = (*it).GetInputSplitInfoVec();
    for (auto input_split_info : input_split_infos) {
      if (!input_split_info.GetAxis().empty()) {
        if (input_split_info.GetAxis()[0] == axis) {
          del_axis = true;
        }
      }
    }
    if (!del_axis) {
      temp_maps.push_back(*it);
    }
  }
  split_maps = temp_maps;
}

bool GetSplitMap(std::vector<AxisSplitMap>& split_maps, ge::NodePtr& cube_node,
                 const string& fused_op_type, OpL1FusionType& L1_fusion_type, int64_t& min_tbe_L1space) {
  string op_slice_info_str = "";
  if (cube_node->GetOpDesc() == nullptr) {
    OP_LOGD(fused_op_type.c_str(), "get desc failed");
    return false;
  }
  ge::AttrUtils::GetStr(cube_node->GetOpDesc(), fe::OP_SLICE_INFO, op_slice_info_str);
  OP_LOGD(fused_op_type.c_str(), "ori _op_slice_info is %s", op_slice_info_str.c_str());
  if (op_slice_info_str.empty()) {
    OP_LOGD(fused_op_type.c_str(), "op_slice_info is null");
    return false;
  }

  OpCalcInfo op_calc_info;
  GetOpSliceInfoFromJson(op_calc_info, op_slice_info_str);
  split_maps = op_calc_info.GetAxisSplitMapVec();
  if (split_maps.empty()) {
    OP_LOGD(fused_op_type.c_str(), "axis split map vector is empty");
    return false;
  }
  L1_fusion_type = op_calc_info.GetL1FusionEnable();
  min_tbe_L1space = op_calc_info.GetMinTbeL1Space();
  return true;
}

void SetSplitMap(std::vector<AxisSplitMap>& split_maps, std::vector<ge::NodePtr>& fusionNodes,
                 const string& fused_op_type, const OpL1FusionType& L1_fusion_type, const int64_t& min_tbe_L1space) {
  OpCalcInfo op_calc_info;
  op_calc_info.Initialize();
  string op_slice_info_str = "";
  op_calc_info.SetAxisSplitMaps(split_maps);
  op_calc_info.SetL1FusionEnable(L1_fusion_type);
  op_calc_info.SetMinTbeL1Space(min_tbe_L1space);
  SetFusionOpSliceInfoToJson(op_calc_info, op_slice_info_str);
  for (auto fusion_node : fusionNodes) {
    if (fusion_node == nullptr) {
      continue;
    }
    ge::AttrUtils::SetStr(fusion_node->GetOpDesc(), fe::FUSION_OP_SLICE_INFO, op_slice_info_str);
  }
  OP_LOGD(fused_op_type.c_str(), "set _fusion_op_slice_info is %s", op_slice_info_str.c_str());
}

void AddElemwiseSplitMap(std::vector<AxisSplitMap>& split_maps, ge::NodePtr& elemWiseNode, int& index) {
  for (uint i = 1; i < elemWiseNode->GetOpDesc()->GetAllInputsDesc().size(); i++) {
    index += 1;
    vector<int64_t> split_flag = {-1};
    for(auto it = split_maps.begin(); it != split_maps.end(); ++it) {
      auto output_split_infos = (*it).GetOutputSplitInfoVec();
      auto input_split_infos = (*it).GetInputSplitInfoVec();
      if (output_split_infos.empty() || input_split_infos.empty()) {
        continue;
      }
      vector<int64_t> out_axis_dim = output_split_infos[0].GetAxis();
      InputSplitInfo input_split_info;
      input_split_info.Initialize();
      input_split_info.SetIndex(index);
      input_split_info.SetAxis(out_axis_dim);
      input_split_info.SetHeadOverLap(split_flag);
      input_split_info.SetTailOverLap(split_flag);
      (*it).AddInputSplitInfo(input_split_info);
    }
  }
}

}  // namespace fe
