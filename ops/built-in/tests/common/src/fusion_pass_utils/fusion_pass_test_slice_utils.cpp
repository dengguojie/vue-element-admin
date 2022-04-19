#include "fusion_pass_test_slice_utils.h"

#include "inc/common/aicore_util_attr_define.h"

namespace fe {
string CreateOpSliceInfoStrFromSplitMap(const std::vector<fe::AxisSplitMap> &vec_split_map) {
  OpCalcInfo op_calc_info;
  op_calc_info.Initialize();
  op_calc_info.SetAxisSplitMaps(const_cast<std::vector<fe::AxisSplitMap> &>(vec_split_map));

  string op_slice_info_str = "";
  SetOpSliceInfoToJson(op_calc_info, op_slice_info_str);
  return op_slice_info_str;
}

string CreateFusionOpSliceInfoStrFromSplitMap(const std::vector<fe::AxisSplitMap> &vec_split_map) {
  OpCalcInfo op_calc_info;
  op_calc_info.Initialize();
  op_calc_info.SetAxisSplitMaps(const_cast<std::vector<fe::AxisSplitMap> &>(vec_split_map));

  string fusion_op_slice_info_str = "";
  SetFusionOpSliceInfoToJson(op_calc_info, fusion_op_slice_info_str);
  return fusion_op_slice_info_str;
}

void SetSplitMapMainNode(std::vector<AxisSplitMap> &split_maps, std::vector<ge::NodePtr> &Nodes,
                         const string &op_type) {
  string op_slice_info_str = CreateOpSliceInfoStrFromSplitMap(split_maps);

  for (auto node : Nodes) {
    if (ge::AttrUtils::SetStr(node->GetOpDesc(), fe::OP_SLICE_INFO, op_slice_info_str)) {
      OP_LOGW(op_type.c_str(), "set OP_SLICE_INFO Succeed");
    };
  }
  OP_LOGD(op_type.c_str(), "set _op_slice_info is %s", op_slice_info_str.c_str());
}

bool SetSplitMapToNodeByType(const ge::ComputeGraphPtr compute_graph_ptr, std::vector<AxisSplitMap> &vec_split_map,
                             const std::vector<string> &type_ops) {
  string op_slice_info_str = CreateOpSliceInfoStrFromSplitMap(vec_split_map);

  auto ptr_nodes = compute_graph_ptr->GetAllNodes();
  for (const auto ptr_node : ptr_nodes) {
    auto op_desc = ptr_node->GetOpDesc();
    if (find(type_ops.cbegin(), type_ops.cend(), op_desc->GetType()) != type_ops.cend()) {
      if (!ge::AttrUtils::SetStr(op_desc, fe::OP_SLICE_INFO, op_slice_info_str)) {
        return false;
      }
    }
  }

  return true;
}

bool SetSplitMapToNodeByName(const ge::ComputeGraphPtr compute_graph_ptr, std::vector<AxisSplitMap> &vec_split_map,
                             const string &name_op) {
  string op_slice_info_str = CreateOpSliceInfoStrFromSplitMap(vec_split_map);

  auto ptr_nodes = compute_graph_ptr->GetAllNodes();
  for (const auto ptr_node : ptr_nodes) {
    auto op_desc = ptr_node->GetOpDesc();
    if (name_op == op_desc->GetName() and !ge::AttrUtils::SetStr(op_desc, fe::OP_SLICE_INFO, op_slice_info_str)) {
      return false;
    }
  }

  return true;
}

string GetFusionOpSliceInfoStrFromGraph(const ge::ComputeGraphPtr compute_graph_ptr) {
  string fusion_op_slice_info_str = "";

  auto ptr_nodes = compute_graph_ptr->GetAllNodes();
  for (const auto ptr_node : ptr_nodes) {
    auto op_desc = ptr_node->GetOpDesc();
    if (ge::AttrUtils::GetStr(op_desc, fe::FUSION_OP_SLICE_INFO, fusion_op_slice_info_str)) {
      return fusion_op_slice_info_str;
    }
  }

  return fusion_op_slice_info_str;
}

InputSplitInfo CreateInputSplitInfo(const size_t &idx, const std::vector<int64_t> &axis,
                                    const std::vector<int64_t> &head_over_lap,
                                    const std::vector<int64_t> &tail_over_lap) {
  InputSplitInfo isi;
  isi.Initialize();
  isi.SetIndex(idx);
  isi.SetAxis(const_cast<std::vector<int64_t> &>(axis));
  isi.SetHeadOverLap(const_cast<std::vector<int64_t> &>(head_over_lap));
  isi.SetTailOverLap(const_cast<std::vector<int64_t> &>(tail_over_lap));
  return isi;
}

OutputSplitInfo CreateOutputSplitInfo(const size_t &idx, const std::vector<int64_t> &axis) {
  OutputSplitInfo osi;
  osi.Initialize();
  osi.SetIndex(idx);
  osi.SetAxis(const_cast<std::vector<int64_t> &>(axis));
  return osi;
}

AxisSplitMap CreateAxisSplitMap(const std::vector<InputSplitInfo> &vec_input_split_info,
                                const std::vector<OutputSplitInfo> &vec_output_split_info) {
  AxisSplitMap as;
  as.Initialize();
  for (const auto &si : vec_input_split_info) {
    as.AddInputSplitInfo(const_cast<InputSplitInfo &>(si));
  }
  for (const auto &si : vec_output_split_info) {
    as.AddOutputSplitInfo(const_cast<OutputSplitInfo &>(si));
  }
  return as;
}
}  // namespace fe