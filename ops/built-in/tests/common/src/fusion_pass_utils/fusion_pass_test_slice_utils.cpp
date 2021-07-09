#include "fusion_pass_test_slice_utils.h"

namespace fe {

    void SetSplitMapMainNode(std::vector<AxisSplitMap>& split_maps, std::vector<ge::NodePtr>& Nodes, const string& op_type) {
        OpCalcInfo op_calc_info;
        op_calc_info.Initialize();
        string op_slice_info_str = "";
        op_calc_info.SetAxisSplitMaps(split_maps);
        SetOpSliceInfoToJson(op_calc_info, op_slice_info_str);
        for (auto node : Nodes) {
          if (ge::AttrUtils::SetStr(node->GetOpDesc(), fe::OP_SLICE_INFO, op_slice_info_str)) {
              OP_LOGW(op_type.c_str(), "set OP_SLICE_INFO Succeed");
          };
        }
        OP_LOGD(op_type.c_str(), "set _op_slice_info is %s", op_slice_info_str.c_str());
    }

}