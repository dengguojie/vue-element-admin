#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#define private public
#define private public
#include "register/op_tiling_registry.h"

#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"

#include "elewise_calculation_ops.h"
#include "nonlinear_fuc_ops.h"
#include "array_ops.h"
#include "op_tiling/vector_tiling.h"
#include "op_tiling/broadcast_v3.h"
#include "op_tiling/tiling_handler.h"

using namespace std;
using namespace ge;
using namespace optiling;

class BroadcastTilingV3 : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "BroadcastTilingV3 SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "BroadcastTilingV3 TearDown" << std::endl;
    }
};

static string to_string(const std::stringstream &tiling_data) {
    auto data = tiling_data.str();
    string result;
    int32_t tmp = 0;
    for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
        memcpy_s(&tmp, sizeof(&tmp), data.c_str() + i, sizeof(tmp));
        result += std::to_string(tmp);
        result += " ";
    }

    return result;
}

static void contruct_tensor(ge::OpDescPtr& op_desc, const std::vector<int64_t>& shape, const ge::DataType dtype,
                            bool is_input=true, ge::Format format=ge::FORMAT_ND) {
  ge::GeTensorDesc tensor;
  tensor.SetShape(ge::GeShape(shape));
  tensor.SetFormat(format);
  tensor.SetDataType(dtype);
  if (is_input) {
    op_desc->AddInputDesc(tensor);
  } else {
    op_desc->AddOutputDesc(tensor);
  }
}

template<typename T1, typename T2>
static bool CompareMap(const std::unordered_map<T1, T2>& map1, const std::unordered_map<T1, T2>& map2) {
  if (map1.size() != map2.size()) {
    std::cout << "map size wrong!" << std::endl;
    return false;
  }
  for (const auto& it : map1) {
    if (map2.count(it.first) == 0) {
      std::cout << "map key not match" << std::endl;
      return false;
    }
    if (map1.at(it.first) != map2.at(it.first)) {
      std::cout << "map value at key: " << it.first << " is not equal" << std::endl;
      return false;
    }
  }
  return true;
}

TEST_F(BroadcastTilingV3, TilingTest1) {
  std::vector<std::vector<int64_t>> inputs {{64,}, {64,},{1,}};
  std::vector<std::vector<int64_t>> outputs {{64,}};;
  ge::DataType dtype = ge::DT_FLOAT;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }
  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;

  v3::BroadcastCompileInfo actual_ptr;

  actual_ptr.base_info_compile.first = true;
  actual_ptr.base_info_compile.second = {{"100", {32,2,43680,21840}}, {"200", {32, 2, 28656, 14320}}};
  actual_ptr.flag_info_compile = {false, false, true, true, false, false, false};
  actual_ptr.ub_factor_align = 1;
  actual_ptr.elewise_vars_compile.first = true;
  actual_ptr.elewise_vars_compile.second = {{ "210000000", { 10000, 20000, 30000 }}, {"210010000", {10000, 20000, 30000 }}, {"220000000", { 10000, 10001, 10002, 20000, 30000 } }};
  actual_ptr.const_block_dims_compile.first = false;
  actual_ptr.const_block_dims_compile.second = {};

  actual_ptr.const_shapes_compile.first = false;
  actual_ptr.const_shapes_compile.second = {};

  actual_ptr.fusion_index_compile.first = true;
  actual_ptr.fusion_index_compile.second = {{0}};

  actual_ptr.broadcast_axis_compile.first = false;
  actual_ptr.broadcast_axis_compile.second = {};

  actual_ptr.soc_version.first = false;
  actual_ptr.soc_version.second = "";

  v3::Broadcast broadcast("autotiling", op_paras, actual_ptr, runInfo);
  ASSERT_TRUE(broadcast.BroadcastTiling());
}

TEST_F(BroadcastTilingV3, TilingTest2) {
  std::vector<std::vector<int64_t>> inputs {{1, 5824}, {100, 1}};
  std::vector<std::vector<int64_t>> outputs {{100, 5824}};;
  ge::DataType dtype = ge::DT_FLOAT;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }
  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;

  v3::BroadcastCompileInfo actual_ptr;

  actual_ptr.base_info_compile.first = true;
  actual_ptr.base_info_compile.second = {{"320", {32, 4, 21840, 10920}}, {"000", {32, 4, 21840, 10920}}};
  actual_ptr.flag_info_compile = {false, false, true, true, false, false, false};
  actual_ptr.ub_factor_align = 128;
  actual_ptr.elewise_vars_compile.first = true;
  actual_ptr.elewise_vars_compile.second = {{ "232000000", {10001, 20000, 30000}}, {"0", {10100}}, {"1", {10100, 20000, 30000}}, {"2", {10100, 20000, 30001}}, {"4", {10100, 20001, 30001}}};
  actual_ptr.const_block_dims_compile.first = false;
  actual_ptr.const_block_dims_compile.second = {};

  actual_ptr.const_shapes_compile.first = false;
  actual_ptr.const_shapes_compile.second = {};

  actual_ptr.fusion_index_compile.first = true;
  actual_ptr.fusion_index_compile.second = {{0},{1}};

  actual_ptr.broadcast_axis_compile.first = false;
  actual_ptr.broadcast_axis_compile.second = {};

  actual_ptr.soc_version.first = false;
  actual_ptr.soc_version.second = "";

  v3::Broadcast broadcast("autotiling", op_paras, actual_ptr, runInfo);
  ASSERT_TRUE(broadcast.BroadcastTiling());
  EXPECT_EQ(runInfo.GetBlockDim(), 25);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "5824 2 2 ");
}

TEST_F(BroadcastTilingV3, TilingTest3) {
  std::vector<std::vector<int64_t>> inputs {{1, 33, 1}, {1, 33, 1089}};
  std::vector<std::vector<int64_t>> outputs {{1, 33, 1089}};;
  ge::DataType dtype = ge::DT_FLOAT;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }
  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;

  v3::BroadcastCompileInfo actual_ptr;
  actual_ptr.base_info_compile.first = true;
  actual_ptr.base_info_compile.second = {{"110", {32, 4, 32768, 16384}}};
  actual_ptr.flag_info_compile = {false, false, true, true, false, false, false};
  actual_ptr.ub_factor_align = 128;
  actual_ptr.elewise_vars_compile.first = true;
  actual_ptr.elewise_vars_compile.second = {{ "210000000", {10000, 20000, 30000}}};
  actual_ptr.const_block_dims_compile.first = false;
  actual_ptr.const_block_dims_compile.second = {};

  actual_ptr.const_shapes_compile.first = false;
  actual_ptr.const_shapes_compile.second = {};

  actual_ptr.fusion_index_compile.first = false;
  actual_ptr.fusion_index_compile.second = {};

  actual_ptr.broadcast_axis_compile.first = false;
  actual_ptr.broadcast_axis_compile.second = {};

  actual_ptr.soc_version.first = false;
  actual_ptr.soc_version.second = "";


  v3::Broadcast broadcast("autotiling", op_paras, actual_ptr, runInfo);
  ASSERT_TRUE(!broadcast.BroadcastTiling());
}

TEST_F(BroadcastTilingV3, TilingTest4) {
  std::vector<std::vector<int64_t>> inputs {{1, 33, 1}, {1, 33, 1089}};
  std::vector<std::vector<int64_t>> outputs {{1, 33, 1089}};;
  ge::DataType dtype = ge::DT_FLOAT;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }
  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;

  v3::BroadcastCompileInfo actual_ptr;
  actual_ptr.base_info_compile.first = true;
  actual_ptr.base_info_compile.second = {{"120", {32, 4, 32768, 16384}}};
  actual_ptr.flag_info_compile = {false, false, true, true, false, false, false};
  actual_ptr.ub_factor_align = 128;
  actual_ptr.elewise_vars_compile.first = true;
  actual_ptr.elewise_vars_compile.second = {{ "210000000", {10000, 20000, 30000}}};
  actual_ptr.const_block_dims_compile.first = false;
  actual_ptr.const_block_dims_compile.second = {};

  actual_ptr.const_shapes_compile.first = false;
  actual_ptr.const_shapes_compile.second = {};

  actual_ptr.fusion_index_compile.first = true;
  actual_ptr.fusion_index_compile.second = {{1}};

  actual_ptr.broadcast_axis_compile.first = false;
  actual_ptr.broadcast_axis_compile.second = {};

  actual_ptr.soc_version.first = false;
  actual_ptr.soc_version.second = "";


  v3::Broadcast broadcast("autotiling", op_paras, actual_ptr, runInfo);
  ASSERT_TRUE(!broadcast.BroadcastTiling());
}

TEST_F(BroadcastTilingV3, TilingTest5) {
  std::vector<std::vector<int64_t>> inputs {{35, 45, 223}, {45, 223}};
  std::vector<std::vector<int64_t>> outputs {{35, 45, 223}};;
  ge::DataType dtype = ge::DT_FLOAT16;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }
  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;

  v3::BroadcastCompileInfo actual_ptr;
  actual_ptr.base_info_compile.first = true;
  actual_ptr.base_info_compile.second = {{"210", {32, 4, 21832, 10912}}};
  actual_ptr.flag_info_compile = {false, false, true, true, false, false, false};
  actual_ptr.ub_factor_align = 128;
  actual_ptr.elewise_vars_compile.first = true;
  actual_ptr.elewise_vars_compile.second = {{ "221000001", {10000, 10001, 10100, 20000, 30000}}, { "221000002", {10000, 10001, 10100, 20000, 30001}}};
  actual_ptr.const_block_dims_compile.first = false;
  actual_ptr.const_block_dims_compile.second = {};

  actual_ptr.const_shapes_compile.first = false;
  actual_ptr.const_shapes_compile.second = {};

  actual_ptr.fusion_index_compile.first = false;
  actual_ptr.fusion_index_compile.second = {};

  actual_ptr.broadcast_axis_compile.first = false;
  actual_ptr.broadcast_axis_compile.second = {};

  actual_ptr.soc_version.first = false;
  actual_ptr.soc_version.second = "";


  v3::Broadcast broadcast("autotiling", op_paras, actual_ptr, runInfo);
ASSERT_TRUE(broadcast.BroadcastTiling());
}

TEST_F(BroadcastTilingV3, TilingTest6) {
  std::vector<std::vector<int64_t>> inputs {{28, 1, 35, 45, 223}, {28, 5, 35, 1, 1}};
  std::vector<std::vector<int64_t>> outputs {{28, 5, 35, 45, 223}};;
  ge::DataType dtype = ge::DT_FLOAT;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }
  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;

  v3::BroadcastCompileInfo actual_ptr;
  actual_ptr.base_info_compile.first = true;
  actual_ptr.base_info_compile.second = {{"100", {32, 2, 43680, 21840}}, {"120", {32, 2, 28656, 14320}}, {"121", {32, 2, 30704, 15344}}, {"210", {32, 2, 30704, 15344}}, {"320", {32, 2, 42320, 21152}}, {"230", {32, 2, 42320, 21152}}, {"000", {32, 2, 30704, 15344}}, {"999", {32, 2, 30704, 15344}}};
  actual_ptr.flag_info_compile = {false, false, true, true, true, false, true};
  actual_ptr.ub_factor_align = 128;
  actual_ptr.elewise_vars_compile.first = true;
  actual_ptr.elewise_vars_compile.second = {{"210000000", {10000, 20000, 30000}}, {"210010000", {10000, 20000, 30000}}, {"212000000", {10000, 10100, 10101}}, {"212000001", {10000, 10100, 10101, 20000, 30000}}, {"212000002", {10000, 10100, 10101, 20000, 30001}}, {"212010002", {10000, 10100, 10101, 20000, 30001}}, {"212000004", {10000, 10100, 10101, 20001, 30001}}, {"212010004", {10000, 10100, 10101, 20001, 30001}}, {"212100000", {10000, 10100, 10101, 10200}}, {"212100001", {10000, 10100, 10101, 10200, 20000, 30000}}, {"212100002", {10000, 10100, 10101, 10200, 20000, 30001}}, {"212100003", {10000, 10100, 10101, 10200, 20000, 30002}}, {"212100005", {10000, 10100, 10101, 10200, 20001, 30001}}, {"212100006", {10000, 10100, 10101, 10200, 20001, 30002}}, {"212100009", {10000, 10100, 10101, 10200, 20002, 30002}}, {"221000000", {10000, 10001, 10100}}, {"221000001", {10000, 10001, 10100, 20000, 30000}}, {"221000002", {10000, 10001, 10100, 20000, 30001}}, {"221000004", {10000, 10001, 10100, 20001, 30001}}, {"232000000", {10001, 20000, 30000}}, {"223000000", {10000, 20000, 30000}}, {"0", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401}}, {"1", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30000}}, {"2", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30001}}, {"3", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30002}}, {"4", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30003}}, {"5", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30004}}, {"7", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30001}}, {"8", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30002}}, {"9", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30003}}, {"10", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30004}}, {"13", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20002, 30002}}, {"14", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20002, 30003}}, {"15", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20002, 30004}}, {"19", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20003, 30003}}, {"20", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20003, 30004}}, {"25", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20004, 30004}}, {"299900000", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301}}, {"299900001", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30000}}, {"299900002", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30001}}, {"299900003", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30002}}, {"299900004", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30003}}, {"299900006", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20001, 30001}}, {"299900007", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20001, 30002}}, {"299900008", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20001, 30003}}, {"299900011", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20002, 30002}}, {"299900012", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20002, 30003}}, {"299900016", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20003, 30003}}};
  actual_ptr.const_block_dims_compile.first = false;
  actual_ptr.const_block_dims_compile.second = {};

  actual_ptr.const_shapes_compile.first = false;
  actual_ptr.const_shapes_compile.second = {};

  actual_ptr.fusion_index_compile.first = true;
  actual_ptr.fusion_index_compile.second = {{0}, {1}, {2}, {3}, {4}};

  actual_ptr.broadcast_axis_compile.first = false;
  actual_ptr.broadcast_axis_compile.second = {};

  actual_ptr.soc_version.first = false;
  actual_ptr.soc_version.second = "";


  v3::Broadcast broadcast("autotiling", op_paras, actual_ptr, runInfo);
ASSERT_TRUE(broadcast.BroadcastTiling());
}

TEST_F(BroadcastTilingV3, TilingTest7) {
  std::vector<std::vector<int64_t>> inputs {{1, 1, 1, 112, 22}, {32, 5, 25, 1, 22}};
  std::vector<std::vector<int64_t>> outputs {{32, 5, 25, 112, 22}};;
  ge::DataType dtype = ge::DT_FLOAT16;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }
  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;

  v3::BroadcastCompileInfo actual_ptr;
  actual_ptr.base_info_compile.first = true;
  actual_ptr.base_info_compile.second = {{"100", {32, 2, 43680, 21840}}, {"120", {32, 2, 28656, 14320}}, {"121", {32, 2, 30704, 15344}}, {"210", {32, 2, 30704, 15344}}, {"320", {32, 2, 42320, 21152}}, {"230", {32, 2, 42320, 21152}}, {"000", {32, 2, 30704, 15344}}, {"999", {32, 2, 30704, 15344}},{"100", {32, 2, 43680, 21840}}, {"120", {32, 2, 28656, 14320}}, {"121", {32, 2, 30704, 15344}}, {"210", {32, 2, 30704, 15344}}, {"320", {32, 2, 42320, 21152}}, {"230", {32, 2, 42320, 21152}}, {"000", {32, 2, 30704, 15344}}, {"999", {32, 2, 30704, 15344}}};
  actual_ptr.flag_info_compile = {false, false, true, true, true, false, true};
  actual_ptr.ub_factor_align = 128;
  actual_ptr.elewise_vars_compile.first = true;
  actual_ptr.elewise_vars_compile.second = {{"210000000", {10000, 20000, 30000}}, {"210010000", {10000, 20000, 30000}}, {"212000000", {10000, 10100, 10101}}, {"212000001", {10000, 10100, 10101, 20000, 30000}}, {"212000002", {10000, 10100, 10101, 20000, 30001}}, {"212010002", {10000, 10100, 10101, 20000, 30001}}, {"212000004", {10000, 10100, 10101, 20001, 30001}}, {"212010004", {10000, 10100, 10101, 20001, 30001}}, {"212100000", {10000, 10100, 10101, 10200}}, {"212100001", {10000, 10100, 10101, 10200, 20000, 30000}}, {"212100002", {10000, 10100, 10101, 10200, 20000, 30001}}, {"212100003", {10000, 10100, 10101, 10200, 20000, 30002}}, {"212100005", {10000, 10100, 10101, 10200, 20001, 30001}}, {"212100006", {10000, 10100, 10101, 10200, 20001, 30002}}, {"212100009", {10000, 10100, 10101, 10200, 20002, 30002}}, {"221000000", {10000, 10001, 10100}}, {"221000001", {10000, 10001, 10100, 20000, 30000}}, {"221000002", {10000, 10001, 10100, 20000, 30001}}, {"221000004", {10000, 10001, 10100, 20001, 30001}}, {"232000000", {10001, 20000, 30000}}, {"223000000", {10000, 20000, 30000}}, {"0", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401}}, {"1", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30000}}, {"2", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30001}}, {"3", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30002}}, {"4", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30003}}, {"5", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30004}}, {"7", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30001}}, {"8", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30002}}, {"9", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30003}}, {"10", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30004}}, {"13", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20002, 30002}}, {"14", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20002, 30003}}, {"15", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20002, 30004}}, {"19", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20003, 30003}}, {"20", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20003, 30004}}, {"25", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20004, 30004}}, {"299900000", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301}}, {"299900001", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30000}}, {"299900002", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30001}}, {"299900003", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30002}}, {"299900004", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30003}}, {"299900006", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20001, 30001}}, {"299900007", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20001, 30002}}, {"299900008", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20001, 30003}}, {"299900011", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20002, 30002}}, {"299900012", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20002, 30003}}, {"299900016", {10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20003, 30003}}};
  actual_ptr.const_block_dims_compile.first = false;
  actual_ptr.const_block_dims_compile.second = {};

  actual_ptr.const_shapes_compile.first = false;
  actual_ptr.const_shapes_compile.second = {};

  actual_ptr.fusion_index_compile.first = true;
  actual_ptr.fusion_index_compile.second = {{0}, {1}, {2}, {3}, {4}};

  actual_ptr.broadcast_axis_compile.first = false;
  actual_ptr.broadcast_axis_compile.second = {};

  actual_ptr.soc_version.first = false;
  actual_ptr.soc_version.second = "";


  v3::Broadcast broadcast("autotiling", op_paras, actual_ptr, runInfo);
  ASSERT_TRUE(broadcast.BroadcastTiling());
}

TEST_F(BroadcastTilingV3, TilingTest8) {
  std::vector<std::vector<int64_t>> inputs {{1, 1, 5824}, {32, 100, 1}};
  std::vector<std::vector<int64_t>> outputs {{32, 100, 5824}};;
  ge::DataType dtype = ge::DT_FLOAT;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }
  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;

  v3::BroadcastCompileInfo actual_ptr;
  actual_ptr.base_info_compile.first = true;
  actual_ptr.base_info_compile.second = {{"100", {48, 2, 32768, 16384}}, {"120", {48, 2, 24576, 12288}}, {"121", {48, 2, 19648, 9824}}, {"210", {48, 2, 19648, 9824}}, {"320", {48, 2, 32752, 16368}}, {"230", {48, 2, 32752, 16368}}, {"000", {48, 2, 19648, 9824}}, {"999", {48, 2, 19648, 9824}}};
  actual_ptr.flag_info_compile = {false, false, true, true, true, false, true};
  actual_ptr.ub_factor_align = 128;
  actual_ptr.elewise_vars_compile.first = true;
  actual_ptr.elewise_vars_compile.second = {{"210000000", {10000, 20000, 30000}}, {"210010000", {10000, 20000, 30000}}, {"212000000", {10000, 10100, 10101}}, {"212000001", {10000, 10100, 10101, 20000, 30000}}, {"212000002", {10000, 10100, 10101, 20000, 30001}}, {"212010002", {10000, 10100, 10101, 20000, 30001}}, {"212000004", {10000, 10100, 10101, 20001, 30001}}, {"212010004", {10000, 10100, 10101, 20001, 30001}}, {"212100000", {10000, 10100, 10101, 10200}}, {"212100001", {10000, 10100, 10101, 10200, 20000, 30000}}, {"212100002", {10000, 10100, 10101, 10200, 20000, 30001}}, {"212100003", {10000, 10100, 10101, 10200, 20000, 30002}}, {"212100005", {10000, 10100, 10101, 10200, 20001, 30001}}, {"212100006", {10000, 10100, 10101, 10200, 20001, 30002}}, {"212100009", {10000, 10100, 10101, 10200, 20002, 30002}}, {"221000000", {10000, 10001, 10100}}, {"221000001", {10000, 10001, 10100, 20000, 30000}}, {"221000002", {10000, 10001, 10100, 20000, 30001}}, {"221000004", {10000, 10001, 10100, 20001, 30001}}, {"232000000", {10001, 20000, 30000}}, {"223000000", {10000, 20000, 30000}}, {"0", {10000, 10001, 10100, 10101, 10200, 10201}}, {"1", {10000, 10001, 10100, 10101, 10200, 10201, 20000, 30000}}, {"2", {10000, 10001, 10100, 10101, 10200, 10201, 20000, 30001}}, {"3", {10000, 10001, 10100, 10101, 10200, 10201, 20000, 30002}}, {"5", {10000, 10001, 10100, 10101, 10200, 10201, 20001, 30001}}, {"6", {10000, 10001, 10100, 10101, 10200, 10201, 20001, 30002}}, {"9", {10000, 10001, 10100, 10101, 10200, 10201, 20002, 30002}}, {"299900000", {10000, 10001, 10100, 10101}}, {"299900001", {10000, 10001, 10100, 10101, 20000, 30000}}, {"299900002", {10000, 10001, 10100, 10101, 20000, 30001}}, {"299900004", {10000, 10001, 10100, 10101, 20001, 30001}}};
  actual_ptr.const_block_dims_compile.first = false;
  actual_ptr.const_block_dims_compile.second = {};

  actual_ptr.const_shapes_compile.first = false;
  actual_ptr.const_shapes_compile.second = {};

  actual_ptr.fusion_index_compile.first = true;
  actual_ptr.fusion_index_compile.second = {{0}, {1}, {2}};

  actual_ptr.broadcast_axis_compile.first = false;
  actual_ptr.broadcast_axis_compile.second = {};

  actual_ptr.soc_version.first = true;
  actual_ptr.soc_version.second = "Ascend920";


  v3::Broadcast broadcast("autotiling", op_paras, actual_ptr, runInfo);
  ASSERT_TRUE(broadcast.BroadcastTiling());
}

TEST_F(BroadcastTilingV3, TilingTest9) {
  std::vector<std::vector<int64_t>> inputs {{1, 1, 5824}, {32, 100, 1}, {32, 100, 1}};
  std::vector<std::vector<int64_t>> outputs {{32, 100, 5824}};;
  ge::DataType dtype = ge::DT_FLOAT;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }
  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;

  v3::BroadcastCompileInfo actual_ptr;
  actual_ptr.base_info_compile.first = true;
  actual_ptr.base_info_compile.second = {{"100", {48, 2, 32768, 16384}}, {"120", {48, 2, 16384, 8192}}, {"121", {48, 2, 16384, 8192}}, {"210", {48, 2, 16384, 8192}}, {"200", {48, 2, 24576, 12288}}, {"000", {48, 2, 16384, 8192}}, {"999", {48, 2, 16384, 8192}}};
  actual_ptr.flag_info_compile = {false, false, true, true, true, false, true};
  actual_ptr.ub_factor_align = 128;
  actual_ptr.elewise_vars_compile.first = true;
  actual_ptr.elewise_vars_compile.second = {{"210000000", {10000, 20000, 30000}}, {"210010000", {10000, 20000, 30000}}, {"212000000", {10000, 10100, 10101}}, {"212000001", {10000, 10100, 10101, 20000, 30000}}, {"212000002", {10000, 10100, 10101, 20000, 30001}}, {"212010002", {10000, 10100, 10101, 20000, 30001}}, {"212000004", {10000, 10100, 10101, 20001, 30001}}, {"212010004", {10000, 10100, 10101, 20001, 30001}}, {"212100000", {10000, 10100, 10101, 10200}}, {"212100001", {10000, 10100, 10101, 10200, 20000, 30000}}, {"212100002", {10000, 10100, 10101, 10200, 20000, 30001}}, {"212100003", {10000, 10100, 10101, 10200, 20000, 30002}}, {"212100005", {10000, 10100, 10101, 10200, 20001, 30001}}, {"212100006", {10000, 10100, 10101, 10200, 20001, 30002}}, {"212100009", {10000, 10100, 10101, 10200, 20002, 30002}}, {"221000000", {10000, 10001, 10100}}, {"221000001", {10000, 10001, 10100, 20000, 30000}}, {"221000002", {10000, 10001, 10100, 20000, 30001}}, {"221000004", {10000, 10001, 10100, 20001, 30001}}, {"220000000", {10000, 10001, 20000, 30000}}, {"0", {10000, 10001, 10100, 10101, 10200, 10201}}, {"1", {10000, 10001, 10100, 10101, 10200, 10201, 20000, 30000}}, {"2", {10000, 10001, 10100, 10101, 10200, 10201, 20000, 30001}}, {"3", {10000, 10001, 10100, 10101, 10200, 10201, 20000, 30002}}, {"5", {10000, 10001, 10100, 10101, 10200, 10201, 20001, 30001}}, {"6", {10000, 10001, 10100, 10101, 10200, 10201, 20001, 30002}}, {"9", {10000, 10001, 10100, 10101, 10200, 10201, 20002, 30002}}, {"299900000", {10000, 10001, 10100, 10101}}, {"299900001", {10000, 10001, 10100, 10101, 20000, 30000}}, {"299900002", {10000, 10001, 10100, 10101, 20000, 30001}}, {"299900004", {10000, 10001, 10100, 10101, 20001, 30001}}};
  actual_ptr.const_block_dims_compile.first = false;
  actual_ptr.const_block_dims_compile.second = {};

  actual_ptr.const_shapes_compile.first = false;
  actual_ptr.const_shapes_compile.second = {};

  actual_ptr.fusion_index_compile.first = true;
  actual_ptr.fusion_index_compile.second = {{0}, {1}, {2}};

  actual_ptr.broadcast_axis_compile.first = false;
  actual_ptr.broadcast_axis_compile.second = {};

  actual_ptr.soc_version.first = true;
  actual_ptr.soc_version.second = "Ascend920";


  v3::Broadcast broadcast("autotiling", op_paras, actual_ptr, runInfo);
  ASSERT_TRUE(broadcast.BroadcastTiling());

}

TEST_F(BroadcastTilingV3, TilingTest10) {
  std::vector<std::vector<int64_t>> inputs {{2, 0, 2}, {2, 0, 2}};
  std::vector<std::vector<int64_t>> outputs {{2, 0, 2}};;
  ge::DataType dtype = ge::DT_FLOAT;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }
  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;

  v3::BroadcastCompileInfo actual_ptr;
  actual_ptr.base_info_compile.first = true;
  actual_ptr.base_info_compile.second = {{"100", {32, 4, 21840, 10920}}};
  actual_ptr.flag_info_compile = {false, false, true, true, true, false, true};
  actual_ptr.ub_factor_align = 128;
  actual_ptr.elewise_vars_compile.first = true;
  actual_ptr.elewise_vars_compile.second = {{"210000000", {20000, 30000}}, {"210010000", {20000, 30000}}};
  actual_ptr.const_block_dims_compile.first = false;
  actual_ptr.const_block_dims_compile.second = {};

  actual_ptr.const_shapes_compile.first = false;
  actual_ptr.const_shapes_compile.second = {};

  actual_ptr.fusion_index_compile.first = true;
  actual_ptr.fusion_index_compile.second = {{0}, {1}, {2}};

  actual_ptr.broadcast_axis_compile.first = false;
  actual_ptr.broadcast_axis_compile.second = {};

  actual_ptr.soc_version.first = false;
  actual_ptr.soc_version.second = "";


  v3::Broadcast broadcast("autotiling", op_paras, actual_ptr, runInfo);
  ASSERT_TRUE(broadcast.BroadcastTiling());
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "");
}

TEST_F(BroadcastTilingV3, TilingTest11) {
  std::vector<std::vector<int64_t>> inputs {{10, 40000, 1}, {10, 40000, 10}};
  std::vector<std::vector<int64_t>> outputs {{10, 40000, 10}};
  ge::DataType dtype = ge::DT_FLOAT16;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }
  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;

  v3::BroadcastCompileInfo actual_ptr;
  actual_ptr.base_info_compile.first = true;
  actual_ptr.base_info_compile.second = {{"000", {48, 2, 32768, 16384}}};
  actual_ptr.flag_info_compile = {true};
  actual_ptr.ub_factor_align = 128;
  actual_ptr.elewise_vars_compile.first = false;
  actual_ptr.elewise_vars_compile.second = {};
  actual_ptr.const_block_dims_compile.first = false;
  actual_ptr.const_block_dims_compile.second = {};

  actual_ptr.const_shapes_compile.first = false;
  actual_ptr.const_shapes_compile.second = {};

  actual_ptr.fusion_index_compile.first = true;
  actual_ptr.fusion_index_compile.second = {{0,1}, {2}};

  actual_ptr.broadcast_axis_compile.first = true;
  actual_ptr.broadcast_axis_compile.second = {false, true};

  actual_ptr.soc_version.first = true;
  actual_ptr.soc_version.second = "Ascend920";


  v3::Broadcast broadcast("autotiling", op_paras, actual_ptr, runInfo);
  ASSERT_TRUE(broadcast.BroadcastTiling());
}

TEST_F(BroadcastTilingV3, TilingTest12) {
  std::vector<std::vector<int64_t>> inputs {{1, 40000, 0}, {10, 40000, 0}};
  std::vector<std::vector<int64_t>> outputs {{10, 40000, 0}};
  ge::DataType dtype = ge::DT_FLOAT16;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }
  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;

  v3::BroadcastCompileInfo actual_ptr;
  actual_ptr.base_info_compile.first = true;
  actual_ptr.base_info_compile.second = {{"000", {48, 2, 32768, 16384}}};
  actual_ptr.flag_info_compile = {true};
  actual_ptr.ub_factor_align = 128;
  actual_ptr.elewise_vars_compile.first = false;
  actual_ptr.elewise_vars_compile.second = {};
  actual_ptr.const_block_dims_compile.first = false;
  actual_ptr.const_block_dims_compile.second = {};

  actual_ptr.const_shapes_compile.first = false;
  actual_ptr.const_shapes_compile.second = {};

  actual_ptr.fusion_index_compile.first = true;
  actual_ptr.fusion_index_compile.second = {{0},{1,2}};

  actual_ptr.broadcast_axis_compile.first = true;
  actual_ptr.broadcast_axis_compile.second = {false, true};

  actual_ptr.soc_version.first = true;
  actual_ptr.soc_version.second = "Ascend920";


  v3::Broadcast broadcast("autotiling", op_paras, actual_ptr, runInfo);
  ASSERT_TRUE(broadcast.BroadcastTiling());
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "");
}

TEST_F(BroadcastTilingV3, TilingTest13) {
  std::vector<std::vector<int64_t>> inputs {{1, 33, 1089}, {1, 33, 1089}};
  std::vector<std::vector<int64_t>> outputs {{1, 33, 1089}};
  ge::DataType dtype = ge::DT_FLOAT;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }
  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;


  std::string compileInfo = R"({ "_ub_factor_align": 128, "_contains_elewise_sch": true, "_pattern": "Broadcast", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 4, 32768, 16384]}, "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateBroadcastTilingHandler(this->test_info_->name(),
                              "autotiling",
                              nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
}

TEST_F(BroadcastTilingV3, TilingTest14) {
  std::vector<std::vector<int64_t>> inputs {{1, 40000, 10}, {10, 40000, 10}};
  std::vector<std::vector<int64_t>> outputs {{10, 40000, 10}};
  ge::DataType dtype = ge::DT_FLOAT16;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }
  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;

  v3::BroadcastCompileInfo actual_ptr;
  actual_ptr.base_info_compile.first = false;
  actual_ptr.base_info_compile.second = {};
  actual_ptr.flag_info_compile = {false, true, true, false, false, false, false};
  actual_ptr.ub_factor_align = 128;
  actual_ptr.elewise_vars_compile.first = true;
  actual_ptr.elewise_vars_compile.second = {{"100000000",{}}};
  actual_ptr.const_block_dims_compile.first = true;
  actual_ptr.const_block_dims_compile.second = {40};

  actual_ptr.const_shapes_compile.first = true;
  actual_ptr.const_shapes_compile.second = {{1, 40000, 10}};

  actual_ptr.fusion_index_compile.first = true;
  actual_ptr.fusion_index_compile.second = {{0},{1,2}};

  actual_ptr.broadcast_axis_compile.first = true;
  actual_ptr.broadcast_axis_compile.second = {false, true};

  actual_ptr.soc_version.first = true;
  actual_ptr.soc_version.second = "Ascend920";


  v3::Broadcast broadcast("autotiling", op_paras, actual_ptr, runInfo);
  ASSERT_TRUE(broadcast.BroadcastTiling());
}

TEST_F(BroadcastTilingV3, TilingTest15) {
  std::vector<std::vector<int64_t>> inputs {{1, 1, 5824}, {32, 100, 1}, {32, 100, 1}};
  std::vector<std::vector<int64_t>> outputs {{32, 100, 5824}};;
  ge::DataType dtype = ge::DT_FLOAT;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }
  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;

  v3::BroadcastCompileInfo actual_ptr;
  actual_ptr.base_info_compile.first = true;
  actual_ptr.base_info_compile.second = {{"100", {48, 2, 32768, 16384}}, {"120", {48, 2, 16384, 8192}}, {"121", {48, 2, 16384, 8192}}, {"210", {48, 2, 16384, 8192}}, {"200", {48, 2, 24576, 12288}}, {"000", {48, 2, 16384, 8192}}, {"999", {48, 2, 16384, 8192}}};
  actual_ptr.flag_info_compile = {false, false, true, true, true, false, true};
  actual_ptr.ub_factor_align = 128;
  actual_ptr.elewise_vars_compile.first = true;
  actual_ptr.elewise_vars_compile.second = {{"210000000", {10000, 20000, 30000}}, {"210010000", {10000, 20000, 30000}}, {"212000000", {10000, 10100, 10101}}, {"212000001", {10000, 10100, 10101, 20000, 30000}}, {"212000002", {10000, 10100, 10101, 20000, 30001}}, {"212010002", {10000, 10100, 10101, 20000, 30001}}, {"212000004", {10000, 10100, 10101, 20001, 30001}}, {"212010004", {10000, 10100, 10101, 20001, 30001}}, {"212100000", {10000, 10100, 10101, 10200}}, {"212100001", {10000, 10100, 10101, 10200, 20000, 30000}}, {"212100002", {10000, 10100, 10101, 10200, 20000, 30001}}, {"212100003", {10000, 10100, 10101, 10200, 20000, 30002}}, {"212100005", {10000, 10100, 10101, 10200, 20001, 30001}}, {"212100006", {10000, 10100, 10101, 10200, 20001, 30002}}, {"212100009", {10000, 10100, 10101, 10200, 20002, 30002}}, {"221000000", {10000, 10001, 10100}}, {"221000001", {10000, 10001, 10100, 20000, 30000}}, {"221000002", {10000, 10001, 10100, 20000, 30001}}, {"221000004", {10000, 10001, 10100, 20001, 30001}}, {"220000000", {10000, 10001, 20000, 30000}}, {"0", {10000, 10001, 10100, 10101, 10200, 10201}}, {"1", {10000, 10001, 10100, 10101, 10200, 10201, 20000, 30000}}, {"2", {10000, 10001, 10100, 10101, 10200, 10201, 20000, 30001}}, {"3", {10000, 10001, 10100, 10101, 10200, 10201, 20000, 30002}}, {"5", {10000, 10001, 10100, 10101, 10200, 10201, 20001, 30001}}, {"6", {10000, 10001, 10100, 10101, 10200, 10201, 20001, 30002}}, {"9", {10000, 10001, 10100, 10101, 10200, 10201, 20002, 30002}}, {"299900000", {10000, 10001, 10100, 10101}}, {"299900001", {10000, 10001, 10100, 10101, 20000, 30000}}, {"299900002", {10000, 10001, 10100, 10101, 20000, 30001}}, {"299900004", {10000, 10001, 10100, 10101, 20001, 30001}}};
  actual_ptr.const_block_dims_compile.first = false;
  actual_ptr.const_block_dims_compile.second = {};

  actual_ptr.const_shapes_compile.first = false;
  actual_ptr.const_shapes_compile.second = {};

  actual_ptr.fusion_index_compile.first = true;
  actual_ptr.fusion_index_compile.second = {{0}, {1}, {2}};

  actual_ptr.broadcast_axis_compile.first = false;
  actual_ptr.broadcast_axis_compile.second = {};

  actual_ptr.soc_version.first = true;
  actual_ptr.soc_version.second = "Ascend920";


  optiling::OpInfo custom_op_info(inputs, ge::DT_FLOAT);
  const std::pair<bool, optiling::OpInfo> op_custom(true, custom_op_info);

  v3::Broadcast broadcast("autotiling", op_paras, actual_ptr, runInfo);
  ASSERT_TRUE(broadcast.BroadcastTiling(custom_op_info));
}

TEST_F(BroadcastTilingV3, TilingTest16) {
  std::vector<std::vector<int64_t>> inputs {{1, 1, 5824}, {32, 100, 1}, {32, 100, 1}};
  std::vector<std::vector<int64_t>> outputs {{32, 100, 5824}};
  ge::DataType dtype = ge::DT_FLOAT;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }
  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;


  std::string compileInfo = R"({"_fusion_index": [[0], [1], [2]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_soc_version": "Ascend920", "_flag_info": [false, false, true, true, true, false, true], "_base_info": {"100": [48, 2, 32768, 16384], "120": [48, 2, 24576, 12288], "121": [48, 2, 19648, 9824], "210": [48, 2, 19648, 9824], "320": [48, 2, 32752, 16368], "230": [48, 2, 32752, 16368], "000": [48, 2, 19648, 9824], "999": [48, 2, 19648, 9824]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212000004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212010004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "212100005": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "212100006": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "212100009": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_2", "_ub_factor_2"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"]}, "_normal_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212000004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212010004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "212100005": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "212100006": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "212100009": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_2", "_ub_factor_2"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"]}, "_custom_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "212000004": [], "212010004": [], "212100000": [], "212100001": [], "212100002": [], "212100003": [], "212100005": [], "212100006": [], "212100009": [], "221000000": [], "221000001": [], "221000002": [], "221000004": [], "232000000": [], "223000000": [], "0": [], "1": [], "2": [], "3": [], "5": [], "6": [], "9": [], "299900000": [], "299900001": [], "299900002": [], "299900004": []}, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "212000000": [10000, 10100, 10101], "212000001": [10000, 10100, 10101, 20000, 30000], "212000002": [10000, 10100, 10101, 20000, 30001], "212010002": [10000, 10100, 10101, 20000, 30001], "212000004": [10000, 10100, 10101, 20001, 30001], "212010004": [10000, 10100, 10101, 20001, 30001], "212100000": [10000, 10100, 10101, 10200], "212100001": [10000, 10100, 10101, 10200, 20000, 30000], "212100002": [10000, 10100, 10101, 10200, 20000, 30001], "212100003": [10000, 10100, 10101, 10200, 20000, 30002], "212100005": [10000, 10100, 10101, 10200, 20001, 30001], "212100006": [10000, 10100, 10101, 10200, 20001, 30002], "212100009": [10000, 10100, 10101, 10200, 20002, 30002], "221000000": [10000, 10001, 10100], "221000001": [10000, 10001, 10100, 20000, 30000], "221000002": [10000, 10001, 10100, 20000, 30001], "221000004": [10000, 10001, 10100, 20001, 30001], "232000000": [10001, 20000, 30000], "223000000": [10000, 20000, 30000], "0": [10000, 10001, 10100, 10101, 10200, 10201], "1": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30000], "2": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30001], "3": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30002], "5": [10000, 10001, 10100, 10101, 10200, 10201, 20001, 30001], "6": [10000, 10001, 10100, 10101, 10200, 10201, 20001, 30002], "9": [10000, 10001, 10100, 10101, 10200, 10201, 20002, 30002], "299900000": [10000, 10001, 10100, 10101], "299900001": [10000, 10001, 10100, 10101, 20000, 30000], "299900002": [10000, 10001, 10100, 10101, 20000, 30001], "299900004": [10000, 10001, 10100, 10101, 20001, 30001]}})";
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateBroadcastTilingHandler(this->test_info_->name(),
                              "autotiling",
                              nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
}

TEST_F(BroadcastTilingV3, TilingTest17) {
  std::vector<std::vector<int64_t>> inputs {{1, 40000, 10}, {10, 40000, 10}};
  std::vector<std::vector<int64_t>> outputs {{10, 40000, 10}};
  ge::DataType dtype = ge::DT_FLOAT16;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }
  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;

  v3::BroadcastCompileInfo actual_ptr;
  actual_ptr.base_info_compile.first = false;
  actual_ptr.base_info_compile.second = {};
  actual_ptr.flag_info_compile = {false, true, true, false, false, false, false};
  actual_ptr.ub_factor_align = 128;
  actual_ptr.elewise_vars_compile.first = true;
  actual_ptr.elewise_vars_compile.second = {{"100000000",{}}};
  actual_ptr.const_block_dims_compile.first = true;
  actual_ptr.const_block_dims_compile.second = {40};

  actual_ptr.const_shapes_compile.first = true;
  actual_ptr.const_shapes_compile.second = {{1, 40000, 10}};

  actual_ptr.fusion_index_compile.first = true;
  actual_ptr.fusion_index_compile.second = {{0},{1,2}};

  actual_ptr.broadcast_axis_compile.first = true;
  actual_ptr.broadcast_axis_compile.second = {false, true};

  actual_ptr.soc_version.first = true;
  actual_ptr.soc_version.second = "Ascend920";


  optiling::OpInfo custom_op_info(inputs, ge::DT_FLOAT);
  const std::pair<bool, optiling::OpInfo> op_custom(true, custom_op_info);
  v3::Broadcast broadcast("autotiling", op_paras, actual_ptr, runInfo);
  ASSERT_TRUE(broadcast.BroadcastTiling(custom_op_info));
}

TEST_F(BroadcastTilingV3, TilingTest18) {
  std::vector<std::vector<int64_t>> inputs {{700,1}, {700,2000}, {700,2000}};
  std::vector<std::vector<int64_t>> outputs {{700,2000},{700,2000}};
  ge::DataType dtype = ge::DT_FLOAT16;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }
  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;


  std::string compileInfo = R"({"_fusion_index": [[0], [1]], "_pattern": "Broadcast", "_ub_factor_align": 128,  "_flag_info": [false, false, true, true, false, false, true], "_base_info": {"000": [32, 4, 13104, 6552], "100": [32, 4, 13104, 6552], "120": [32, 4, 13096, 6544], "200": [32, 4, 13104, 6552], "210": [32, 4, 13104, 6552]}, "_elewise_vars": {"0": [10000,10001,10002, 10100,10101,10102],"1": [10000,10001,10002, 10100,10101,10102, 20000, 30000], "2": [10000,10001,10002, 10100,10101,10102, 20000, 30001], "4": [10000,10001,10002, 10100,10101,10102, 20001, 30001], "210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "212000000": [10000, 10100, 10101, 10102],"212000001": [10000, 10100, 10101, 10102, 20000, 30000],"212000002": [10000, 10100, 10101, 10102, 20000, 30001],"212000004": [10000, 10100, 10101, 10102, 20001, 30001],"212010002": [10000, 10100, 10101, 2000, 30001],"212010004": [10000, 10100, 10101, 20001, 30001], "220000000": [ 10000, 10001, 10002, 20000, 30000 ], "221000000": [10000, 10001, 10100],"221000001": [10000, 10001, 10100, 20000, 30000],"221000002": [10000, 10001, 10100, 20000, 30001], "221000004": [10000, 10001, 10100, 20001, 30001]}})";
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateBroadcastTilingHandler(this->test_info_->name(),
                              "autotiling",
                              nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
}

TEST_F(BroadcastTilingV3, TilingTest19) {
  std::vector<std::vector<int64_t>> inputs {{1,1}, {1,}};
  std::vector<std::vector<int64_t>> outputs {{1,1}, };
  ge::DataType dtype = ge::DT_FLOAT;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }
  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;


  std::string compileInfo = R"({"_ub_factor_align": 128,  "_pattern": "Broadcast", "_fusion_index": [[0,1]], "push_status": 0, "_flag_info": [false, false, true, true, true, false, false], "_base_info": {"320": [32, 4, 21840, 10920], "100": [32, 4, 21840, 10920]}, "_elewise_vars": { "232000000": [10001, 20000, 30000],  "210000000": [20000, 30000], "210010000": [10001, 20000, 30000] } })";
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateBroadcastTilingHandler(this->test_info_->name(),
                              "autotiling",
                              nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
}

TEST_F(BroadcastTilingV3, TilingTest20) {
  std::vector<std::vector<int64_t>> inputs {{2, 26, 26, 3, 80}, {1}};
  std::vector<std::vector<int64_t>> outputs {{2, 26, 26, 3, 80}};
  ge::DataType dtype = ge::DT_FLOAT16;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }
  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  optiling::utils::OpRunInfo runInfo;


  std::string compileInfo = R"({"_ub_factor_align": 128,  "_pattern": "Broadcast", "_fusion_index": [[0,1,2,3,4]], "push_status": 0, "_flag_info": [false, false, true, true, true, false, false], "_base_info": {"230": [32, 2, 42320, 21152]}, "_elewise_vars": { "223000000": [10000, 20000, 30000] } })";
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateBroadcastTilingHandler(this->test_info_->name(),
                              "autotiling",
                              nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
}

TEST_F(BroadcastTilingV3, TilingTest22) {
  std::vector<std::vector<int64_t>> inputs {{2, 26, 26, 3, 80}, {1}};
  std::vector<std::vector<int64_t>> outputs {{2, 26, 26, 3, 80}};
  ge::DataType dtype = ge::DT_FLOAT16;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }
  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  op_paras.SetAttr("alpha", 123);
  optiling::utils::OpRunInfo runInfo;


  std::string compileInfo = R"({"_ub_factor_align": 128,  "_pattern": "Broadcast", "_fusion_index": [[0,1,2,3,4]], "push_status": 0, "_flag_info": [false, false, true, true, true, false, false], "_base_info": {"230": [32, 2, 42320, 21152]}, "_elewise_vars": { "223000000": [10000, 20000, 30000] },  "_attr_vars": { "223000000": [{"length":1, "name":"alpha", "type":"int32", "src_type":"int32"}] } })";
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateBroadcastTilingHandler(this->test_info_->name(),
                              "autotiling",
                              nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(op_paras, runInfo));
}

TEST_F(BroadcastTilingV3, TilingTest23) {
  std::vector<std::vector<int64_t>> inputs {{1, 40000, 10}, {10, 40000, 10}};
  std::vector<std::vector<int64_t>> outputs {{10, 40000, 10}};
  ge::DataType dtype = ge::DT_FLOAT16;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }
  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  op_paras.SetAttr("alpha", 123);
  optiling::utils::OpRunInfo runInfo;

  v3::BroadcastCompileInfo actual_ptr;
  actual_ptr.base_info_compile.first = false;
  actual_ptr.base_info_compile.second = {};
  actual_ptr.flag_info_compile = {false, true, true, false, false, false, false};
  actual_ptr.ub_factor_align = 128;
  actual_ptr.elewise_vars_compile.first = true;
  actual_ptr.elewise_vars_compile.second = {{"100000000",{}}};
  actual_ptr.const_block_dims_compile.first = true;
  actual_ptr.const_block_dims_compile.second = {40};

  std::string var_attr_list_compileInfo = R"({"_var_attr_mode":0,"_var_attrs": [{"length":1,"name":"alpha","type":"int32","src_type":"int32"}]})";
  actual_ptr.varAttrWrap.ParseVarAttr(nlohmann::json::parse(var_attr_list_compileInfo));

  actual_ptr.const_shapes_compile.first = true;
  actual_ptr.const_shapes_compile.second = {{1, 40000, 10}};

  actual_ptr.fusion_index_compile.first = true;
  actual_ptr.fusion_index_compile.second = {{0},{1,2}};

  actual_ptr.broadcast_axis_compile.first = true;
  actual_ptr.broadcast_axis_compile.second = {false, true};

  actual_ptr.soc_version.first = true;
  actual_ptr.soc_version.second = "Ascend920";


  v3::Broadcast broadcast("autotiling", op_paras, actual_ptr, runInfo);
  ASSERT_TRUE(broadcast.BroadcastTiling());
}

