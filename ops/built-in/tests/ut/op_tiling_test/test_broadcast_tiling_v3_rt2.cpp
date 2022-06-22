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

//#include "op_tiling/vector_tiling_rt2.h"
#include "common_autotiling_util.h"
#include "test_common.h"
#include "common/utils/ut_op_util.h"

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

TEST_F(BroadcastTilingV3, broadcast_rt_case1) {
    std::string compileInfo = R"({"_fusion_index": [[0], [1], [2]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_soc_version": "Ascend910B", "_flag_info": [false, false, true, true, true, false, true], "_base_info": {"100": [48, 2, 32768, 16384, 2], "120": [48, 2, 24576, 12288, 2], "121": [48, 2, 19648, 9824, 2], "210": [48, 2, 19648, 9824, 2], "320": [48, 2, 32752, 16368, 2], "230": [48, 2, 32752, 16368, 2], "000": [48, 2, 19648, 9824, 2], "999": [48, 2, 19648, 9824, 2]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212000004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212010004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "212100005": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "212100006": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "212100009": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_2", "_ub_factor_2"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"]}, "_normal_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212000004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212010004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "212100005": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "212100006": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "212100009": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_2", "_ub_factor_2"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"]}, "_custom_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "212000004": [], "212010004": [], "212100000": [], "212100001": [], "212100002": [], "212100003": [], "212100005": [], "212100006": [], "212100009": [], "221000000": [], "221000001": [], "221000002": [], "221000004": [], "232000000": [], "223000000": [], "0": [], "1": [], "2": [], "3": [], "5": [], "6": [], "9": [], "299900000": [], "299900001": [], "299900002": [], "299900004": []}, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "212000000": [10000, 10100, 10101], "212000001": [10000, 10100, 10101, 20000, 30000], "212000002": [10000, 10100, 10101, 20000, 30001], "212010002": [10000, 10100, 10101, 20000, 30001], "212000004": [10000, 10100, 10101, 20001, 30001], "212010004": [10000, 10100, 10101, 20001, 30001], "212100000": [10000, 10100, 10101, 10200], "212100001": [10000, 10100, 10101, 10200, 20000, 30000], "212100002": [10000, 10100, 10101, 10200, 20000, 30001], "212100003": [10000, 10100, 10101, 10200, 20000, 30002], "212100005": [10000, 10100, 10101, 10200, 20001, 30001], "212100006": [10000, 10100, 10101, 10200, 20001, 30002], "212100009": [10000, 10100, 10101, 10200, 20002, 30002], "221000000": [10000, 10001, 10100], "221000001": [10000, 10001, 10100, 20000, 30000], "221000002": [10000, 10001, 10100, 20000, 30001], "221000004": [10000, 10001, 10100, 20001, 30001], "232000000": [10001, 20000, 30000], "223000000": [10000, 20000, 30000], "0": [10000, 10001, 10100, 10101, 10200, 10201], "1": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30000], "2": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30001], "3": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30002], "5": [10000, 10001, 10100, 10101, 10200, 10201, 20001, 30001], "6": [10000, 10001, 10100, 10101, 10200, 10201, 20001, 30002], "9": [10000, 10001, 10100, 10101, 10200, 10201, 20002, 30002], "299900000": [10000, 10001, 10100, 10101], "299900001": [10000, 10001, 10100, 10101, 20000, 30000], "299900002": [10000, 10001, 10100, 10101, 20000, 30001], "299900004": [10000, 10001, 10100, 10101, 20001, 30001]}})";
    std::vector<std::vector<int64_t>> inputs{{1, 1, 5824}, {32, 100, 1}, {32, 100, 1}};
    std::vector<std::vector<int64_t>> outputs{{32, 100, 5824}};
    ge::DataType dtype = ge::DT_FLOAT;
    AutoTilingTest test(inputs, outputs, dtype, dtype);
    optiling::v3::BroadcastCompileInfo broadcast_info;
    test.SetCompileInfo(compileInfo, &broadcast_info);
    EXPECT_EQ(test.Test(), true);
}


TEST_F(BroadcastTilingV3, broadcast_rt_case2) {
    std::string compileInfo = R"({"_fusion_index": [[0]], "_pattern": "Broadcast", "_ub_factor_align": 1, "_soc_version": "Ascend910", "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100":[32, 2, 43680, 21840, 2], "200":[32, 2, 28656, 14320, 2]},  "_elewise_vars": { "210000000":[10000, 20000, 30000 ], "210010000":[10000, 20000, 30000 ], "220000000":[10000, 10001, 10002, 20000, 30000]}})";
    std::vector<std::vector<int64_t>> inputs {{64,}, {64,},{1,}};
    std::vector<std::vector<int64_t>> outputs {{64,}};
    ge::DataType dtype = ge::DT_FLOAT;
    AutoTilingTest test(inputs, outputs, dtype, dtype);
    optiling::v3::BroadcastCompileInfo broadcast_info;
    test.SetCompileInfo(compileInfo, &broadcast_info);
    EXPECT_EQ(test.Test(), true);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case3) {
  std::string compileInfo = R"({"_fusion_index": [[0],[1]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_soc_version": "Ascend910", "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"320":[32, 4, 21840, 10920, 4], "000":[32, 4, 21840, 10920, 4]},  "_elewise_vars": { "232000000":[10001, 20000, 30000], "0":[10100], "1":[10100, 20000, 30000], "2":[10100, 20000, 30001], "4": [10100, 20001, 30001]}})";
  std::vector<std::vector<int64_t>> inputs {{1, 5824}, {100, 1}};
  std::vector<std::vector<int64_t>> outputs {{100, 5824}};;
  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;
  test.SetCompileInfo(compileInfo, &broadcast_info);
  EXPECT_EQ(test.Test(), true);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case4) {
  std::string compileInfo = R"({"_fusion_index": [[0],[1]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"110":[32, 4, 32768, 16384, 4]},  "_elewise_vars": { "232000000":[10001, 20000, 30000], "0":[10100], "1":[10100, 20000, 30000], "2":[10100, 20000, 30001], "4": [10100, 20001, 30001]}})";
  std::vector<std::vector<int64_t>> inputs {{1, 33, 1}, {1, 33, 1089}};
  std::vector<std::vector<int64_t>> outputs {{1, 33, 1089}};;
  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;
  test.SetCompileInfo(compileInfo, &broadcast_info);
  EXPECT_EQ(test.Test(), false);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case5) {
  std::string compileInfo = R"({"_fusion_index": [[0],[1]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"120":[32, 4, 32768, 16384, 4]},  "_elewise_vars": { "232000000":[10001, 20000, 30000], "0":[10100], "1":[10100, 20000, 30000], "2":[10100, 20000, 30001], "4": [10100, 20001, 30001]}})";
  std::vector<std::vector<int64_t>> inputs {{1, 33, 1}, {1, 33, 1089}};
  std::vector<std::vector<int64_t>> outputs {{1, 33, 1089}};;
  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;
  test.SetCompileInfo(compileInfo, &broadcast_info);
  EXPECT_EQ(test.Test(), false);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case6) {
  std::string compileInfo = R"({"_fusion_index": [[0], [1], [2]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_brc_avoid_bank_conflict": true, "_is_vnchwconv_align": true, "_classify_inputs_num": 2, "_contains_elewise_sch": true, "_soc_version": "Ascend910", "_all_unknown_last_const": false, "_is_pure_brc": false, "_has_store_align": true, "_flag_info": [false, false, true, true, true, false, true], "_base_info": {"100": [32, 4, 21840, 10920, 1], "120": [32, 4, 21840, 10920, 4], "121": [32, 4, 21840, 10920, 4], "210": [32, 4, 21840, 10920, 4], "320": [32, 4, 21832, 10912, 1], "230": [32, 4, 21832, 10912, 1], "000": [32, 4, 21840, 10920, 4], "888": [32, 4, 16384, 8192, 4], "999": [32, 4, 21840, 10920, 4]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "232010000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "223010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "488800001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "488800002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "488800003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"]}, "_normal_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "232010000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "223010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "488800001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "488800002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "488800003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"]}, "_custom_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "212100000": [], "212100001": [], "212100002": [], "212100003": [], "221000000": [], "221000001": [], "221000002": [], "232000000": [], "232010000": [], "223000000": [], "223010000": [], "0": [], "1": [], "2": [], "3": [], "488800001": [], "488800002": [], "488800003": [], "299900000": [], "299900001": [], "299900002": []}, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "212000000": [10000, 10100, 10101], "212000001": [10000, 10100, 10101, 20000, 30000], "212000002": [10000, 10100, 10101, 20000, 30001], "212010002": [10000, 10100, 10101, 20000, 30001], "212100000": [10000, 10100, 10101, 10200], "212100001": [10000, 10100, 10101, 10200, 20000, 30000], "212100002": [10000, 10100, 10101, 10200, 20000, 30001], "212100003": [10000, 10100, 10101, 10200, 20000, 30002], "221000000": [10000, 10001, 10100], "221000001": [10000, 10001, 10100, 20000, 30000], "221000002": [10000, 10001, 10100, 20000, 30001], "232000000": [10001, 20000, 30000], "232010000": [10001, 20000, 30000], "223000000": [10000, 20000, 30000], "223010000": [10000, 20000, 30000], "0": [10000, 10001, 10100, 10101, 10200, 10201], "1": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30000], "2": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30001], "3": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30002], "488800001": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30000], "488800002": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30001], "488800003": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30002], "299900000": [10000, 10001, 10100, 10101], "299900001": [10000, 10001, 10100, 10101, 20000, 30000], "299900002": [10000, 10001, 10100, 10101, 20000, 30001]}})";
  std::vector<std::vector<int64_t>> inputs {{35, 45, 223}, {45, 223}};
  std::vector<std::vector<int64_t>> outputs {{35, 45, 223}};
  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;
  test.SetCompileInfo(compileInfo, &broadcast_info);
  EXPECT_EQ(test.Test(), true);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case7) {
  std::string compileInfo = R"({"_fusion_index": [[0], [1], [2], [3], [4]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_brc_avoid_bank_conflict": true, "_is_vnchwconv_align": true, "_classify_inputs_num": 2, "_contains_elewise_sch": true, "_soc_version": "Ascend910", "_all_unknown_last_const": false, "_is_pure_brc": false, "_has_store_align": true, "_flag_info": [false, false, true, true, true, false, true], "_base_info": {"100": [32, 4, 21840, 10920, 1], "120": [32, 4, 21840, 10920, 4], "121": [32, 4, 21840, 10920, 4], "210": [32, 4, 21840, 10920, 4], "320": [32, 4, 21832, 10912, 1], "230": [32, 4, 21832, 10912, 1], "000": [32, 4, 21840, 10920, 4], "888": [32, 4, 16384, 8192, 4], "999": [32, 4, 21840, 10920, 4]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "232010000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "223010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_2"], "4": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_3"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_4"], "488800001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_0"], "488800002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_1"], "488800003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_2"], "488800004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_3"], "488800005": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_4"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_1"], "299900003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_2"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_3"]}, "_normal_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "232010000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "223010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_2"], "4": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_3"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_4"], "488800001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_0"], "488800002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_1"], "488800003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_2"], "488800004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_3"], "488800005": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_4"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_1"], "299900003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_2"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_3"]}, "_custom_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "212100000": [], "212100001": [], "212100002": [], "212100003": [], "221000000": [], "221000001": [], "221000002": [], "232000000": [], "232010000": [], "223000000": [], "223010000": [], "0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "488800001": [], "488800002": [], "488800003": [], "488800004": [], "488800005": [], "299900000": [], "299900001": [], "299900002": [], "299900003": [], "299900004": []}, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "212000000": [10000, 10100, 10101], "212000001": [10000, 10100, 10101, 20000, 30000], "212000002": [10000, 10100, 10101, 20000, 30001], "212010002": [10000, 10100, 10101, 20000, 30001], "212100000": [10000, 10100, 10101, 10200], "212100001": [10000, 10100, 10101, 10200, 20000, 30000], "212100002": [10000, 10100, 10101, 10200, 20000, 30001], "212100003": [10000, 10100, 10101, 10200, 20000, 30002], "221000000": [10000, 10001, 10100], "221000001": [10000, 10001, 10100, 20000, 30000], "221000002": [10000, 10001, 10100, 20000, 30001], "232000000": [10001, 20000, 30000], "232010000": [10001, 20000, 30000], "223000000": [10000, 20000, 30000], "223010000": [10000, 20000, 30000], "0": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401], "1": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30000], "2": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30001], "3": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30002], "4": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30003], "5": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30004], "488800001": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30000], "488800002": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30001], "488800003": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30002], "488800004": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30003], "488800005": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30004], "299900000": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301], "299900001": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30000], "299900002": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30001], "299900003": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30002], "299900004": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30003]}})";
  std::vector<std::vector<int64_t>> inputs {{28, 1, 35, 45, 223}, {28, 5, 35, 1, 1}};
  std::vector<std::vector<int64_t>> outputs {{28, 5, 35, 45, 223}};
  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;
  test.SetCompileInfo(compileInfo, &broadcast_info);
  EXPECT_EQ(test.Test(), true);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case8) {
  std::string compileInfo = R"({"_fusion_index": [[0], [1], [2], [3], [4]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_brc_avoid_bank_conflict": true, "_is_vnchwconv_align": true, "_classify_inputs_num": 2, "_contains_elewise_sch": true, "_soc_version": "Ascend910", "_all_unknown_last_const": false, "_is_pure_brc": false, "_has_store_align": true, "_flag_info": [false, false, true, true, true, false, true], "_base_info": {"100": [32, 4, 21840, 10920, 1], "120": [32, 4, 21840, 10920, 4], "121": [32, 4, 21840, 10920, 4], "210": [32, 4, 21840, 10920, 4], "320": [32, 4, 21832, 10912, 1], "230": [32, 4, 21832, 10912, 1], "000": [32, 4, 21840, 10920, 4], "888": [32, 4, 16384, 8192, 4], "999": [32, 4, 21840, 10920, 4]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "232010000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "223010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_2"], "4": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_3"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_4"], "488800001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_0"], "488800002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_1"], "488800003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_2"], "488800004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_3"], "488800005": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_4"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_1"], "299900003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_2"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_3"]}, "_normal_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "232010000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "223010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_2"], "4": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_3"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_4"], "488800001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_0"], "488800002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_1"], "488800003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_2"], "488800004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_3"], "488800005": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_4"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_1"], "299900003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_2"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_3"]}, "_custom_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "212100000": [], "212100001": [], "212100002": [], "212100003": [], "221000000": [], "221000001": [], "221000002": [], "232000000": [], "232010000": [], "223000000": [], "223010000": [], "0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "488800001": [], "488800002": [], "488800003": [], "488800004": [], "488800005": [], "299900000": [], "299900001": [], "299900002": [], "299900003": [], "299900004": []}, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "212000000": [10000, 10100, 10101], "212000001": [10000, 10100, 10101, 20000, 30000], "212000002": [10000, 10100, 10101, 20000, 30001], "212010002": [10000, 10100, 10101, 20000, 30001], "212100000": [10000, 10100, 10101, 10200], "212100001": [10000, 10100, 10101, 10200, 20000, 30000], "212100002": [10000, 10100, 10101, 10200, 20000, 30001], "212100003": [10000, 10100, 10101, 10200, 20000, 30002], "221000000": [10000, 10001, 10100], "221000001": [10000, 10001, 10100, 20000, 30000], "221000002": [10000, 10001, 10100, 20000, 30001], "232000000": [10001, 20000, 30000], "232010000": [10001, 20000, 30000], "223000000": [10000, 20000, 30000], "223010000": [10000, 20000, 30000], "0": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401], "1": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30000], "2": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30001], "3": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30002], "4": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30003], "5": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30004], "488800001": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30000], "488800002": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30001], "488800003": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30002], "488800004": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30003], "488800005": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30004], "299900000": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301], "299900001": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30000], "299900002": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30001], "299900003": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30002], "299900004": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30003]}})";
  std::vector<std::vector<int64_t>> inputs {{1, 1, 1, 112, 22}, {32, 5, 25, 1, 22}};
  std::vector<std::vector<int64_t>> outputs {{32, 5, 25, 112, 22}};
  ge::DataType dtype = ge::DT_FLOAT16;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;
  test.SetCompileInfo(compileInfo, &broadcast_info);
  EXPECT_EQ(test.Test(), true);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case9) {
  std::string compileInfo = R"({"_fusion_index": [[0], [1], [2]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_brc_avoid_bank_conflict": true, "_is_vnchwconv_align": true, "_classify_inputs_num": 2, "_contains_elewise_sch": true, "_soc_version": "Ascend910", "_all_unknown_last_const": false, "_is_pure_brc": false, "_has_store_align": true, "_flag_info": [false, false, true, true, true, false, true], "_base_info": {"100": [32, 2, 43680, 21840, 1], "120": [32, 2, 43680, 21840, 2], "121": [32, 2, 43680, 21840, 2], "210": [32, 2, 43680, 21840, 2], "320": [32, 2, 43664, 21824, 1], "230": [32, 2, 43664, 21824, 1], "000": [32, 2, 43680, 21840, 2], "888": [32, 2, 32768, 16384, 2], "999": [32, 2, 43680, 21840, 2]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "232010000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "223010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "488800001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "488800002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "488800003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"]}, "_normal_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "232010000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "223010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "488800001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "488800002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "488800003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"]}, "_custom_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "212100000": [], "212100001": [], "212100002": [], "212100003": [], "221000000": [], "221000001": [], "221000002": [], "232000000": [], "232010000": [], "223000000": [], "223010000": [], "0": [], "1": [], "2": [], "3": [], "488800001": [], "488800002": [], "488800003": [], "299900000": [], "299900001": [], "299900002": []}, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "212000000": [10000, 10100, 10101], "212000001": [10000, 10100, 10101, 20000, 30000], "212000002": [10000, 10100, 10101, 20000, 30001], "212010002": [10000, 10100, 10101, 20000, 30001], "212100000": [10000, 10100, 10101, 10200], "212100001": [10000, 10100, 10101, 10200, 20000, 30000], "212100002": [10000, 10100, 10101, 10200, 20000, 30001], "212100003": [10000, 10100, 10101, 10200, 20000, 30002], "221000000": [10000, 10001, 10100], "221000001": [10000, 10001, 10100, 20000, 30000], "221000002": [10000, 10001, 10100, 20000, 30001], "232000000": [10001, 20000, 30000], "232010000": [10001, 20000, 30000], "223000000": [10000, 20000, 30000], "223010000": [10000, 20000, 30000], "0": [10000, 10001, 10100, 10101, 10200, 10201], "1": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30000], "2": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30001], "3": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30002], "488800001": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30000], "488800002": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30001], "488800003": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30002], "299900000": [10000, 10001, 10100, 10101], "299900001": [10000, 10001, 10100, 10101, 20000, 30000], "299900002": [10000, 10001, 10100, 10101, 20000, 30001]}})";
  std::vector<std::vector<int64_t>> inputs {{1, 1, 5824}, {32, 100, 1}};
  std::vector<std::vector<int64_t>> outputs {{32, 100, 5824}};
  ge::DataType dtype = ge::DT_FLOAT16;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;
  test.SetCompileInfo(compileInfo, &broadcast_info);
  EXPECT_EQ(test.Test(), true);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case10) {
  std::string compileInfo = R"({"_fusion_index": [[0],[1],[2]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_soc_version": "Ascend910", "_flag_info": [false, false, true, true, true, false, true], "_base_info": {"100": [32, 4, 21840, 10920, 4]},  "_elewise_vars": {"210000000":[20000, 30000], "210010000":[20000, 30000]}})";
  std::vector<std::vector<int64_t>> inputs {{2, 0, 2}, {2, 0, 2}};
  std::vector<std::vector<int64_t>> outputs {{2, 0, 2}};
  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;
  test.SetCompileInfo(compileInfo, &broadcast_info);
  EXPECT_EQ(test.Test(), true);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case11) {
  std::string compileInfo = R"({"_fusion_index": [[0, 1],[2]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_soc_version": "Ascend910B", "_flag_info": [true], "_base_info": {"000":[48, 2, 32768, 16384, 2]},  "_elewise_vars": {}, "_broadcast_axis": [false, true]})";
  std::vector<std::vector<int64_t>> inputs {{10, 40000, 1}, {10, 40000, 10}};
  std::vector<std::vector<int64_t>> outputs {{10, 40000, 10}};
  ge::DataType dtype = ge::DT_FLOAT16;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;
  test.SetCompileInfo(compileInfo, &broadcast_info);
  EXPECT_EQ(test.Test(), true);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case12) {
  std::string compileInfo = R"({"_fusion_index": [[0, 1],[2]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_soc_version": "Ascend910B", "_flag_info": [true], "_base_info": {"000":[48, 2, 32768, 16384, 2]},  "_elewise_vars": {}, "_broadcast_axis": [false, true]})";
  std::vector<std::vector<int64_t>> inputs {{1, 40000, 0}, {10, 40000, 0}};
  std::vector<std::vector<int64_t>> outputs {{10, 40000, 0}};
  ge::DataType dtype = ge::DT_FLOAT16;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;
  test.SetCompileInfo(compileInfo, &broadcast_info);
  EXPECT_EQ(test.Test(), true);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case13) {
  std::string compileInfo = R"({ "_classify_inputs_num": 2, "_ub_factor_align": 128, "_contains_elewise_sch": true, "_pattern": "Broadcast", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 4, 32768, 16384, 4]}, "_elewise_vars": { "210000000": [ 10000, 20000, 30000] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";
  std::vector<std::vector<int64_t>> inputs {{1, 33, 1089}, {1, 33, 1089}};
  std::vector<std::vector<int64_t>> outputs {{1, 33, 1089}};
  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;
  test.SetCompileInfo(compileInfo, &broadcast_info);
  EXPECT_EQ(test.Test(), true);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case14) {
  std::vector<std::vector<int64_t>> inputs {{1, 40000, 10}, {10, 40000, 10}};
  std::vector<std::vector<int64_t>> outputs {{10, 40000, 10}};
  ge::DataType dtype = ge::DT_FLOAT16;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;

  v3::BroadcastCompileInfo actual_ptr;
  actual_ptr.pattern = SchPattern::BROADCAST;
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
  actual_ptr.soc_version.second = "Ascend910B";

  test.SetCompileInfo(&actual_ptr);
  EXPECT_EQ(test.Test(), true);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case15) {
  std::vector<std::vector<int64_t>> inputs {{1, 1, 5824}, {32, 100, 1}, {32, 100, 1}};
  std::vector<std::vector<int64_t>> outputs {{32, 100, 5824}};;
  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;

  v3::BroadcastCompileInfo actual_ptr;
  actual_ptr.pattern = SchPattern::BROADCAST;
  actual_ptr.base_info_compile.first = true;
  actual_ptr.base_info_compile.second = {{"100", {48, 2, 32768, 16384, 2}}, {"120", {48, 2, 16384, 8192, 2}}, {"121", {48, 2, 16384, 8192, 2}}, {"210", {48, 2, 16384, 8192, 2}}, {"200", {48, 2, 24576, 12288, 2}}, {"000", {48, 2, 16384, 8192, 2}}, {"999", {48, 2, 16384, 8192, 2}}};
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
  actual_ptr.soc_version.second = "Ascend910B";

  test.SetCompileInfo(&actual_ptr);
  OpInfo op_info(&actual_ptr);
  op_info.SetInputShape(&inputs);
  op_info.SetInputType(&dtype);
  EXPECT_EQ(test.Test(&op_info), true);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case16) {
  std::string compileInfo = R"({"_fusion_index": [[0], [1], [2]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_soc_version": "Ascend910B", "_flag_info": [false, false, true, true, true, false, true], "_base_info": {"100": [48, 2, 32768, 16384, 2], "120": [48, 2, 24576, 12288, 2], "121": [48, 2, 19648, 9824, 2], "210": [48, 2, 19648, 9824, 2], "320": [48, 2, 32752, 16368, 2], "230": [48, 2, 32752, 16368, 2], "000": [48, 2, 19648, 9824, 2], "999": [48, 2, 19648, 9824, 2]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212000004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212010004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "212100005": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "212100006": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "212100009": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_2", "_ub_factor_2"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"]}, "_normal_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212000004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212010004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "212100005": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "212100006": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "212100009": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_1"], "6": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_2", "_ub_factor_2"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"]}, "_custom_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "212000004": [], "212010004": [], "212100000": [], "212100001": [], "212100002": [], "212100003": [], "212100005": [], "212100006": [], "212100009": [], "221000000": [], "221000001": [], "221000002": [], "221000004": [], "232000000": [], "223000000": [], "0": [], "1": [], "2": [], "3": [], "5": [], "6": [], "9": [], "299900000": [], "299900001": [], "299900002": [], "299900004": []}, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "212000000": [10000, 10100, 10101], "212000001": [10000, 10100, 10101, 20000, 30000], "212000002": [10000, 10100, 10101, 20000, 30001], "212010002": [10000, 10100, 10101, 20000, 30001], "212000004": [10000, 10100, 10101, 20001, 30001], "212010004": [10000, 10100, 10101, 20001, 30001], "212100000": [10000, 10100, 10101, 10200], "212100001": [10000, 10100, 10101, 10200, 20000, 30000], "212100002": [10000, 10100, 10101, 10200, 20000, 30001], "212100003": [10000, 10100, 10101, 10200, 20000, 30002], "212100005": [10000, 10100, 10101, 10200, 20001, 30001], "212100006": [10000, 10100, 10101, 10200, 20001, 30002], "212100009": [10000, 10100, 10101, 10200, 20002, 30002], "221000000": [10000, 10001, 10100], "221000001": [10000, 10001, 10100, 20000, 30000], "221000002": [10000, 10001, 10100, 20000, 30001], "221000004": [10000, 10001, 10100, 20001, 30001], "232000000": [10001, 20000, 30000], "223000000": [10000, 20000, 30000], "0": [10000, 10001, 10100, 10101, 10200, 10201], "1": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30000], "2": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30001], "3": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30002], "5": [10000, 10001, 10100, 10101, 10200, 10201, 20001, 30001], "6": [10000, 10001, 10100, 10101, 10200, 10201, 20001, 30002], "9": [10000, 10001, 10100, 10101, 10200, 10201, 20002, 30002], "299900000": [10000, 10001, 10100, 10101], "299900001": [10000, 10001, 10100, 10101, 20000, 30000], "299900002": [10000, 10001, 10100, 10101, 20000, 30001], "299900004": [10000, 10001, 10100, 10101, 20001, 30001]}})";
  std::vector<std::vector<int64_t>> inputs {{1, 1, 5824}, {32, 100, 1}, {32, 100, 1}};
  std::vector<std::vector<int64_t>> outputs {{32, 100, 5824}};
  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;
  test.SetCompileInfo(compileInfo, &broadcast_info);
  EXPECT_EQ(test.Test(), true);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case17) {
  std::vector<std::vector<int64_t>> inputs {{1, 40000, 10}, {10, 40000, 10}};
  std::vector<std::vector<int64_t>> outputs {{10, 40000, 10}};
  ge::DataType dtype = ge::DT_FLOAT16;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;

  v3::BroadcastCompileInfo actual_ptr;
  actual_ptr.pattern = SchPattern::BROADCAST;
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
  actual_ptr.soc_version.second = "Ascend910B";

  test.SetCompileInfo(&actual_ptr);
  OpInfo op_info(&actual_ptr);
  op_info.SetInputShape(&inputs);
  op_info.SetInputType(&dtype);
  EXPECT_EQ(test.Test(&op_info), true);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case18) {
  std::string compileInfo = R"({"_fusion_index": [[0], [1]], "_pattern": "Broadcast", "_ub_factor_align": 128,  "_flag_info": [false, false, true, true, false, false, true], "_base_info": {"000": [32, 4, 13104, 6552, 4], "100": [32, 4, 13104, 6552, 4], "120": [32, 4, 13096, 6544, 4], "200": [32, 4, 13104, 6552, 4], "210": [32, 4, 13104, 6552, 4]}, "_elewise_vars": {"0": [10000,10001,10002, 10100,10101,10102],"1": [10000,10001,10002, 10100,10101,10102, 20000, 30000], "2": [10000,10001,10002, 10100,10101,10102, 20000, 30001], "4": [10000,10001,10002, 10100,10101,10102, 20001, 30001], "210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "212000000": [10000, 10100, 10101, 10102],"212000001": [10000, 10100, 10101, 10102, 20000, 30000],"212000002": [10000, 10100, 10101, 10102, 20000, 30001],"212000004": [10000, 10100, 10101, 10102, 20001, 30001],"212010002": [10000, 10100, 10101, 2000, 30001],"212010004": [10000, 10100, 10101, 20001, 30001], "220000000": [ 10000, 10001, 10002, 20000, 30000 ], "221000000": [10000, 10001, 10100],"221000001": [10000, 10001, 10100, 20000, 30000],"221000002": [10000, 10001, 10100, 20000, 30001], "221000004": [10000, 10001, 10100, 20001, 30001]}})";
  std::vector<std::vector<int64_t>> inputs {{700,1}, {700,2000}, {700,2000}};
  std::vector<std::vector<int64_t>> outputs {{700,2000},{700,2000}};
  ge::DataType dtype = ge::DT_FLOAT16;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;
  test.SetCompileInfo(compileInfo, &broadcast_info);
  EXPECT_EQ(test.Test(), true);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case19) {
  std::string compileInfo = R"({"_ub_factor_align": 128,  "_pattern": "Broadcast", "_fusion_index": [[0,1]], "push_status": 0, "_flag_info": [false, false, true, true, true, false, false], "_base_info": {"320": [32, 4, 21840, 10920, 4], "100": [32, 4, 21840, 10920, 4]}, "_elewise_vars": { "232000000": [10001, 20000, 30000],  "210000000": [20000, 30000], "210010000": [10001, 20000, 30000] } })";
  std::vector<std::vector<int64_t>> inputs {{1,1}, {1,}};
  std::vector<std::vector<int64_t>> outputs {{1,1}, };
  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;
  test.SetCompileInfo(compileInfo, &broadcast_info);
  EXPECT_EQ(test.Test(), true);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case20) {
  std::string compileInfo = R"({"_ub_factor_align": 128,  "_pattern": "Broadcast", "_fusion_index": [[0,1,2,3,4]], "push_status": 0, "_flag_info": [false, false, true, true, true, false, false], "_base_info": {"230": [32, 2, 42320, 21152, 2]}, "_elewise_vars": { "223000000": [10000, 20000, 30000] } })";
  std::vector<std::vector<int64_t>> inputs {{2, 26, 26, 3, 80}, {1}};
  std::vector<std::vector<int64_t>> outputs {{2, 26, 26, 3, 80}};
  ge::DataType dtype = ge::DT_FLOAT16;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;
  test.SetCompileInfo(compileInfo, &broadcast_info);
  EXPECT_EQ(test.Test(), true);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case22) {
  std::string compileInfo = R"({"_ub_factor_align": 128,  "_pattern": "Broadcast", "_fusion_index": [[0,1,2,3,4]], "push_status": 0, "_flag_info": [false, false, true, true, true, false, false], "_base_info": {"230": [32, 2, 42320, 21152, 2]}, "_elewise_vars": { "223000000": [10000, 20000, 30000] },  "_attr_vars": { "223000000": [{"length":1, "name":"alpha", "type":"int32", "src_type":"int32"}] } })";
  std::vector<std::vector<int64_t>> inputs {{2, 26, 26, 3, 80}, {1}};
  std::vector<std::vector<int64_t>> outputs {{2, 26, 26, 3, 80}};
  ge::DataType dtype = ge::DT_FLOAT16;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;
  test.SetCompileInfo(compileInfo, &broadcast_info);
  std::string var_attr_list_compileInfo = R"(
    {
      "_var_attr_mode":0,
      "_var_attrs": [
        {
          "length":1,
          "name":"alpha",
          "index":0,
          "type":"int32",
          "src_type":"int64"
        }
      ]
    }
  )";
  broadcast_info.var_attr_wrap.ParseVarAttr(nlohmann::json::parse(var_attr_list_compileInfo));
  std::vector<std::pair<std::string, int64_t>> common_attr = {{"int64", {2}}};
  test.SetAttrs<int64_t>(common_attr);

  EXPECT_EQ(test.Test(), true);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case23) {
  std::vector<std::vector<int64_t>> inputs {{1, 40000, 10}, {10, 40000, 10}};
  std::vector<std::vector<int64_t>> outputs {{10, 40000, 10}};
  ge::DataType dtype = ge::DT_FLOAT16;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;

  v3::BroadcastCompileInfo actual_ptr;
  actual_ptr.pattern = SchPattern::BROADCAST;
  actual_ptr.base_info_compile.first = false;
  actual_ptr.base_info_compile.second = {};
  actual_ptr.flag_info_compile = {false, true, true, false, false, false, false};
  actual_ptr.ub_factor_align = 128;
  actual_ptr.elewise_vars_compile.first = true;
  actual_ptr.elewise_vars_compile.second = {{"100000000",{}}};
  actual_ptr.const_block_dims_compile.first = true;
  actual_ptr.const_block_dims_compile.second = {40};

  std::string var_attr_list_compileInfo = R"(
    {
      "_var_attr_mode":0,
      "_var_attrs": [
        {
          "length":1,
          "name":"alpha",
          "index":0,
          "type":"int32",
          "src_type":"int64"
        }
      ]
    }
  )";
  actual_ptr.var_attr_wrap.ParseVarAttr(nlohmann::json::parse(var_attr_list_compileInfo));

  actual_ptr.const_shapes_compile.first = true;
  actual_ptr.const_shapes_compile.second = {{1, 40000, 10}};

  actual_ptr.fusion_index_compile.first = true;
  actual_ptr.fusion_index_compile.second = {{0},{1,2}};

  actual_ptr.broadcast_axis_compile.first = true;
  actual_ptr.broadcast_axis_compile.second = {false, true};

  actual_ptr.soc_version.first = true;
  actual_ptr.soc_version.second = "Ascend910B";

  std::vector<std::pair<std::string, int64_t>> common_attr = {{"int64", {2}}};
  test.SetAttrs<int64_t>(common_attr);
  test.SetCompileInfo(&actual_ptr);
  EXPECT_EQ(test.Test(), true);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case24) {
  std::string compileInfo = R"({"_ub_factor_align": 128,  "_pattern": "Broadcast", "_fusion_index": [[0,1,2,3]],"_soc_version": "Ascend910B", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [48, 4, 16384, 8192, 4]}, "_elewise_vars": { "210000000": [10000, 20000, 30000],"210010000": [10000, 20000, 30000]} })";
  std::vector<std::vector<int64_t>> inputs {{8, 7, 7, 6}};
  std::vector<std::vector<int64_t>> outputs {{8, 7, 7, 6}};
  ge::DataType dtype = ge::DT_FLOAT16;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;
  test.SetCompileInfo(compileInfo, &broadcast_info);
  EXPECT_EQ(test.Test(), true);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case25) {
  std::string compileInfo = R"({"_ub_factor_align": 128,  "_pattern": "Broadcast", "_fusion_index": [[0],[1],[2],[3]],"_soc_version": "Ascend910B", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"0": [48, 4, 16384, 8192, 4]}, "_elewise_vars": { "210000000": [10000, 20000, 30000],"210010000": [10000, 20000, 30000]} })";
  std::vector<std::vector<int64_t>> inputs {{8, 7, 7, 6},{1,7,1,6},{8,1,7,1}};
  std::vector<std::vector<int64_t>> outputs {{8, 7, 7, 6}};
  ge::DataType dtype = ge::DT_FLOAT16;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;
  test.SetCompileInfo(compileInfo, &broadcast_info);
  EXPECT_EQ(test.Test(), false);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case26) {
  std::vector<std::vector<int64_t>> inputs {{2, 1, 128, 16, 16}, {2, 1, 128, 1, 16}};
  std::vector<std::vector<int64_t>> outputs {{2, 1, 128, 16, 16}};
  std::vector<std::vector<int64_t>> ori_inputs {{2, 128, 16, 1}, {2, 128, 1, 16}};
  std::vector<std::vector<int64_t>> ori_outputs {{2, 128, 16, 16}};
  std::vector<ge::DataType> dtype = {ge::DT_FLOAT};
  std::vector<ge::Format> format = {ge::FORMAT_ND};

  AutoTilingTest test(ori_inputs, inputs, ori_outputs, outputs, dtype, dtype, format, format, format, format);

  v3::BroadcastCompileInfo actual_ptr;
  actual_ptr.pattern = SchPattern::BROADCAST;
  actual_ptr.base_info_compile.first = true;
  actual_ptr.base_info_compile.second = {{"100", {32, 4, 21840, 10920}}, {"121", {32, 4, 16376, 8184}}, {"210", {32, 4, 16376, 8184}},  {"000", {32, 4, 16376, 8184}}, {"999", {32, 4, 16376, 8184}}};
  actual_ptr.flag_info_compile = {false, false, true, true, false, false, true};
  actual_ptr.ub_factor_align = 128;
  actual_ptr.elewise_vars_compile.first = true;
  actual_ptr.elewise_vars_compile.second = {{"3", {40300, 40301, 10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30002}}, };
  actual_ptr.const_block_dims_compile.first = false;
  actual_ptr.const_block_dims_compile.second = {};

  actual_ptr.const_shapes_compile.first = false;
  actual_ptr.const_shapes_compile.second = {};

  actual_ptr.fusion_index_compile.first = true;
  actual_ptr.fusion_index_compile.second = {{0}, {1}, {2}, {3}, {4}};

  actual_ptr.broadcast_axis_compile.first = false;
  actual_ptr.broadcast_axis_compile.second = {};

  actual_ptr.contains_need_pad_compute = true;

  actual_ptr.pad_axis_index = 3;

  actual_ptr.disable_fuse_axes_compile.first = true;
  actual_ptr.disable_fuse_axes_compile.second = {1, 4};

  actual_ptr.soc_version.first = true;
  actual_ptr.soc_version.second = "Ascend910";

  test.SetCompileInfo(&actual_ptr);
  EXPECT_EQ(test.Test(), true);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case27) {
  std::string compileInfo = R"({"_ub_factor_align": 128, "_pad_axis_index": 3, "_contains_need_pad_compute": true, "_disable_fuse_axes": [1, 4] ,"_pattern": "Broadcast", "_fusion_index": [[0],[1],[2],[3],[4]],"_soc_version": "Ascend910", "push_status": 0, "_flag_info": [false, false, true, true, false, false, true], "_base_info": {"0": [32, 4, 16376, 8184]}, "_elewise_vars": {"3": [40300, 40301, 10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30002]} })";
  std::vector<std::vector<int64_t>> inputs {{2, 1, 128, 16, 16}, {2, 1, 128, 1, 16}};
  std::vector<std::vector<int64_t>> outputs {{2, 1, 128, 16, 16}};
  std::vector<std::vector<int64_t>> ori_inputs {{2, 128, 16, 1}, {2, 128, 1, 16}};
  std::vector<std::vector<int64_t>> ori_outputs {{2, 128, 16, 16}};
  std::vector<ge::DataType> dtype = {ge::DT_FLOAT};
  std::vector<ge::Format> format = {ge::FORMAT_ND};

  AutoTilingTest test(ori_inputs, inputs, ori_outputs, outputs, dtype, dtype, format, format, format, format);
  optiling::v3::BroadcastCompileInfo broadcast_info;
  test.SetCompileInfo(compileInfo, &broadcast_info);
  EXPECT_EQ(test.Test(), false);
  
}

TEST_F(BroadcastTilingV3, broadcast_rt_case28) {
  // [(-1, -1), (-1, -1, -1, -1, -1)] fp32
  std::string compileInfo = R"({"_fusion_index": [[0, 1, 2], [3], [4]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_brc_avoid_bank_conflict": true, "_is_vnchwconv_align": true, "_classify_inputs_num": 2, "_contains_elewise_sch": true, "_soc_version": "Ascend910", "_all_unknown_last_const": false, "_is_pure_brc": false, "_has_store_align": true, "_flag_info": [false, false, true, true, true, false, true], "_base_info": {"100": [32, 4, 21840, 10920, 1], "120": [32, 4, 21840, 10920, 4], "210": [32, 4, 21840, 10920, 4], "320": [32, 4, 21832, 10912, 1], "230": [32, 4, 21832, 10912, 1], "000": [32, 4, 21840, 10920, 4], "888": [32, 4, 16384, 8192, 4], "999": [32, 4, 21840, 10920, 4]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "232010000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "223010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1"], "1": ["_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "488800001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "488800002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "488800003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"]}, "_normal_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "232010000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "223010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1"], "1": ["_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "488800001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "488800002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "488800003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"]}, "_custom_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "221000000": [], "221000001": [], "221000002": [], "232000000": [], "232010000": [], "223000000": [], "223010000": [], "0": [], "1": [], "2": [], "3": [], "488800001": [], "488800002": [], "488800003": [], "299900000": [], "299900001": [], "299900002": []}, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "212000000": [10000, 10100, 10101], "212000001": [10000, 10100, 10101, 20000, 30000], "212000002": [10000, 10100, 10101, 20000, 30001], "212010002": [10000, 10100, 10101, 20000, 30001], "221000000": [10000, 10001, 10100], "221000001": [10000, 10001, 10100, 20000, 30000], "221000002": [10000, 10001, 10100, 20000, 30001], "232000000": [10001, 20000, 30000], "232010000": [10001, 20000, 30000], "223000000": [10000, 20000, 30000], "223010000": [10000, 20000, 30000], "0": [10001, 10100, 10101, 10200, 10201], "1": [10001, 10100, 10101, 10200, 10201, 20000, 30000], "2": [10001, 10100, 10101, 10200, 10201, 20000, 30001], "3": [10001, 10100, 10101, 10200, 10201, 20000, 30002], "488800001": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30000], "488800002": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30001], "488800003": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30002], "299900000": [10000, 10001, 10100, 10101], "299900001": [10000, 10001, 10100, 10101, 20000, 30000], "299900002": [10000, 10001, 10100, 10101, 20000, 30001]}})";
  std::vector<std::vector<int64_t>> inputs {{2, 129}, {4, 88, 9, 2, 129}};
  std::vector<std::vector<int64_t>> outputs {{4, 88, 9, 2, 129}};
  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;
  test.SetCompileInfo(compileInfo, &broadcast_info);
  EXPECT_EQ(test.Test(), true);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case29) {
  // [(-1, -1), (-1, -1, -1, -1, -1)] fp32
  std::string compileInfo = R"({"_fusion_index": [[0, 1, 2], [3], [4]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_brc_avoid_bank_conflict": true, "_is_vnchwconv_align": true, "_classify_inputs_num": 2, "_contains_elewise_sch": true, "_soc_version": "Ascend910", "_all_unknown_last_const": false, "_is_pure_brc": false, "_has_store_align": true, "_flag_info": [false, false, true, true, true, false, true], "_base_info": {"100": [32, 4, 21840, 10920, 1], "120": [32, 4, 21840, 10920, 4], "210": [32, 4, 21840, 10920, 4], "320": [32, 4, 21832, 10912, 1], "230": [32, 4, 21832, 10912, 1], "000": [32, 4, 21840, 10920, 4], "888": [32, 4, 16384, 8192, 4], "999": [32, 4, 21840, 10920, 4]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "232010000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "223010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1"], "1": ["_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "488800001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "488800002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "488800003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"]}, "_normal_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "232010000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "223010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1"], "1": ["_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "488800001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_0"], "488800002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_1"], "488800003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_block_factor_0", "_ub_factor_2"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"]}, "_custom_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "221000000": [], "221000001": [], "221000002": [], "232000000": [], "232010000": [], "223000000": [], "223010000": [], "0": [], "1": [], "2": [], "3": [], "488800001": [], "488800002": [], "488800003": [], "299900000": [], "299900001": [], "299900002": []}, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "212000000": [10000, 10100, 10101], "212000001": [10000, 10100, 10101, 20000, 30000], "212000002": [10000, 10100, 10101, 20000, 30001], "212010002": [10000, 10100, 10101, 20000, 30001], "221000000": [10000, 10001, 10100], "221000001": [10000, 10001, 10100, 20000, 30000], "221000002": [10000, 10001, 10100, 20000, 30001], "232000000": [10001, 20000, 30000], "232010000": [10001, 20000, 30000], "223000000": [10000, 20000, 30000], "223010000": [10000, 20000, 30000], "0": [10001, 10100, 10101, 10200, 10201], "1": [10001, 10100, 10101, 10200, 10201, 20000, 30000], "2": [10001, 10100, 10101, 10200, 10201, 20000, 30001], "3": [10001, 10100, 10101, 10200, 10201, 20000, 30002], "488800001": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30000], "488800002": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30001], "488800003": [10000, 10001, 10100, 10101, 10200, 10201, 20000, 30002], "299900000": [10000, 10001, 10100, 10101], "299900001": [10000, 10001, 10100, 10101, 20000, 30000], "299900002": [10000, 10001, 10100, 10101, 20000, 30001]}})";
  std::vector<std::vector<int64_t>> inputs {{2, 49890}, {4, 88, 9, 2, 49890}};
  std::vector<std::vector<int64_t>> outputs {{4, 88, 9, 2, 49890}};
  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;
  test.SetCompileInfo(compileInfo, &broadcast_info);
  EXPECT_EQ(test.Test(), true);
}

TEST_F(BroadcastTilingV3, broadcast_rt_case30) {
  // [(-1, -1, -1), (-1,)] [fp32, int32] pure brc
  std::string compileInfo = R"({"_fusion_index": [[0], [1], [2], [3], [4], [5]], "_pattern": "Broadcast", "_brc_avoid_bank_conflict": true, "_ub_factor_align": 128, "_is_vnchwconv_align": true, "_classify_inputs_num": 2, "_contains_elewise_sch": false, "_soc_version": "Ascend910", "_all_unknown_last_const": false, "_is_pure_brc": true, "_has_store_align": false, "_flag_info": [false, false, true, false, false, false, false], "_base_info": {"500": [32, 4, 21840, 10920, 4], "510": [32, 4, 21840, 10920, 4], "400": [32, 4, 21840, 10920, 4], "410": [32, 4, 21840, 10920, 4]}, "_vars": {"350000000": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1"], "350000001": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_0"], "350000002": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_1"], "350000003": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_2"], "350000004": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_3"], "350000005": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_4"], "350000006": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_5"], "351000000": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1"], "351000001": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_0"], "351000002": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_1"], "351000003": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_2"], "351000004": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_3"], "351000005": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_4"], "351000006": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_5"], "340000000": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0"], "340000001": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_0"], "340000002": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_1"], "340000003": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_2"], "340000004": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_3"], "340000005": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_4"], "340000006": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_5"], "341010000": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0"], "341010001": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_0"], "341010002": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_1"], "341010003": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_2"], "341010004": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_3"], "341010005": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_4"], "341010006": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_5"]}, "_normal_vars": {"350000000": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1"], "350000001": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_0"], "350000002": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_1"], "350000003": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_2"], "350000004": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_3"], "350000005": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_4"], "350000006": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_5"], "351000000": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1"], "351000001": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_0"], "351000002": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_1"], "351000003": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_2"], "351000004": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_3"], "351000005": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_4"], "351000006": ["_dim_0_0", "_dim_1_1", "_dim_2_0", "_dim_3_1", "_dim_4_0", "_dim_5_1", "_block_factor_0", "_ub_factor_5"], "340000000": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0"], "340000001": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_0"], "340000002": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_1"], "340000003": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_2"], "340000004": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_3"], "340000005": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_4"], "340000006": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_5"], "341010000": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0"], "341010001": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_0"], "341010002": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_1"], "341010003": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_2"], "341010004": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_3"], "341010005": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_4"], "341010006": ["_dim_0_1", "_dim_1_0", "_dim_2_1", "_dim_3_0", "_dim_4_1", "_dim_5_0", "_block_factor_0", "_ub_factor_5"]}, "_custom_vars": {"350000000": [], "350000001": [], "350000002": [], "350000003": [], "350000004": [], "350000005": [], "350000006": [], "351000000": [], "351000001": [], "351000002": [], "351000003": [], "351000004": [], "351000005": [], "351000006": [], "340000000": [], "340000001": [], "340000002": [], "340000003": [], "340000004": [], "340000005": [], "340000006": [], "341010000": [], "341010001": [], "341010002": [], "341010003": [], "341010004": [], "341010005": [], "341010006": []}, "_elewise_vars": {"350000000": [10000, 10101, 10200, 10301, 10400, 10501], "350000001": [10000, 10101, 10200, 10301, 10400, 10501, 20000, 30000], "350000002": [10000, 10101, 10200, 10301, 10400, 10501, 20000, 30001], "350000003": [10000, 10101, 10200, 10301, 10400, 10501, 20000, 30002], "350000004": [10000, 10101, 10200, 10301, 10400, 10501, 20000, 30003], "350000005": [10000, 10101, 10200, 10301, 10400, 10501, 20000, 30004], "350000006": [10000, 10101, 10200, 10301, 10400, 10501, 20000, 30005], "351000000": [10000, 10101, 10200, 10301, 10400, 10501], "351000001": [10000, 10101, 10200, 10301, 10400, 10501, 20000, 30000], "351000002": [10000, 10101, 10200, 10301, 10400, 10501, 20000, 30001], "351000003": [10000, 10101, 10200, 10301, 10400, 10501, 20000, 30002], "351000004": [10000, 10101, 10200, 10301, 10400, 10501, 20000, 30003], "351000005": [10000, 10101, 10200, 10301, 10400, 10501, 20000, 30004], "351000006": [10000, 10101, 10200, 10301, 10400, 10501, 20000, 30005], "340000000": [10001, 10100, 10201, 10300, 10401, 10500], "340000001": [10001, 10100, 10201, 10300, 10401, 10500, 20000, 30000], "340000002": [10001, 10100, 10201, 10300, 10401, 10500, 20000, 30001], "340000003": [10001, 10100, 10201, 10300, 10401, 10500, 20000, 30002], "340000004": [10001, 10100, 10201, 10300, 10401, 10500, 20000, 30003], "340000005": [10001, 10100, 10201, 10300, 10401, 10500, 20000, 30004], "340000006": [10001, 10100, 10201, 10300, 10401, 10500, 20000, 30005], "341010000": [10001, 10100, 10201, 10300, 10401, 10500], "341010001": [10001, 10100, 10201, 10300, 10401, 10500, 20000, 30000], "341010002": [10001, 10100, 10201, 10300, 10401, 10500, 20000, 30001], "341010003": [10001, 10100, 10201, 10300, 10401, 10500, 20000, 30002], "341010004": [10001, 10100, 10201, 10300, 10401, 10500, 20000, 30003], "341010005": [10001, 10100, 10201, 10300, 10401, 10500, 20000, 30004], "341010006": [10001, 10100, 10201, 10300, 10401, 10500, 20000, 30005]}, "compile_shape": [-1, -1, -1]})";
  std::vector<std::vector<int64_t>> inputs {{1, 2, 1, 9, 1, 100}, {4, 2, 1, 9, 8, 100}};
  std::vector<std::vector<int64_t>> outputs {{4, 2, 1, 9, 8, 100}};
  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::BroadcastCompileInfo broadcast_info;
  test.SetCompileInfo(compileInfo, &broadcast_info);
  EXPECT_EQ(test.Test(), true);
}
