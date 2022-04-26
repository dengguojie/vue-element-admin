#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "selection_ops.h"
#include "array_ops.h"

using namespace std;

class TileWithAxisTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TileWithAxisTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TileWithAxisTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream &tiling_data) {

  auto data = tiling_data.str();
  string result;
  int32_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }

  return result;
}

using namespace ge;
#include "common/utils/ut_op_util.h"
using namespace ut_util;

TEST_F(TileWithAxisTiling, TileWithAxisTiling_test_1) {


  std::string op_name = "TileWithAxis";

  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = R"({"ori_axis_value": -1, "attr_axis": 0, "attr_tiles": 2, "_fusion_index": [[0], [1], [2], [3], [4], [5]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_flag_info": [false, false, true, false, false, false, false], "_base_info": {"000": [32, 4, 32760, 16376]}, "_elewise_vars": {"0": [10100], "1": [10100, 20000, 30000], "2": [10100, 20000, 30001], "3": [10100, 20000, 30002], "4": [10100, 20000, 30003], "5": [10100, 20000, 30004], "6": [10100, 20000, 30005], "8": [10100, 20001, 30001], "9": [10100, 20001, 30002], "10": [10100, 20001, 30003], "11": [10100, 20001, 30004], "12": [10100, 20001, 30005], "15": [10100, 20002, 30002], "16": [10100, 20002, 30003], "17": [10100, 20002, 30004], "18": [10100, 20002, 30005], "22": [10100, 20003, 30003], "23": [10100, 20003, 30004], "24": [10100, 20003, 30005], "29": [10100, 20004, 30004], "30": [10100, 20004, 30005], "36": [10100, 20005, 30005]}, "push_status": 1})";

  std::vector<int64_t> input{5, 2, 16, 8, 3};
  std::vector<int64_t> output{5, 2, 16, 8, 3};
  ge::DataType in_dtype = ge::DT_FLOAT16;

  auto opParas = op::TileWithAxis("TileWithAxis");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, in_dtype, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, in_dtype, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "5 1 16 ");
}

TEST_F(TileWithAxisTiling, TileWithAxisTiling_test_2) {


  std::string op_name = "TileWithAxis";

  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = R"({"ori_axis_value": -1, "attr_axis": 0, "attr_tiles": 2, "_fusion_index": [[0], [1], [2], [3], [4], [5]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_flag_info": [false, false, true, false, false, false, false], "_base_info": {"000": [32, 4, 32760, 16376]}, "_elewise_vars": {"0": [10100], "1": [10100, 20000, 30000], "2": [10100, 20000, 30001], "3": [10100, 20000, 30002], "4": [10100, 20000, 30003], "5": [10100, 20000, 30004], "6": [10100, 20000, 30005], "8": [10100, 20001, 30001], "9": [10100, 20001, 30002], "10": [10100, 20001, 30003], "11": [10100, 20001, 30004], "12": [10100, 20001, 30005], "15": [10100, 20002, 30002], "16": [10100, 20002, 30003], "17": [10100, 20002, 30004], "18": [10100, 20002, 30005], "22": [10100, 20003, 30003], "23": [10100, 20003, 30004], "24": [10100, 20003, 30005], "29": [10100, 20004, 30004], "30": [10100, 20004, 30005], "36": [10100, 20005, 30005]}, "push_status": 1})";

  std::vector<int64_t> input{5, 2, 16, 8, 3};
  std::vector<int64_t> output{5, 2, 16, 8, 3};
  ge::DataType in_dtype = ge::DT_FLOAT;

  auto opParas = op::TileWithAxis("TileWithAxis");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, in_dtype, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, in_dtype, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "5 1 16 ");
}

TEST_F(TileWithAxisTiling, TileWithAxisTiling_test_3) {


  std::string op_name = "TileWithAxis";

  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = R"({"ori_axis_value": -1, "attr_axis": 0, "attr_tiles": 2, "_fusion_index": [[0], [1], [2], [3], [4], [5]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_flag_info": [false, false, true, false, false, false, false], "_base_info": {"000": [32, 4, 32760, 16376]}, "_elewise_vars": {"0": [10100], "1": [10100, 20000, 30000], "2": [10100, 20000, 30001], "3": [10100, 20000, 30002], "4": [10100, 20000, 30003], "5": [10100, 20000, 30004], "6": [10100, 20000, 30005], "8": [10100, 20001, 30001], "9": [10100, 20001, 30002], "10": [10100, 20001, 30003], "11": [10100, 20001, 30004], "12": [10100, 20001, 30005], "15": [10100, 20002, 30002], "16": [10100, 20002, 30003], "17": [10100, 20002, 30004], "18": [10100, 20002, 30005], "22": [10100, 20003, 30003], "23": [10100, 20003, 30004], "24": [10100, 20003, 30005], "29": [10100, 20004, 30004], "30": [10100, 20004, 30005], "36": [10100, 20005, 30005]}, "push_status": 1})";

  std::vector<int64_t> input{5, 2, 16, 8, 3};
  std::vector<int64_t> output{5, 2, 16, 8, 3};
  ge::DataType in_dtype = ge::DT_INT64;

  auto opParas = op::TileWithAxis("TileWithAxis");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, in_dtype, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, in_dtype, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "5 1 16 ");
}

TEST_F(TileWithAxisTiling, TileWithAxisTiling_test_4) {


  std::string op_name = "TileWithAxis";

  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = R"({"ori_axis_value": -1, "attr_axis": 0, "attr_tiles": 2, "_fusion_index": [[0], [1], [2], [3], [4], [5]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_flag_info": [false, false, true, false, false, false, false], "_base_info": {"000": [32, 4, 32760, 16376]}, "_elewise_vars": {"0": [10100], "1": [10100, 20000, 30000], "2": [10100, 20000, 30001], "3": [10100, 20000, 30002], "4": [10100, 20000, 30003], "5": [10100, 20000, 30004], "6": [10100, 20000, 30005], "8": [10100, 20001, 30001], "9": [10100, 20001, 30002], "10": [10100, 20001, 30003], "11": [10100, 20001, 30004], "12": [10100, 20001, 30005], "15": [10100, 20002, 30002], "16": [10100, 20002, 30003], "17": [10100, 20002, 30004], "18": [10100, 20002, 30005], "22": [10100, 20003, 30003], "23": [10100, 20003, 30004], "24": [10100, 20003, 30005], "29": [10100, 20004, 30004], "30": [10100, 20004, 30005], "36": [10100, 20005, 30005]}, "push_status": 1})";

  std::vector<int64_t> input{5, 2, 16, 8, 3};
  std::vector<int64_t> output{5, 2, 16, 8, 3};
  ge::DataType in_dtype = ge::DT_INT32;

  auto opParas = op::TileWithAxis("TileWithAxis");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, in_dtype, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, in_dtype, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "5 1 16 ");
}

TEST_F(TileWithAxisTiling, TileWithAxisTiling_test_5) {


  std::string op_name = "TileWithAxis";

  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = R"({"ori_axis_value": -1, "attr_axis": 0, "attr_tiles": 2, "_fusion_index": [[0], [1], [2], [3], [4], [5]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_flag_info": [false, false, true, false, false, false, false], "_base_info": {"000": [32, 4, 32760, 16376]}, "_elewise_vars": {"0": [10100], "1": [10100, 20000, 30000], "2": [10100, 20000, 30001], "3": [10100, 20000, 30002], "4": [10100, 20000, 30003], "5": [10100, 20000, 30004], "6": [10100, 20000, 30005], "8": [10100, 20001, 30001], "9": [10100, 20001, 30002], "10": [10100, 20001, 30003], "11": [10100, 20001, 30004], "12": [10100, 20001, 30005], "15": [10100, 20002, 30002], "16": [10100, 20002, 30003], "17": [10100, 20002, 30004], "18": [10100, 20002, 30005], "22": [10100, 20003, 30003], "23": [10100, 20003, 30004], "24": [10100, 20003, 30005], "29": [10100, 20004, 30004], "30": [10100, 20004, 30005], "36": [10100, 20005, 30005]}, "push_status": 1})";

  std::vector<int64_t> input{5, 2, 16, 8, 3};
  std::vector<int64_t> output{5, 2, 16, 8, 3};
  ge::DataType in_dtype = ge::DT_INT16;

  auto opParas = op::TileWithAxis("TileWithAxis");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, in_dtype, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, in_dtype, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "5 1 16 ");
}

TEST_F(TileWithAxisTiling, TileWithAxisTiling_test_6) {


  std::string op_name = "TileWithAxis";

  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = R"({"ori_axis_value": -1, "attr_axis": 0, "attr_tiles": 2, "_fusion_index": [[0], [1], [2], [3], [4], [5]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_flag_info": [false, false, true, false, false, false, false], "_base_info": {"000": [32, 4, 32760, 16376]}, "_elewise_vars": {"0": [10100], "1": [10100, 20000, 30000], "2": [10100, 20000, 30001], "3": [10100, 20000, 30002], "4": [10100, 20000, 30003], "5": [10100, 20000, 30004], "6": [10100, 20000, 30005], "8": [10100, 20001, 30001], "9": [10100, 20001, 30002], "10": [10100, 20001, 30003], "11": [10100, 20001, 30004], "12": [10100, 20001, 30005], "15": [10100, 20002, 30002], "16": [10100, 20002, 30003], "17": [10100, 20002, 30004], "18": [10100, 20002, 30005], "22": [10100, 20003, 30003], "23": [10100, 20003, 30004], "24": [10100, 20003, 30005], "29": [10100, 20004, 30004], "30": [10100, 20004, 30005], "36": [10100, 20005, 30005]}, "push_status": 1})";

  std::vector<int64_t> input{5, 2, 16, 8, 3};
  std::vector<int64_t> output{5, 2, 16, 8, 3};
  ge::DataType in_dtype = ge::DT_INT8;

  auto opParas = op::TileWithAxis("TileWithAxis");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, in_dtype, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, in_dtype, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "5 1 16 ");
}

TEST_F(TileWithAxisTiling, TileWithAxisTiling_test_7) {


  std::string op_name = "TileWithAxis";

  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = R"({"ori_axis_value": -1, "attr_axis": 0, "attr_tiles": 2, "_fusion_index": [[0], [1], [2], [3], [4], [5]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_flag_info": [false, false, true, false, false, false, false], "_base_info": {"000": [32, 4, 32760, 16376]}, "_elewise_vars": {"0": [10100], "1": [10100, 20000, 30000], "2": [10100, 20000, 30001], "3": [10100, 20000, 30002], "4": [10100, 20000, 30003], "5": [10100, 20000, 30004], "6": [10100, 20000, 30005], "8": [10100, 20001, 30001], "9": [10100, 20001, 30002], "10": [10100, 20001, 30003], "11": [10100, 20001, 30004], "12": [10100, 20001, 30005], "15": [10100, 20002, 30002], "16": [10100, 20002, 30003], "17": [10100, 20002, 30004], "18": [10100, 20002, 30005], "22": [10100, 20003, 30003], "23": [10100, 20003, 30004], "24": [10100, 20003, 30005], "29": [10100, 20004, 30004], "30": [10100, 20004, 30005], "36": [10100, 20005, 30005]}, "push_status": 1})";

  std::vector<int64_t> input{5, 2, 16, 8, 3};
  std::vector<int64_t> output{5, 2, 16, 8, 3};
  ge::DataType in_dtype = ge::DT_UINT64;

  auto opParas = op::TileWithAxis("TileWithAxis");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, in_dtype, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, in_dtype, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "5 1 16 ");
}

TEST_F(TileWithAxisTiling, TileWithAxisTiling_test_8) {


  std::string op_name = "TileWithAxis";

  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = R"({"ori_axis_value": -1, "attr_axis": 0, "attr_tiles": 2, "_fusion_index": [[0], [1], [2], [3], [4], [5]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_flag_info": [false, false, true, false, false, false, false], "_base_info": {"000": [32, 4, 32760, 16376]}, "_elewise_vars": {"0": [10100], "1": [10100, 20000, 30000], "2": [10100, 20000, 30001], "3": [10100, 20000, 30002], "4": [10100, 20000, 30003], "5": [10100, 20000, 30004], "6": [10100, 20000, 30005], "8": [10100, 20001, 30001], "9": [10100, 20001, 30002], "10": [10100, 20001, 30003], "11": [10100, 20001, 30004], "12": [10100, 20001, 30005], "15": [10100, 20002, 30002], "16": [10100, 20002, 30003], "17": [10100, 20002, 30004], "18": [10100, 20002, 30005], "22": [10100, 20003, 30003], "23": [10100, 20003, 30004], "24": [10100, 20003, 30005], "29": [10100, 20004, 30004], "30": [10100, 20004, 30005], "36": [10100, 20005, 30005]}, "push_status": 1})";

  std::vector<int64_t> input{5, 2, 16, 8, 3};
  std::vector<int64_t> output{5, 2, 16, 8, 3};
  ge::DataType in_dtype = ge::DT_UINT32;

  auto opParas = op::TileWithAxis("TileWithAxis");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, in_dtype, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, in_dtype, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "5 1 16 ");
}

TEST_F(TileWithAxisTiling, TileWithAxisTiling_test_9) {


  std::string op_name = "TileWithAxis";

  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = R"({"ori_axis_value": -1, "attr_axis": 0, "attr_tiles": 2, "_fusion_index": [[0], [1], [2], [3], [4], [5]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_flag_info": [false, false, true, false, false, false, false], "_base_info": {"000": [32, 4, 32760, 16376]}, "_elewise_vars": {"0": [10100], "1": [10100, 20000, 30000], "2": [10100, 20000, 30001], "3": [10100, 20000, 30002], "4": [10100, 20000, 30003], "5": [10100, 20000, 30004], "6": [10100, 20000, 30005], "8": [10100, 20001, 30001], "9": [10100, 20001, 30002], "10": [10100, 20001, 30003], "11": [10100, 20001, 30004], "12": [10100, 20001, 30005], "15": [10100, 20002, 30002], "16": [10100, 20002, 30003], "17": [10100, 20002, 30004], "18": [10100, 20002, 30005], "22": [10100, 20003, 30003], "23": [10100, 20003, 30004], "24": [10100, 20003, 30005], "29": [10100, 20004, 30004], "30": [10100, 20004, 30005], "36": [10100, 20005, 30005]}, "push_status": 1})";

  std::vector<int64_t> input{5, 2, 16, 8, 3};
  std::vector<int64_t> output{5, 2, 16, 8, 3};
  ge::DataType in_dtype = ge::DT_UINT16;

  auto opParas = op::TileWithAxis("TileWithAxis");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, in_dtype, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, in_dtype, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "5 1 16 ");
}

TEST_F(TileWithAxisTiling, TileWithAxisTiling_test_a) {


  std::string op_name = "TileWithAxis";

  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = R"({"ori_axis_value": -1, "attr_axis": 0, "attr_tiles": 2, "_fusion_index": [[0], [1], [2], [3], [4], [5]], "_pattern": "Broadcast", "_ub_factor_align": 128, "_flag_info": [false, false, true, false, false, false, false], "_base_info": {"000": [32, 4, 32760, 16376]}, "_elewise_vars": {"0": [10100], "1": [10100, 20000, 30000], "2": [10100, 20000, 30001], "3": [10100, 20000, 30002], "4": [10100, 20000, 30003], "5": [10100, 20000, 30004], "6": [10100, 20000, 30005], "8": [10100, 20001, 30001], "9": [10100, 20001, 30002], "10": [10100, 20001, 30003], "11": [10100, 20001, 30004], "12": [10100, 20001, 30005], "15": [10100, 20002, 30002], "16": [10100, 20002, 30003], "17": [10100, 20002, 30004], "18": [10100, 20002, 30005], "22": [10100, 20003, 30003], "23": [10100, 20003, 30004], "24": [10100, 20003, 30005], "29": [10100, 20004, 30004], "30": [10100, 20004, 30005], "36": [10100, 20005, 30005]}, "push_status": 1})";

  std::vector<int64_t> input{5, 2, 16, 8, 3};
  std::vector<int64_t> output{5, 2, 16, 8, 3};
  ge::DataType in_dtype = ge::DT_UINT8;

  auto opParas = op::TileWithAxis("TileWithAxis");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, in_dtype, ge::FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, in_dtype, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "5 1 16 ");
}
