#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "op_tiling/vector_tiling.h"
#include "op_tiling/norm.h"
#include "graph/utils/op_desc_utils.h"

using namespace std;
using namespace ge;
using namespace optiling;

class NormTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "NormTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "NormTilingTest TearDown" << std::endl;
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
static bool compare_map(const std::unordered_map<T1, T2>& map1, const std::unordered_map<T1, T2>& map2) {
  if (map1.size() != map2.size()) {
    return false;
  }
  for (const auto& it: map1) {
    if (map2.count(it.first) == 0) {
      return false;
    }
    if (map1.at(it.first) != map2.at(it.first)) {
      return false;
    }
  }
  return true;
}

static bool compare_norm_struct(const NormCompileInfo& actual_struct,
                                const NormCompileInfo& expect_struct) {
  if (actual_struct.ori_axis != expect_struct.ori_axis) {
    return false;
  }
  if (actual_struct.core_num != expect_struct.core_num) {
    return false;
  }
  if (actual_struct.min_block_size != expect_struct.min_block_size) {
    return false;
  }
  if (actual_struct.is_keep_dims != expect_struct.is_keep_dims) {
    return false;
  }
  if (actual_struct.max_ub_count != expect_struct.max_ub_count) {
    return false;
  }
  if (actual_struct.workspace_max_ub_count != expect_struct.workspace_max_ub_count) {
    return false;
  }
  if (actual_struct.pad_max_ub_count != expect_struct.pad_max_ub_count) {
    return false;
  }
  if (actual_struct.pad_max_entire_size != expect_struct.pad_max_entire_size) {
    return false;
  }
  if (actual_struct.workspace_type != expect_struct.workspace_type) {
    return false;
  }
  if (actual_struct.workspace_bytes != expect_struct.workspace_bytes) {
    return false;
  }
  if (actual_struct.workspace_diff_count != expect_struct.workspace_diff_count) {
    return false;
  }
  if (!compare_map(actual_struct.norm_vars, expect_struct.norm_vars)) {
    return false;
  }
  if (actual_struct.is_fuse_axis != expect_struct.is_fuse_axis) {
    return false;
  }
  if (actual_struct.is_const != expect_struct.is_const) {
    return false;
  }
  if (actual_struct.is_const_post != expect_struct.is_const_post) {
    return false;
  }
  if (actual_struct.const_tiling_key != expect_struct.const_tiling_key) {
    return false;
  }
  if (actual_struct.const_block_dim != expect_struct.const_block_dim) {
    return false;
  }
  if (actual_struct.const_workspace_size != expect_struct.const_workspace_size) {
    return false;
  }

  return true;
}

TEST_F(NormTilingTest, NormTilingCustomUnsupported) {
  std::string compileInfo = R"({ "_fuse_axis": true, "_ori_axis": [2], "_pattern": "Norm", "_common_info": [32, 16, 1, 16080, 16120, 16080, 128], "_workspace_info": {"_workspace_type": [1], "_workspace_bytes": [4], "_workspace_diff_count": 0}, "_norm_vars": {"40000400": [100, 101, 200, 300]}, "_vars": {"40000400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]}})";
  std::vector<std::vector<int64_t>> inputs {
    {2, 10496, 41}
  };
  std::vector<std::vector<int64_t>> outputs {
    {2, 10496, 41}
  };
  ge::DataType dtype = ge::DT_FLOAT16;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    contruct_tensor(op_desc, inputs[i], dtype);
  }
  for (std::size_t i = 0; i < outputs.size(); i++) {
    contruct_tensor(op_desc, outputs[i], dtype, false);
  }
  ge::Operator op_paras = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  nlohmann::json compile_info = nlohmann::json::parse(compileInfo.c_str());
  optiling::utils::OpRunInfo runInfo;
  optiling::OpInfo c_op_info(inputs, DT_FLOAT);
  std::shared_ptr<AutoTilingCompileInfo> outer_compile_info =
    CreateNormTilingHandler("NormDoTilingWithOpInfo", "Norm", compile_info);
  ASSERT_FALSE(outer_compile_info->DoTiling(op_paras, runInfo, c_op_info));
}

TEST_F(NormTilingTest, ParseFailTest1) {
  std::string compileInfo = R"({ "_fuse_axis": true, "_ori_axis": [2], "_pattern": "Norm", "_common_info": [32, 16, 1, 16080, 16120, 16080], "_workspace_info": {"_workspace_type": [1], "_workspace_bytes": [4], "_workspace_diff_count": 0}, "_norm_vars": {"40000400": [100, 101, 200, 300]}, "_vars": {"40000400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]}})";
  nlohmann::json op_info = nlohmann::json::parse(compileInfo.c_str());

  auto parse_ptr = 
      std::static_pointer_cast<NormTilingHandler>(CreateNormTilingHandler("ParseFailTest1", "Norm", op_info));
  bool is_success = parse_ptr ? true : false;
  ASSERT_FALSE(is_success);
}

TEST_F(NormTilingTest, ParseSuccessTest1) {
  std::string compileInfo = R"({ "_fuse_axis": true, "_ori_axis": [2], "_pattern": "Norm", "_common_info": [32, 16, 1, 16080, 16120, 16080, 128], "_workspace_info": {"_workspace_type": [1], "_workspace_bytes": [4], "_workspace_diff_count": 0}, "_norm_vars": {"40000400": [100, 101, 200, 300]}, "_vars": {"40000400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]}})";
  nlohmann::json op_info = nlohmann::json::parse(compileInfo.c_str());

  auto parse_ptr = 
      std::static_pointer_cast<NormTilingHandler>(CreateNormTilingHandler("ParseFailTest1", "Norm", op_info));
  bool is_success = parse_ptr ? true : false;
  ASSERT_TRUE(is_success);
}

TEST_F(NormTilingTest, ParseTest1) {
  std::string compileInfo = R"({ "_fuse_axis": true, "_ori_axis": [2], "_pattern": "Norm", "_common_info": [32, 16, 1, 16080, 16120, 16080, 128], "_workspace_info": {"_workspace_type": [1], "_workspace_bytes": [4], "_workspace_diff_count": 0}, "_norm_vars": {"40000400": [100, 101, 200, 300]}, "_vars": {"40000400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"]}})";
  nlohmann::json op_info = nlohmann::json::parse(compileInfo.c_str());

  NormCompileInfo actual_struct("norm", op_info);
  NormCompileInfo expect_struct;
  expect_struct.ori_axis = {2};
  expect_struct.core_num = 32;
  expect_struct.min_block_size = 16;
  expect_struct.is_keep_dims = 1;
  expect_struct.max_ub_count = 16080;
  expect_struct.workspace_max_ub_count = 16120;
  expect_struct.pad_max_ub_count = 16080;
  expect_struct.pad_max_entire_size = 128;
  expect_struct.workspace_type = {1};
  expect_struct.workspace_bytes = {4};
  expect_struct.workspace_diff_count = 0;
  expect_struct.norm_vars = {{"40000400", {100, 101, 200, 300}}};
  expect_struct.is_fuse_axis = true;
  ASSERT_TRUE(compare_norm_struct(actual_struct, expect_struct));
}

TEST_F(NormTilingTest, ParseTest2) {
  std::string compileInfo = R"({ "_fuse_axis": false, "_ori_axis": [1], "_pattern": "Norm", "_common_info": [32, 8, 1, 21448, 21496, 16216, 128], "_workspace_info": {"_workspace_type": [1], "_workspace_bytes": [4], "_workspace_diff_count": 0}, "_reduce_shape_known": true, "_const_shape_post": false})";
  nlohmann::json op_info = nlohmann::json::parse(compileInfo.c_str());

  NormCompileInfo actual_struct("norm", op_info);
  NormCompileInfo expect_struct;
  expect_struct.ori_axis = {1};
  expect_struct.core_num = 32;
  expect_struct.min_block_size = 8;
  expect_struct.is_keep_dims = 1;
  expect_struct.max_ub_count = 21448;
  expect_struct.workspace_max_ub_count = 21496;
  expect_struct.pad_max_ub_count = 16216;
  expect_struct.pad_max_entire_size = 128;
  expect_struct.workspace_type = {1};
  expect_struct.workspace_bytes = {4};
  expect_struct.workspace_diff_count = 0;
  expect_struct.is_fuse_axis = false;
  expect_struct.is_const = true;
  expect_struct.is_const_post = false;
  ASSERT_TRUE(compare_norm_struct(actual_struct, expect_struct));
}

TEST_F(NormTilingTest, ParseTest3) {
  std::string compileInfo = R"({ "_fuse_axis": true, "_ori_axis": [3], "_pattern": "Norm", "_common_info": [32, 8, 1, 16216, 16248, 16216, 128], "_workspace_info": {"_workspace_type": [1], "_workspace_bytes": [4], "_workspace_diff_count": 0}, "_reduce_shape_known": true, "_const_shape_post": true, "_const_workspace_size": [], "_const_tiling_key": 10000400, "_block_dims": 32, "_vars": {"10000400": []}})";
  nlohmann::json op_info = nlohmann::json::parse(compileInfo.c_str());

  NormCompileInfo actual_struct("norm", op_info);
  NormCompileInfo expect_struct;
  expect_struct.ori_axis = {3};
  expect_struct.core_num = 32;
  expect_struct.min_block_size = 8;
  expect_struct.is_keep_dims = 1;
  expect_struct.max_ub_count = 16216;
  expect_struct.workspace_max_ub_count = 16248;
  expect_struct.pad_max_ub_count = 16216;
  expect_struct.pad_max_entire_size = 128;
  expect_struct.workspace_type = {1};
  expect_struct.workspace_bytes = {4};
  expect_struct.workspace_diff_count = 0;
  expect_struct.is_fuse_axis = true;
  expect_struct.is_const = true;
  expect_struct.is_const_post = true;
  expect_struct.const_tiling_key = 10000400;
  expect_struct.const_block_dim = 32;
  ASSERT_TRUE(compare_norm_struct(actual_struct, expect_struct));
}

TEST_F(NormTilingTest, TilingTest1) {
  std::vector<std::vector<int64_t>> inputs {
    {2, 10496, 41}
  };
  std::vector<std::vector<int64_t>> outputs {
    {2, 10496, 41}
  };
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

  NormCompileInfo op_compile_info;
  op_compile_info.ori_axis = {2};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 16;
  op_compile_info.is_keep_dims = 1;
  op_compile_info.max_ub_count = 16080;
  op_compile_info.workspace_max_ub_count = 16120;
  op_compile_info.pad_max_ub_count = 16080;
  op_compile_info.pad_max_entire_size = 128;
  op_compile_info.workspace_type = {1};
  op_compile_info.workspace_bytes = {4};
  op_compile_info.workspace_diff_count = 0;
  op_compile_info.norm_vars = {{"40000400", {100, 101, 200, 300}}};
  op_compile_info.is_fuse_axis = true;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "20992 41 656 328 ");
}

TEST_F(NormTilingTest, TilingTest2) {
  std::vector<std::vector<int64_t>> inputs {
    {16, 5, 15003}
  };
  std::vector<std::vector<int64_t>> outputs {
    {16, 5, 15003}
  };
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

  NormCompileInfo op_compile_info;
  op_compile_info.ori_axis = {2};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.is_keep_dims = 1;
  op_compile_info.max_ub_count = 12896;
  op_compile_info.workspace_max_ub_count = 12896;
  op_compile_info.pad_max_ub_count = 12896;
  op_compile_info.pad_max_entire_size = 128;
  op_compile_info.workspace_type = {1};
  op_compile_info.workspace_bytes = {4};
  op_compile_info.workspace_diff_count = 0;
  op_compile_info.norm_vars = {{"100400", {100, 101, 200, 300}}};
  op_compile_info.is_fuse_axis = true;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(runInfo.GetBlockDim(), 10);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "80 15003 8 7502 ");
}

TEST_F(NormTilingTest, TilingTest3) {
  std::vector<std::vector<int64_t>> inputs {
    {16, 5, 15003}
  };
  std::vector<std::vector<int64_t>> outputs {
    {16, 5, 15003}
  };
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

  NormCompileInfo op_compile_info;
  op_compile_info.ori_axis = {1};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.is_keep_dims = 1;
  op_compile_info.max_ub_count = 16336;
  op_compile_info.workspace_max_ub_count = 16360;
  op_compile_info.pad_max_ub_count = 16336;
  op_compile_info.pad_max_entire_size = 128;
  op_compile_info.workspace_type = {1};
  op_compile_info.workspace_bytes = {4};
  op_compile_info.workspace_diff_count = 0;
  op_compile_info.norm_vars = {{"2100900", {100, 101, 102, 200, 300}}};
  op_compile_info.is_fuse_axis = true;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "16 5 15003 7504 2 ");
}

TEST_F(NormTilingTest, TilingTest4) {
  std::vector<std::vector<int64_t>> inputs {
    {31, 2400}
  };
  std::vector<std::vector<int64_t>> outputs {
    {31, 2400}
  };
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

  NormCompileInfo op_compile_info;
  op_compile_info.ori_axis = {0};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.is_keep_dims = 1;
  op_compile_info.max_ub_count = 16336;
  op_compile_info.workspace_max_ub_count = 16360;
  op_compile_info.pad_max_ub_count = 16336;
  op_compile_info.pad_max_entire_size = 128;
  op_compile_info.workspace_type = {1};
  op_compile_info.workspace_bytes = {4};
  op_compile_info.workspace_diff_count = 0;
  op_compile_info.norm_vars = {{"1000500", {100, 101, 200, 300}}};
  op_compile_info.is_fuse_axis = true;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(runInfo.GetBlockDim(), 30);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "31 2400 80 31 ");
}

TEST_F(NormTilingTest, TilingTest5) {
  std::vector<std::vector<int64_t>> inputs {
    {1968, 3, 3}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1968, 3, 3}
  };
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

  NormCompileInfo op_compile_info;
  op_compile_info.ori_axis = {0, 2};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.is_keep_dims = 1;
  op_compile_info.max_ub_count = 16216;
  op_compile_info.workspace_max_ub_count = 16248;
  op_compile_info.pad_max_ub_count = 16216;
  op_compile_info.pad_max_entire_size = 128;
  op_compile_info.workspace_type = {1, 0, 0};
  op_compile_info.workspace_bytes = {4, 4, 4};
  op_compile_info.workspace_diff_count = 2;
  op_compile_info.norm_vars = {{"21001200", {100, 101, 102, 200, 300}}};
  op_compile_info.is_fuse_axis = true;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1968 3 3 3 677 ");
}

TEST_F(NormTilingTest, TilingTest6) {
  std::vector<std::vector<int64_t>> inputs {
    {1, 7, 543, 76}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1, 7, 543, 76}
  };
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

  NormCompileInfo op_compile_info;
  op_compile_info.ori_axis = {3};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.is_keep_dims = 1;
  op_compile_info.max_ub_count = 16216;
  op_compile_info.workspace_max_ub_count = 16248;
  op_compile_info.pad_max_ub_count = 16216;
  op_compile_info.pad_max_entire_size = 128;
  op_compile_info.workspace_type = {1};
  op_compile_info.workspace_bytes = {4};
  op_compile_info.workspace_diff_count = 0;
  op_compile_info.is_fuse_axis = true;
  op_compile_info.is_const = true;
  op_compile_info.is_const_post = true;
  op_compile_info.const_tiling_key = 10000400;
  op_compile_info.const_block_dim = 32;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
}

TEST_F(NormTilingTest, TilingTest7) {
  std::vector<std::vector<int64_t>> inputs {
    {2, 10, 2, 3, 7}
  };
  std::vector<std::vector<int64_t>> outputs {
    {2, 10, 2, 3, 7}
  };
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

  NormCompileInfo op_compile_info;
  op_compile_info.ori_axis = {0, 1, 2, 3, 4};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.is_keep_dims = 1;
  op_compile_info.max_ub_count = 21624;
  op_compile_info.workspace_max_ub_count = 21664;
  op_compile_info.pad_max_ub_count = 16216;
  op_compile_info.pad_max_entire_size = 128;
  op_compile_info.workspace_type = {1};
  op_compile_info.workspace_bytes = {4};
  op_compile_info.workspace_diff_count = 0;
  op_compile_info.norm_vars = {{"62000", {100, 101, 102, 103, 104, 300}}};
  op_compile_info.is_fuse_axis = false;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 10 2 3 7 2 ");
}

TEST_F(NormTilingTest, TilingTest8) {
  std::vector<std::vector<int64_t>> inputs {
    {2, 3}
  };
  std::vector<std::vector<int64_t>> outputs {
    {2, 3}
  };
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

  NormCompileInfo op_compile_info;
  op_compile_info.ori_axis = {1};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.is_keep_dims = 1;
  op_compile_info.max_ub_count = 21448;
  op_compile_info.workspace_max_ub_count = 21496;
  op_compile_info.pad_max_ub_count = 16216;
  op_compile_info.pad_max_entire_size = 128;
  op_compile_info.workspace_type = {1};
  op_compile_info.workspace_bytes = {4};
  op_compile_info.workspace_diff_count = 0;
  op_compile_info.is_fuse_axis = true;
  op_compile_info.is_const = true;
  op_compile_info.is_const_post = false;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 2 0 2 ");
}

TEST_F(NormTilingTest, TilingTest9) {
  std::vector<std::vector<int64_t>> inputs {
    {2, 3}
  };
  std::vector<std::vector<int64_t>> outputs {
    {2, 3}
  };
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

  NormCompileInfo op_compile_info;
  op_compile_info.ori_axis = {1};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.is_keep_dims = 1;
  op_compile_info.max_ub_count = 21448;
  op_compile_info.workspace_max_ub_count = 21496;
  op_compile_info.pad_max_ub_count = 16216;
  op_compile_info.pad_max_entire_size = 128;
  op_compile_info.workspace_type = {1};
  op_compile_info.workspace_bytes = {4};
  op_compile_info.workspace_diff_count = 0;
  op_compile_info.is_fuse_axis = false;
  op_compile_info.is_const = true;
  op_compile_info.is_const_post = false;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 2 0 2 ");
}

