#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "op_tiling/vector_tiling.h"
#include "op_tiling/norm.h"
#include "graph/utils/op_desc_utils.h"
#include "op_tiling/tiling_handler.h"

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

  if (actual_struct.ori_reduce_axis != expect_struct.ori_reduce_axis) {
    return false;
  }
  if (actual_struct.ori_broadcast_axis != expect_struct.ori_broadcast_axis) {
    return false;
  }
  if (actual_struct.is_broadcast_axis_known != expect_struct.is_broadcast_axis_known) {
    return false;
  }
  if (actual_struct.input_type != expect_struct.input_type) {
    return false;
  }
  if (actual_struct.exist_output_after_reduce != expect_struct.exist_output_after_reduce) {
    return false;
  }
  if (actual_struct.exist_workspace_after_reduce != expect_struct.exist_workspace_after_reduce) {
    return false;
  }
  if (!compare_map(actual_struct.available_ub_size, expect_struct.available_ub_size)) {
    return false;
  }
  if (actual_struct.core_num != expect_struct.core_num) {
    return false;
  }
  if (actual_struct.min_block_size != expect_struct.min_block_size) {
    return false;
  }
  if (actual_struct.transpose_max_entire_size != expect_struct.transpose_max_entire_size) {
    return false;
  }
  if (!compare_map(actual_struct.workspace_info, expect_struct.workspace_info)) {
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
  if (!compare_map(actual_struct.const_block_dims, expect_struct.const_block_dims)) {
    return false;
  }
  if (actual_struct.reduce_attr_name != expect_struct.reduce_attr_name) {
    return false;
  }
  if (actual_struct.is_reduce_attr_is_int != expect_struct.is_reduce_attr_is_int) {
    return false;
  }
  if (actual_struct.reduce_axis_type != expect_struct.reduce_axis_type) {
    return false;
  }
  if (actual_struct.broadcast_axis_type != expect_struct.broadcast_axis_type) {
    return false;
  }
  if (actual_struct.exist_vc_unsupported_type != expect_struct.exist_vc_unsupported_type) {
    return false;
  }
  return true;
}

TEST_F(NormTilingTest, NormTilingCustomUnsupported) {
  std::string compileInfo = R"({ "_fuse_axis": true, "_input_type": [0], "_ori_reduce_axis": [2], "_pattern": "Norm", "_common_info": [32, 16, 128], "_available_ub_size": {"4000": [15792, 16120, 15792, 15792]}, "_exist_workspace_after_reduce": false, "_exist_output_after_reduce": false, "_workspace_info": {"400400000": [32]}, "_norm_vars": {"400400000": [20000, 20001, 30000, 40000]}})";
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
  std::shared_ptr<AutoTilingHandler> outer_compile_info =
    CreateNormTilingHandler("NormDoTilingWithOpInfo", "Norm", compile_info);
  ASSERT_FALSE(outer_compile_info->DoTiling(op_paras, runInfo, c_op_info));
}

TEST_F(NormTilingTest, ParseFailTest1) {
  std::string compileInfo = R"({ "_fuse_axis": true, "_input_type": [0], "_ori_reduce_axis": [2], "_pattern": "Norm", "_common_info": [32, 16], "_available_ub_size": {"4000": [15792, 16120, 15792]}, "_exist_workspace_after_reduce": false, "_exist_output_after_reduce": false, "_workspace_info": {"200400000": [32]}, "_norm_vars": {"200400000": [20000, 20001, 30000, 40000]}})";
  nlohmann::json op_info = nlohmann::json::parse(compileInfo.c_str());

  auto parse_ptr = 
      std::static_pointer_cast<NormTilingHandler>(CreateNormTilingHandler("ParseFailTest1", "Norm", op_info));
  bool is_success = parse_ptr ? true : false;
  ASSERT_FALSE(is_success);
}

TEST_F(NormTilingTest, ParseSuccessTest1) {
  std::string compileInfo = R"({ "_fuse_axis": true, "_input_type": [0], "_ori_reduce_axis": [2], "_pattern": "Norm", "_common_info": [32, 16, 128], "_available_ub_size": {"4000": [15792, 16120, 15792, 15792]}, "_exist_workspace_after_reduce": false, "_exist_output_after_reduce": false, "_workspace_info": {"400400000": [32]}, "_norm_vars": {"400400000": [20000, 20001, 30000, 40000]}})";
  nlohmann::json op_info = nlohmann::json::parse(compileInfo.c_str());

  auto parse_ptr = 
      std::static_pointer_cast<NormTilingHandler>(CreateNormTilingHandler("ParseSuccessTest1", "Norm", op_info));
  bool is_success = parse_ptr ? true : false;
  ASSERT_TRUE(is_success);
}

TEST_F(NormTilingTest, ParseTest1) {
  std::string compileInfo = R"({ "_fuse_axis": true, "_input_type": [0], "_ori_reduce_axis": [2], "_pattern": "Norm", "_common_info": [32, 16, 128], "_available_ub_size": {"4000": [15792, 16120, 15792, 15792]}, "_exist_workspace_after_reduce": false, "_exist_output_after_reduce": false, "_workspace_info": {"200400000": [32]}, "_norm_vars": {"200400000": [20000, 20001, 30000, 40000]}})";
  nlohmann::json op_info = nlohmann::json::parse(compileInfo.c_str());

  NormCompileInfo actual_struct("norm", op_info);
  NormCompileInfo expect_struct;
  expect_struct.input_type = {0};
  expect_struct.ori_reduce_axis = {2};
  expect_struct.core_num = 32;
  expect_struct.min_block_size = 16;
  expect_struct.transpose_max_entire_size = 128;
  expect_struct.exist_output_after_reduce = false;
  expect_struct.exist_workspace_after_reduce = false;
  expect_struct.available_ub_size = {{4000, {15792, 16120, 15792, 15792}}};
  expect_struct.workspace_info = {{200400000, {32}}};
  expect_struct.norm_vars = {{200400000, {20000, 20001, 30000, 40000}}};
  expect_struct.is_fuse_axis = true;
  ASSERT_TRUE(compare_norm_struct(actual_struct, expect_struct));
}

TEST_F(NormTilingTest, ParseTest2) {
  std::string compileInfo = R"({ "_fuse_axis": false, "_input_type": [0], "_ori_reduce_axis": [1], "_ori_broadcast_axis": [1], "_pattern": "Norm", "_common_info": [32, 8, 128], "_available_ub_size": {"4000": [15792, 16120, 15792, 15792]}, "_is_const": true, "_const_shape_post": false, "_exist_workspace_after_reduce": false, "_exist_output_after_reduce": false})";
  nlohmann::json op_info = nlohmann::json::parse(compileInfo.c_str());

  NormCompileInfo actual_struct("norm", op_info);
  NormCompileInfo expect_struct;
  expect_struct.input_type = {0};
  expect_struct.ori_reduce_axis = {1};
  expect_struct.ori_broadcast_axis = {1};
  expect_struct.is_broadcast_axis_known = true;
  expect_struct.core_num = 32;
  expect_struct.min_block_size = 8;
  expect_struct.transpose_max_entire_size = 128;
  expect_struct.exist_output_after_reduce = false;
  expect_struct.exist_workspace_after_reduce = false;
  expect_struct.available_ub_size = {{4000, {15792, 16120, 15792, 15792}}};
  expect_struct.is_fuse_axis = false;
  expect_struct.is_const = true;
  expect_struct.is_const_post = false;
  ASSERT_TRUE(compare_norm_struct(actual_struct, expect_struct));
}

TEST_F(NormTilingTest, ParseTest3) {
  std::string compileInfo = R"({"_exist_vc_unsupported_type": false, "reduce_axis_attr_name": "axis", "reduce_axis_attr_dtype": "ListInt", "_reduce_axis_type": 9, "_broadcast_axis_type_list": [1, 2], "_fuse_axis": false, "_input_type": [0], "_ori_reduce_axis": [1], "_ori_broadcast_axis": [1], "_pattern": "Norm", "_common_info": [32, 8, 128], "_available_ub_size": {"4000": [15792, 16120, 15792, 15792]}, "_is_const": true, "_const_shape_post": true, "_const_block_dims": {"4000": 25}, "_exist_workspace_after_reduce": false, "_exist_output_after_reduce": false})";
  nlohmann::json op_info = nlohmann::json::parse(compileInfo.c_str());

  NormCompileInfo actual_struct("norm", op_info);
  NormCompileInfo expect_struct;
  expect_struct.reduce_attr_name = "axis";
  expect_struct.is_reduce_attr_is_int = false;
  expect_struct.reduce_axis_type = 9;
  expect_struct.broadcast_axis_type = {1, 2};
  expect_struct.input_type = {0};
  expect_struct.ori_reduce_axis = {1};
  expect_struct.ori_broadcast_axis = {1};
  expect_struct.is_broadcast_axis_known = true;
  expect_struct.core_num = 32;
  expect_struct.min_block_size = 8;
  expect_struct.transpose_max_entire_size = 128;
  expect_struct.exist_output_after_reduce = false;
  expect_struct.exist_workspace_after_reduce = false;
  expect_struct.available_ub_size = {{4000, {15792, 16120, 15792, 15792}}};
  expect_struct.is_fuse_axis = false;
  expect_struct.is_const = true;
  expect_struct.is_const_post = true;
  expect_struct.const_block_dims = {{4000, 25}};
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
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {2};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 16;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{4000, {15792, 16120, 15792, 15792}}};
  op_compile_info.workspace_info = {{400400000, {32}}};
  op_compile_info.norm_vars = {{400400000, {20000, 20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "20992 41 656 328 ");
}

TEST_F(NormTilingTest, TilingTest2) {
  std::vector<std::vector<int64_t>> inputs {
    {16, 5, 30000}
  };
  std::vector<std::vector<int64_t>> outputs {
    {16, 5, 30000}
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
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {2};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 16;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{4000, {15792, 16120, 15792, 15792}}};
  op_compile_info.workspace_info = {{400001, {4}}};
  op_compile_info.norm_vars = {{400001, {20000, 20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(runInfo.GetBlockDim(), 27);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "80 30000 3 15000 ");
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
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {1};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{9000, {21784, 16360, 16336, 16336}}};
  op_compile_info.workspace_info = {{900021, {4}}};
  op_compile_info.norm_vars = {{900021, {20000, 20001, 20002, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "16 5 15003 7504 2 ");
}

TEST_F(NormTilingTest, TilingTest4) {
  std::vector<std::vector<int64_t>> inputs {
    {10, 1, 7}
  };
  std::vector<std::vector<int64_t>> outputs {
    {10, 1, 7}
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
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {2};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{4000, {21448, 16120, 16088, 16088}}};
  op_compile_info.workspace_info = {{400400000, {32}}};
  op_compile_info.norm_vars = {{400400000, {20000, 20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(runInfo.GetBlockDim(), 5);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "10 7 2 2 ");
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
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {0, 2};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = true;
  op_compile_info.available_ub_size = {{12000, {16216, 13000, 16216, 16216}}};
  op_compile_info.workspace_info = {{101200010, {4, -4, -4}}};
  op_compile_info.norm_vars = {{101200010, {20000, 20001, 20002, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1968 3 3 3 541 ");
}

TEST_F(NormTilingTest, TilingTest6) {
  std::vector<std::vector<int64_t>> inputs {
    {7, 543, 76}
  };
  std::vector<std::vector<int64_t>> outputs {
    {7, 543, 76}
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
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {1};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{9000, {21784, 16360, 16336, 16336}}};
  op_compile_info.is_fuse_axis = false;
  op_compile_info.is_const = true;
  op_compile_info.is_const_post = false;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
}

TEST_F(NormTilingTest, TilingTest7) {
  std::vector<std::vector<int64_t>> inputs {
    {7, 543, 76}
  };
  std::vector<std::vector<int64_t>> outputs {
    {7, 543, 76}
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
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {1};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{9000, {21784, 16360, 16336, 16336}}};
  op_compile_info.is_fuse_axis = false;
  op_compile_info.is_const = true;
  op_compile_info.is_const_post = true;
  op_compile_info.const_block_dims = {{9000, 28}};
  op_compile_info.workspace_info = {{9000, {4}}};

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(runInfo.GetBlockDim(), 28);
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
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {1};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{4000, {15792, 16120, 15792, 15792}}};
  op_compile_info.workspace_info = {{400400000, {32}}};
  op_compile_info.norm_vars = {{400400000, {20000, 20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 3 2 2 ");
}

TEST_F(NormTilingTest, TilingTest9) {
  std::vector<std::vector<int64_t>> inputs {
    {32, 32, 32}
  };
  std::vector<std::vector<int64_t>> outputs {
    {32, 32, 32}
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
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {0, 1, 2};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{14000, {16216, 13000, 16216, 16216}}};
  op_compile_info.workspace_info = {{1400090, {4}}};
  op_compile_info.norm_vars = {{1400090, {20000, 20001, 20002, 40000}}};
  op_compile_info.is_fuse_axis = false;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "32 32 32 11 ");
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
}

TEST_F(NormTilingTest, TilingTest10) {
  std::vector<std::vector<int64_t>> inputs {
    {1000, 1}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1000, 1}
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
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {1};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{5000, {21784, 16360, 16336, 16336}}};
  op_compile_info.workspace_info = {{500011, {32}}};
  op_compile_info.norm_vars = {{500011, {20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1000 32 32 ");
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
}

TEST_F(NormTilingTest, TilingTest11) {
  std::vector<std::vector<int64_t>> inputs {
    {1000, 1}, {1, 2000}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1000, 1}, {1000, 2000}
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
  op_compile_info.input_type = {1, 2};
  op_compile_info.ori_reduce_axis = {1};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = true;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{4012, {10552, 10744, 10552, 10552}}};
  op_compile_info.workspace_info = {{401201, {4, 4, 4, 4, 4}}};
  op_compile_info.norm_vars = {{401201, {20000, 20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1000 2000 32 2000 ");
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
}

TEST_F(NormTilingTest, TilingTest12) {
  std::vector<std::vector<int64_t>> inputs {
    {3, 1968, 3, 3}
  };
  std::vector<std::vector<int64_t>> outputs {
    {3, 1968, 3, 3}
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
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {1, 3};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = true;
  op_compile_info.available_ub_size = {{20000, {16216, 13000, 16216, 16216}}};
  op_compile_info.workspace_info = {{102000001, {4, -4, -4}}};
  op_compile_info.norm_vars = {{102000001, {20000, 20001, 20002, 20003, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(runInfo.GetBlockDim(), 3);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "3 1968 3 3 1 541 ");
}

TEST_F(NormTilingTest, TilingTest13) {
  std::vector<std::vector<int64_t>> inputs {
    {1968, 32, 512}, {512}, {512}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1968, 32, 512}, {1968, 32, 1}, {1968, 32, 1}
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
  op_compile_info.input_type = {0, 1, 1};
  op_compile_info.ori_reduce_axis = {2};
  op_compile_info.ori_broadcast_axis = {0, 1};
  op_compile_info.is_broadcast_axis_known = true;
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = true;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{4005, {21152, 16120, 15864, 15864}}};
  op_compile_info.workspace_info = {{1300400500, {32}}};
  op_compile_info.norm_vars = {{1300400500, {20000, 20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "62976 512 1968 41 ");
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
}

TEST_F(NormTilingTest, TilingTest14) {
  std::vector<std::vector<int64_t>> inputs {
    {95, 10, 1, 87, 16}
  };
  std::vector<std::vector<int64_t>> outputs {
    {95, 10, 1, 87, 16}
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
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {1, 3, 4};
  op_compile_info.ori_disable_fuse_axes = {1, 4};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 16;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = true;
  op_compile_info.available_ub_size = {{42000, {15920, 12840, 15920, 15920}}};
  op_compile_info.workspace_info = {{304200000, {32}}};
  op_compile_info.norm_vars = {{304200000, {20000, 20003, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "95 87 3 1 ");
}

TEST_F(NormTilingTest, TilingTest15) {
  std::vector<std::vector<int64_t>> inputs {
    {1, 23, 512}, {1, 23, 512}, {1, 23, 1}, {1, 23, 1}, {512}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1, 23, 512}, {1, 23, 512}
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
  op_compile_info.input_type = {0, 0, 1, 1, 2};
  op_compile_info.ori_reduce_axis = {2};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 16;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{4012, {9056, 7048, 9056, 9056}}};
  op_compile_info.workspace_info = {{300401200, {32, 32}}};
  op_compile_info.norm_vars = {{300401200, {20000, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(runInfo.GetBlockDim(), 23);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "23 1 1 ");
}

TEST_F(NormTilingTest, TilingTest16) {
  std::vector<std::vector<int64_t>> inputs {
    {21, 93, 143}, {21, 93, 143}, {21, 1, 1}, {21, 1, 1}, {143}
  };
  std::vector<std::vector<int64_t>> outputs {
    {21, 93, 143}, {21, 93, 143}
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
  op_compile_info.input_type = {0, 0, 1, 1, 2};
  op_compile_info.ori_reduce_axis = {1, 2};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 16;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{10044, {9056, 7048, 9056, 9056}}, {12044, {9056, 7048, 9056, 9056}}};
  op_compile_info.workspace_info = {{1004401, {4, 4}}};
  op_compile_info.norm_vars = {{1004401, {20000, 20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(runInfo.GetBlockDim(), 21);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "21 93 1 47 ");
}

TEST_F(NormTilingTest, TilingTest17) {
  std::vector<std::vector<int64_t>> inputs {
    {11, 20, 512}, {11, 20, 1}, {11, 20, 1}, {512}, {512}
  };
  std::vector<std::vector<int64_t>> outputs {
    {11, 20, 512}, {11, 20, 512}
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

  std::vector<std::vector<int32_t>> reduce_axis{{-1}};
  OpInfo op_info(inputs, dtype, reduce_axis);

  NormCompileInfo op_compile_info;
  op_compile_info.input_type = {0, 0, 1, 1, 2};
  op_compile_info.reduce_axis_type = 3;
  op_compile_info.broadcast_axis_type = {1, 2};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 16;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{4012, {9040, 7256, 9056, 9056}}};
  op_compile_info.workspace_info = {{300401200, {32, 32}}};
  op_compile_info.norm_vars = {{300401200, {20000, 20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling(op_info));
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "220 512 7 7 ");
}

TEST_F(NormTilingTest, TilingTest18) {
  std::vector<std::vector<int64_t>> inputs {
    {11, 20, 512}, {512}, {512}
  };
  std::vector<std::vector<int64_t>> outputs {
    {11, 20, 512}, {11, 20, 1}, {11, 20, 1}
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
  op_paras.SetAttr("axis", -1);

  optiling::utils::OpRunInfo runInfo;

  NormCompileInfo op_compile_info;
  op_compile_info.reduce_attr_name = "axis";
  op_compile_info.is_reduce_attr_is_int = true;
  op_compile_info.input_type = {0, 1, 1};
  op_compile_info.reduce_axis_type = 3;
  op_compile_info.broadcast_axis_type = {2};
  op_compile_info.ori_broadcast_axis = {0, 1};
  op_compile_info.is_broadcast_axis_known = true;
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 16;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = true;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{4005, {21080, 16350, 15808, 15808}}};
  op_compile_info.workspace_info = {{1300400500, {32}}};
  op_compile_info.norm_vars = {{1300400500, {20000, 20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "220 512 16 16 ");
  EXPECT_EQ(runInfo.GetBlockDim(), 14);
}

TEST_F(NormTilingTest, TilingTest19) {
  std::vector<std::vector<int64_t>> inputs {
    {11, 20, 512}
  };
  std::vector<std::vector<int64_t>> outputs {
    {11, 20, 512}
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
  op_paras.SetAttr("axis", {-1});
  optiling::utils::OpRunInfo runInfo;

  NormCompileInfo op_compile_info;
  op_compile_info.input_type = {0};
  op_compile_info.reduce_attr_name = "axis";
  op_compile_info.is_reduce_attr_is_int = false;
  op_compile_info.reduce_axis_type = 9;
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 16;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{4000, {15824, 16360, 15824, 15824}}};
  op_compile_info.workspace_info = {{300400000, {32}}};
  op_compile_info.norm_vars = {{300400000, {20000, 20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "220 512 7 7 ");
}

TEST_F(NormTilingTest, TilingTest20) {
  std::vector<std::vector<int64_t>> inputs {
    {1968, 32, 512}, {512}, {512}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1968, 32, 512}, {1968, 32, 1}, {1968, 32, 1}
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
  op_paras.SetAttr("alpha", 2);

  NormCompileInfo op_compile_info;
  op_compile_info.input_type = {0, 1, 1};
  op_compile_info.ori_reduce_axis = {2};
  op_compile_info.reduce_axis_type = 3;
  op_compile_info.ori_broadcast_axis = {0, 1};
  op_compile_info.is_broadcast_axis_known = true;
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = true;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{4005, {21152, 16120, 15864, 15864}}};
  op_compile_info.workspace_info = {{1300400500, {32}}};
  op_compile_info.norm_vars = {{1300400500, {20000, 20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;
  std::vector<VarAttr> var_attr_list;
  std::string var_attr_list_compileInfo = R"({"_var_attr_mode":0,"_var_attrs": [{"length":1,"name":"alpha","type":"int32","src_type":"int32"}]})";
  op_compile_info.varAttrWrap.ParseVarAttr(nlohmann::json::parse(var_attr_list_compileInfo));

  optiling::Norm norm("norm", op_paras, op_compile_info, runInfo);
  ASSERT_TRUE(norm.DoTiling());
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "62976 512 1968 41 2 ");
  EXPECT_EQ(runInfo.GetBlockDim(), 32);
}