#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "graph/utils/op_desc_utils.h"
#include "op_tiling/vector_tiling.h"
#include "op_tiling/norm.h"
#include "op_tiling/tiling_handler.h"
//#include "op_tiling/vector_tiling_rt2.h"
#include "common_autotiling_util.h"

using namespace std;
using namespace ge;
using namespace optiling;

class NormTilingRt2Test : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "NormTilingRt2Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "NormTilingRt2Test TearDown" << std::endl;
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
  if (!compare_map(actual_struct.block_size_map, expect_struct.block_size_map)) {
    return false;
  }

  return true;
}

TEST_F(NormTilingRt2Test, ParseFailTest1) {
  std::string compileInfo = R"({"_pattern": "Norm", "_fuse_axis": true, "_input_type": [0], "_ori_reduce_axis": [2], "_pattern": "Norm", "_common_info": [32, 16], "_available_ub_size": {"4000": [15792, 16120, 15792]}, "_exist_workspace_after_reduce": false, "_exist_output_after_reduce": false, "_workspace_info": {"200400000": [32]}, "_norm_vars": {"200400000": [20000, 20001, 30000, 40000]}})";

  AutoTilingTest test;
  NormCompileInfo op_compile_info;
  ASSERT_FALSE(test.TestParse(compileInfo, &op_compile_info));

}

TEST_F(NormTilingRt2Test, ParseSuccessTest1) {
  std::string compileInfo = R"({ "_fuse_axis": true, "_input_type": [0], "_ori_reduce_axis": [2], "_pattern": "Norm", "_common_info": [32, 16, 128], "_available_ub_size": {"4000": [15792, 16120, 15792, 15792]}, "_exist_workspace_after_reduce": false, "_exist_output_after_reduce": false, "_workspace_info": {"400400000": [32]}, "_norm_vars": {"400400000": [20000, 20001, 30000, 40000]}})";

  AutoTilingTest test;
  NormCompileInfo op_compile_info;
  ASSERT_TRUE(test.TestParse(compileInfo, &op_compile_info));
}

TEST_F(NormTilingRt2Test, ParseTest1) {
  std::string compileInfo = R"({ "_fuse_axis": true, "_input_type": [0], "_ori_reduce_axis": [2], "_pattern": "Norm", "_common_info": [32, 16, 128], "_available_ub_size": {"4000": [15792, 16120, 15792, 15792]}, "_exist_workspace_after_reduce": false, "_block_size": {"4000": 16}, "_exist_output_after_reduce": false, "_workspace_info": {"200400000": [32]}, "_norm_vars": {"200400000": [20000, 20001, 30000, 40000]}})";
  NormCompileInfo actual_struct;
  AutoTilingTest test;
  bool ret = test.TestParse(compileInfo, &actual_struct);

  NormCompileInfo expect_struct;
  expect_struct.input_type = {0};
  expect_struct.ori_reduce_axis = {2};
  expect_struct.core_num = 32;
  expect_struct.min_block_size = 16;
  expect_struct.transpose_max_entire_size = 128;
  expect_struct.exist_output_after_reduce = false;
  expect_struct.exist_workspace_after_reduce = false;
  expect_struct.available_ub_size = {{4000, {15792, 16120, 15792, 15792}}};
  expect_struct.block_size_map = {{4000, 16}};
  expect_struct.workspace_info = {{200400000, {32}}};
  expect_struct.norm_vars = {{200400000, {20000, 20001, 30000, 40000}}};
  expect_struct.is_fuse_axis = true;
  ASSERT_TRUE(ret);
  ASSERT_TRUE(compare_norm_struct(actual_struct, expect_struct));
}

TEST_F(NormTilingRt2Test, ParseTest2) {
  std::string compileInfo = R"({ "_fuse_axis": false, "_input_type": [0], "_ori_reduce_axis": [1], "_ori_broadcast_axis": [1], "_pattern": "Norm", "_common_info": [32, 8, 128], "_available_ub_size": {"4000": [15792, 16120, 15792, 15792]}, "_is_const": true, "_const_shape_post": false, "_exist_workspace_after_reduce": false, "_exist_output_after_reduce": false})";
  NormCompileInfo actual_struct;
  AutoTilingTest test;
  bool ret = test.TestParse(compileInfo, &actual_struct);

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
  ASSERT_TRUE(ret);
  ASSERT_TRUE(compare_norm_struct(actual_struct, expect_struct));
}

TEST_F(NormTilingRt2Test, ParseTest3) {
  std::string compileInfo = R"({"_exist_vc_unsupported_type": false, "reduce_axis_attr_name": "axis", "reduce_axis_attr_dtype": "ListInt", "_reduce_axis_type": 9, "_broadcast_axis_type_list": [1, 2], "_fuse_axis": false, "_input_type": [0], "_ori_reduce_axis": [1], "_ori_broadcast_axis": [1], "_pattern": "Norm", "_common_info": [32, 8, 128], "_available_ub_size": {"4000": [15792, 16120, 15792, 15792]}, "_is_const": true, "_const_shape_post": true, "_const_block_dims": {"4000": 25}, "_exist_workspace_after_reduce": false, "_exist_output_after_reduce": false})";
  NormCompileInfo actual_struct;
  AutoTilingTest test;
  bool ret = test.TestParse(compileInfo, &actual_struct);

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
  ASSERT_TRUE(ret);
  ASSERT_TRUE(compare_norm_struct(actual_struct, expect_struct));
}

TEST_F(NormTilingRt2Test, TilingTest1) {
  std::vector<std::vector<int64_t>> inputs {
    {2, 10496, 41}
  };
  std::vector<std::vector<int64_t>> outputs {
    {2, 10496, 41}
  };
  ge::DataType dtype = ge::DT_FLOAT16;

  NormCompileInfo op_compile_info;
  op_compile_info.pattern = SchPattern::NORM;
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {2};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 16;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{4000, {15792, 16120, 15792, 15792}}};
  op_compile_info.block_size_map = {{4000, 16}};
  op_compile_info.workspace_info = {{400400000, {32}}};
  op_compile_info.norm_vars = {{400400000, {20000, 20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  test.SetCompileInfo(&op_compile_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "20992, 41, 656, 328";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
}

TEST_F(NormTilingRt2Test, TilingTest2) {
  std::vector<std::vector<int64_t>> inputs {
    {16, 5, 30000}
  };
  std::vector<std::vector<int64_t>> outputs {
    {16, 5, 30000}
  };
  ge::DataType dtype = ge::DT_FLOAT16;

  NormCompileInfo op_compile_info;
  op_compile_info.pattern = SchPattern::NORM;
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {2};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 16;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{4000, {15792, 16120, 15792, 15792}}};
  op_compile_info.block_size_map = {{4000, 16}};
  op_compile_info.workspace_info = {{400001, {4}}};
  op_compile_info.norm_vars = {{400001, {20000, 20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  test.SetCompileInfo(&op_compile_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "80, 30000, 3, 15000";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 27);
}

TEST_F(NormTilingRt2Test, TilingTest3) {
  std::vector<std::vector<int64_t>> inputs {
    {16, 5, 15003}
  };
  std::vector<std::vector<int64_t>> outputs {
    {16, 5, 15003}
  };
  ge::DataType dtype = ge::DT_FLOAT;

  NormCompileInfo op_compile_info;
  op_compile_info.pattern = SchPattern::NORM;
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {1};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{9000, {21784, 16360, 16336, 16336}}};
  op_compile_info.block_size_map = {{9000, 8}};
  op_compile_info.workspace_info = {{900022, {4}}};
  op_compile_info.norm_vars = {{900022, {20000, 20001, 20002, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  test.SetCompileInfo(&op_compile_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "16, 5, 15003, 7504, 3752";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
}

TEST_F(NormTilingRt2Test, TilingTest4) {
  std::vector<std::vector<int64_t>> inputs {
    {10, 1, 7}
  };
  std::vector<std::vector<int64_t>> outputs {
    {10, 1, 7}
  };
  ge::DataType dtype = ge::DT_FLOAT;

  NormCompileInfo op_compile_info;
  op_compile_info.pattern = SchPattern::NORM;
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {2};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{4000, {21448, 16120, 16088, 16088}}};
  op_compile_info.block_size_map = {{4000, 8}};
  op_compile_info.workspace_info = {{400400000, {32}}};
  op_compile_info.norm_vars = {{400400000, {20000, 20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  test.SetCompileInfo(&op_compile_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "10, 7, 2, 2";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 5);
}

TEST_F(NormTilingRt2Test, TilingTest5) {
  std::vector<std::vector<int64_t>> inputs {
    {1968, 3, 3}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1968, 3, 3}
  };
  ge::DataType dtype = ge::DT_FLOAT;

  NormCompileInfo op_compile_info;
  op_compile_info.pattern = SchPattern::NORM;
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {0, 2};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = true;
  op_compile_info.available_ub_size = {{12000, {16216, 13000, 16216, 16216}}};
  op_compile_info.block_size_map = {{12000, 8}};
  op_compile_info.workspace_info = {{101200010, {4, -4, -4}}};
  op_compile_info.norm_vars = {{101200010, {20000, 20001, 20002, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  test.SetCompileInfo(&op_compile_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "1968, 3, 3, 3, 541";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 1);
}

TEST_F(NormTilingRt2Test, TilingTest6) {
  std::vector<std::vector<int64_t>> inputs {
    {7, 543, 76}
  };
  std::vector<std::vector<int64_t>> outputs {
    {7, 543, 76}
  };
  ge::DataType dtype = ge::DT_FLOAT;

  NormCompileInfo op_compile_info;
  op_compile_info.pattern = SchPattern::NORM;
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {1};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{9000, {21784, 16360, 16336, 16336}}};
  op_compile_info.block_size_map = {{9000, 8}};
  op_compile_info.is_fuse_axis = false;
  op_compile_info.is_const = true;
  op_compile_info.is_const_post = false;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  test.SetCompileInfo(&op_compile_info);

  EXPECT_EQ(test.Test(), true);
}

TEST_F(NormTilingRt2Test, TilingTest7) {
  std::vector<std::vector<int64_t>> inputs {
    {7, 543, 76}
  };
  std::vector<std::vector<int64_t>> outputs {
    {7, 543, 76}
  };
  ge::DataType dtype = ge::DT_FLOAT;

  NormCompileInfo op_compile_info;
  op_compile_info.pattern = SchPattern::NORM;
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {1};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{9000, {21784, 16360, 16336, 16336}}};
  op_compile_info.block_size_map = {{9000, 8}};
  op_compile_info.is_fuse_axis = false;
  op_compile_info.is_const = true;
  op_compile_info.is_const_post = true;
  op_compile_info.const_block_dims = {{9000, 28}};
  op_compile_info.workspace_info = {{9000, {4}}};

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  test.SetCompileInfo(&op_compile_info);

  EXPECT_EQ(test.Test(), true);
  EXPECT_EQ(test.GetBlockDims(), 28);
}

TEST_F(NormTilingRt2Test, TilingTest8) {
  std::vector<std::vector<int64_t>> inputs {
    {2, 3}
  };
  std::vector<std::vector<int64_t>> outputs {
    {2, 3}
  };
  ge::DataType dtype = ge::DT_FLOAT;

  NormCompileInfo op_compile_info;
  op_compile_info.pattern = SchPattern::NORM;
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {1};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{4000, {15792, 16120, 15792, 15792}}};
  op_compile_info.block_size_map = {{4000, 8}};
  op_compile_info.workspace_info = {{400400000, {32}}};
  op_compile_info.norm_vars = {{400400000, {20000, 20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  test.SetCompileInfo(&op_compile_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "2, 3, 2, 2";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 1);
}

TEST_F(NormTilingRt2Test, TilingTest9) {
  std::vector<std::vector<int64_t>> inputs {
    {32, 32, 32}
  };
  std::vector<std::vector<int64_t>> outputs {
    {32, 32, 32}
  };
  ge::DataType dtype = ge::DT_FLOAT;

  NormCompileInfo op_compile_info;
  op_compile_info.pattern = SchPattern::NORM;
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {0, 1, 2};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{14000, {16216, 13000, 16216, 16216}}};
  op_compile_info.block_size_map = {{14000, 8}};
  op_compile_info.workspace_info = {{1400090, {4}}};
  op_compile_info.norm_vars = {{1400090, {20000, 20001, 20002, 40000}}};
  op_compile_info.is_fuse_axis = false;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  test.SetCompileInfo(&op_compile_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "32, 32, 32, 11";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 1);
}

TEST_F(NormTilingRt2Test, TilingTest10) {
  std::vector<std::vector<int64_t>> inputs {
    {1000, 1}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1000, 1}
  };
  ge::DataType dtype = ge::DT_FLOAT;

  NormCompileInfo op_compile_info;
  op_compile_info.pattern = SchPattern::NORM;
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {1};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{5000, {21784, 16360, 16336, 16336}}};
  op_compile_info.block_size_map = {{5000, 8}};
  op_compile_info.workspace_info = {{500011, {32}}};
  op_compile_info.norm_vars = {{500011, {20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  test.SetCompileInfo(&op_compile_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "1000, 32, 32";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
}

TEST_F(NormTilingRt2Test, TilingTest11) {
  std::vector<std::vector<int64_t>> inputs {
    {1000, 1}, {1, 2000}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1000, 1}, {1000, 2000}
  };
  ge::DataType dtype = ge::DT_FLOAT;

  NormCompileInfo op_compile_info;
  op_compile_info.pattern = SchPattern::NORM;
  op_compile_info.input_type = {1, 2};
  op_compile_info.ori_reduce_axis = {1};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = true;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{4012, {10552, 10744, 10552, 10552}}};
  op_compile_info.block_size_map = {{4012, 8}};
  op_compile_info.workspace_info = {{401201, {4, 4, 4, 4, 4}}};
  op_compile_info.norm_vars = {{401201, {20000, 20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  test.SetCompileInfo(&op_compile_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "1000, 2000, 32, 2000";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
}

TEST_F(NormTilingRt2Test, TilingTest12) {
  std::vector<std::vector<int64_t>> inputs {
    {3, 1968, 3, 3}
  };
  std::vector<std::vector<int64_t>> outputs {
    {3, 1968, 3, 3}
  };
  ge::DataType dtype = ge::DT_FLOAT;

  NormCompileInfo op_compile_info;
  op_compile_info.pattern = SchPattern::NORM;
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {1, 3};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 8;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = true;
  op_compile_info.available_ub_size = {{20000, {16216, 13000, 16216, 16216}}};
  op_compile_info.block_size_map = {{20000, 8}};
  op_compile_info.workspace_info = {{102000001, {4, -4, -4}}};
  op_compile_info.norm_vars = {{102000001, {20000, 20001, 20002, 20003, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  test.SetCompileInfo(&op_compile_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "3, 1968, 3, 3, 1, 541";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 3);
}

TEST_F(NormTilingRt2Test, TilingTest13) {
  std::vector<std::vector<int64_t>> inputs {
    {1968, 32, 512}, {512}, {512}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1968, 32, 512}, {1968, 32, 1}, {1968, 32, 1}
  };
  ge::DataType dtype = ge::DT_FLOAT;

  NormCompileInfo op_compile_info;
  op_compile_info.pattern = SchPattern::NORM;
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
  op_compile_info.block_size_map = {{4005, 8}};
  op_compile_info.workspace_info = {{1300400500, {32}}};
  op_compile_info.norm_vars = {{1300400500, {20000, 20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  test.SetCompileInfo(&op_compile_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "62976, 512, 1968, 41";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
}

TEST_F(NormTilingRt2Test, TilingTest14) {
  std::vector<std::vector<int64_t>> inputs {
    {95, 10, 1, 87, 16}
  };
  std::vector<std::vector<int64_t>> outputs {
    {95, 10, 1, 87, 16}
  };
  ge::DataType dtype = ge::DT_FLOAT16;

  NormCompileInfo op_compile_info;
  op_compile_info.pattern = SchPattern::NORM;
  op_compile_info.input_type = {0};
  op_compile_info.ori_reduce_axis = {1, 3, 4};
  op_compile_info.ori_disable_fuse_axes = {1, 4};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 16;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = true;
  op_compile_info.available_ub_size = {{42000, {15920, 12840, 15920, 15920}}};
  op_compile_info.block_size_map = {{42000, 16}};
  op_compile_info.workspace_info = {{304200000, {32}}};
  op_compile_info.norm_vars = {{304200000, {20000, 20003, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  test.SetCompileInfo(&op_compile_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "95, 87, 3, 1";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
}

TEST_F(NormTilingRt2Test, TilingTest15) {
  std::vector<std::vector<int64_t>> inputs {
    {1, 23, 512}, {1, 23, 512}, {1, 23, 1}, {1, 23, 1}, {512}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1, 23, 512}, {1, 23, 512}
  };
  ge::DataType dtype = ge::DT_FLOAT16;

  NormCompileInfo op_compile_info;
  op_compile_info.pattern = SchPattern::NORM;
  op_compile_info.input_type = {0, 0, 1, 1, 2};
  op_compile_info.ori_reduce_axis = {2};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 16;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{4012, {9056, 7048, 9056, 9056}}};
  op_compile_info.block_size_map = {{4012, 16}};
  op_compile_info.workspace_info = {{300401200, {32, 32}}};
  op_compile_info.norm_vars = {{300401200, {20000, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  test.SetCompileInfo(&op_compile_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "23, 1, 1";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 23);
}

TEST_F(NormTilingRt2Test, TilingTest16) {
  std::vector<std::vector<int64_t>> inputs {
    {21, 93, 143}, {21, 93, 143}, {21, 1, 1}, {21, 1, 1}, {143}
  };
  std::vector<std::vector<int64_t>> outputs {
    {21, 93, 143}, {21, 93, 143}
  };
  ge::DataType dtype = ge::DT_FLOAT16;

  NormCompileInfo op_compile_info;
  op_compile_info.pattern = SchPattern::NORM;
  op_compile_info.input_type = {0, 0, 1, 1, 2};
  op_compile_info.ori_reduce_axis = {1, 2};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 16;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{10044, {9056, 7048, 9056, 9056}}, {12044, {9056, 7048, 9056, 9056}}};
  op_compile_info.block_size_map = {{10044, 16}};
  op_compile_info.workspace_info = {{1004401, {4, 4}}};
  op_compile_info.norm_vars = {{1004401, {20000, 20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  test.SetCompileInfo(&op_compile_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "21, 93, 1, 47";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 21);
}

TEST_F(NormTilingRt2Test, TilingTest17) {
  std::vector<std::vector<int64_t>> inputs {
    {11, 20, 512}, {11, 20, 1}, {11, 20, 1}, {512}, {512}
  };
  std::vector<std::vector<int64_t>> outputs {
    {11, 20, 512}, {11, 20, 512}
  };
  ge::DataType dtype = ge::DT_FLOAT16;

  std::vector<int64_t> reduce_axis{-1};

  NormCompileInfo op_compile_info;
  op_compile_info.pattern = SchPattern::NORM;
  op_compile_info.input_type = {0, 0, 1, 1, 2};
  op_compile_info.reduce_axis_type = 3;
  op_compile_info.broadcast_axis_type = {1, 2};
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 16;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{4012, {9040, 7256, 9056, 9056}}};
  op_compile_info.block_size_map = {{4012, 16}};
  op_compile_info.workspace_info = {{300401200, {32, 32}}};
  op_compile_info.norm_vars = {{300401200, {20000, 20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  test.SetCompileInfo(&op_compile_info);
  OpInfo op_info(&op_compile_info);
  op_info.SetAxes(&reduce_axis);

  EXPECT_EQ(test.Test(&op_info), true);
  std::string expect_tiling_data = "220, 512, 7, 7";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
}

TEST_F(NormTilingRt2Test, TilingTest18) {
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
  op_compile_info.pattern = SchPattern::NORM;
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
  op_compile_info.block_size_map = {{4005, 16}};
  op_compile_info.workspace_info = {{1300400500, {32}}};
  op_compile_info.norm_vars = {{1300400500, {20000, 20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  test.SetCompileInfo(&op_compile_info);
  std::vector<std::pair<std::string, int64_t>> reduce_attr = {{"int", -1}};
  test.SetAttrs<int64_t>(reduce_attr);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "220, 512, 16, 16";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 14);
}

TEST_F(NormTilingRt2Test, TilingTest19) {
  std::vector<std::vector<int64_t>> inputs {
    {11, 20, 512}
  };
  std::vector<std::vector<int64_t>> outputs {
    {11, 20, 512}
  };
  ge::DataType dtype = ge::DT_FLOAT16;

  NormCompileInfo op_compile_info;
  op_compile_info.pattern = SchPattern::NORM;
  op_compile_info.input_type = {0};
  op_compile_info.reduce_attr_index = 0;
  op_compile_info.reduce_attr_name = "axis";
  op_compile_info.is_reduce_attr_is_int = false;
  op_compile_info.reduce_axis_type = 9;
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 16;
  op_compile_info.transpose_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{4000, {15824, 16360, 15824, 15824}}};
  op_compile_info.block_size_map = {{4000, 16}};
  op_compile_info.workspace_info = {{300400000, {32}}};
  op_compile_info.norm_vars = {{300400000, {20000, 20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  test.SetCompileInfo(&op_compile_info);
  std::vector<std::pair<std::string, std::vector<int64_t>>> reduce_attr = {{"list_int", {-1}}};
  test.SetAttrs<std::vector<int64_t>>(reduce_attr);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "220, 512, 7, 7";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
}

 TEST_F(NormTilingRt2Test, TilingTest20) {
   std::vector<std::vector<int64_t>> inputs {
     {1968, 32, 512}, {512}, {512}
   };
   std::vector<std::vector<int64_t>> outputs {
     {1968, 32, 512}, {1968, 32, 1}, {1968, 32, 1}
   };
   ge::DataType dtype = ge::DT_FLOAT;

   NormCompileInfo op_compile_info;
   op_compile_info.pattern = SchPattern::NORM;
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
   op_compile_info.block_size_map = {{4005, 8}};
   op_compile_info.workspace_info = {{1300400500, {32}}};
   op_compile_info.norm_vars = {{1300400500, {20000, 20001, 30000, 40000}}};
   op_compile_info.is_fuse_axis = true;
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
   op_compile_info.var_attr_wrap.ParseVarAttr(nlohmann::json::parse(var_attr_list_compileInfo));
   AutoTilingTest test(inputs, outputs, dtype, dtype);
   test.SetCompileInfo(&op_compile_info);
   std::vector<std::pair<std::string, int64_t>> common_attr = {{"int64", {2}}};
   test.SetAttrs<int64_t>(common_attr);

   EXPECT_EQ(test.Test(), true);
   std::string expect_tiling_data = "62976, 512, 1968, 41, 2";
   EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
   EXPECT_EQ(test.GetBlockDims(), 32);
 }