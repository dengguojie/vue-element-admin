#include <iostream>
#include <fstream>
#include <vector>
#include "securec.h"

#include <gtest/gtest.h>
#include "op_tiling/vector_tiling.h"
#include "op_tiling/tuple_reduce.h"
#include "graph/utils/op_desc_utils.h"
#include "op_tiling/tiling_handler.h"

using namespace std;
using namespace ge;
using namespace optiling;

class TupleReduceTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "TupleReduceTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "TupleReduceTilingTest TearDown" << std::endl;
    }
};

static string to_string(const std::stringstream &tiling_data) {
  auto data = tiling_data.str();
  string result;
  int32_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
    memcpy_s(&tmp, sizeof(tmp), data.c_str() + i, sizeof(tmp));
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

static bool compare_tuple_reduce_struct(const optiling::TupleReduce::TupleReduceCompileInfo ptr1,
                                        const optiling::TupleReduce::TupleReduceCompileInfo ptr2) {
  if (ptr1.is_const != ptr2.is_const) {
    std::cout << "ERROR1";
    return false;
  }
  return true;
}

TEST_F(TupleReduceTilingTest, ParseTest1) {
  std::string compileInfo = R"({ "_disable_fuse_axes": [],
                                 "_reduce_axis": [1, 2],
                                 "_shapes_length": [4, 4],
                                 "_max_shape_len": 4,
                                 "_fusible_code": [0, 1, 1, 0],
                                 "_fused_reduce_axis": [1],
                                 "_fused_broadcast_axis": [],
                                 "_fused_disable_fuse_axes": [],
                                 "_pattern": "TupleReduce",
                                 "_is_const": true,
                                 "_common_info": [32, 262144, 32, true, 0],
                                 "_runtime": true,
                                 "_graph_info": [2, 8, 4, 4, true],
                                 "_vars": {"64110": []},
                                 "_normal_vars": {"64110": []},
                                 "_attr_vars": {"64110": []},
                                 "_custom_vars": {"64110": []}})";
  nlohmann::json op_info = nlohmann::json::parse(compileInfo.c_str());

  optiling::TupleReduce::TupleReduceCompileInfo actual_struct("tuple_reduce", op_info);
  optiling::TupleReduce::TupleReduceCompileInfo expect_struct;
  expect_struct.common_info = {32, 262144, 32, 1, 0};
  expect_struct.core_num = 32;
  expect_struct.ub_size = 262144;
  expect_struct.block_size = 32;
  expect_struct.atomic_support = true;
  expect_struct.dim_var = 0;
  expect_struct.reduce_axis = {1, 2};
  expect_struct.fused_reduce_axis = {1};
  expect_struct.fusible_code = {0, 1, 1, 0};
  expect_struct.is_const = true;
  expect_struct.runtime = true;
  expect_struct.graph_info = {2, 8, 4, 4, 1};
  expect_struct.inputs_num = 2;
  expect_struct.buffer_count = 8;
  expect_struct.max_dtype_size = 4;
  expect_struct.min_dtype_size = 4;
  expect_struct.keep_dims = true;
  expect_struct.shapes_length = {4, 4};
  expect_struct.max_shape_len = 4;
  expect_struct.each_buffer_size = 32768;
  expect_struct.parsed_success = true;
  
  ASSERT_TRUE(compare_tuple_reduce_struct(actual_struct, expect_struct));
}

TEST_F(TupleReduceTilingTest, TilingTest1) {
  std::vector<std::vector<int64_t>> inputs {
    {64, 768, 16, 16}, {64, 768, 16, 16}
  };
  std::vector<std::vector<int64_t>> outputs {
    {64, 768, 16, 16}, {64, 768, 16, 16}
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

  optiling::TupleReduce::TupleReduceCompileInfo compileInfo;
  compileInfo.core_num = 32;
  compileInfo.ub_size = 262144;
  compileInfo.block_size = 32;
  compileInfo.atomic_support = true;
  compileInfo.dim_var = 0;

  compileInfo.reduce_axis = {1, 2};
  compileInfo.fused_reduce_axis = {1};
  compileInfo.fusible_code = {0, 1, 1, 0};

  compileInfo.is_const = true;
  compileInfo.runtime = true;

  compileInfo.inputs_num = 2;
  compileInfo.buffer_count = 8;
  compileInfo.max_dtype_size = 4;
  compileInfo.min_dtype_size = 4;
  compileInfo.keep_dims = true;
  compileInfo.shapes_length = {4, 4};
  compileInfo.max_shape_len = 4;
  compileInfo.each_buffer_size = 32768;

  optiling::TupleReduce::TupleReduce tupleReduce("TupleReduce", op_paras, compileInfo, runInfo);
  ASSERT_TRUE(tupleReduce.DoTiling());
}

TEST_F(TupleReduceTilingTest, TilingTest2) {
  std::vector<std::vector<int64_t>> inputs {
    {24, 512, 102400}, {24, 512, 102400}
  };
  std::vector<std::vector<int64_t>> outputs {
    {24, 512, 102400}, {24, 512, 102400}
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

  optiling::TupleReduce::TupleReduceCompileInfo compileInfo;
  compileInfo.core_num = 32;
  compileInfo.ub_size = 262144;
  compileInfo.block_size = 32;
  compileInfo.atomic_support = true;
  compileInfo.dim_var = 0;

  compileInfo.reduce_axis = {0, 1};
  compileInfo.fused_reduce_axis = {0};
  compileInfo.fusible_code = {1, 1, 0};

  compileInfo.is_const = true;
  compileInfo.runtime = true;

  compileInfo.inputs_num = 2;
  compileInfo.buffer_count = 8;
  compileInfo.max_dtype_size = 4;
  compileInfo.min_dtype_size = 4;
  compileInfo.keep_dims = true;
  compileInfo.shapes_length = {3, 3};
  compileInfo.max_shape_len = 3;
  compileInfo.each_buffer_size = 32768;

  optiling::TupleReduce::TupleReduce tupleReduce("TupleReduce", op_paras, compileInfo, runInfo);
  ASSERT_TRUE(tupleReduce.DoTiling());
}
