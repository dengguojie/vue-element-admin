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

static string to_string(const std::stringstream& tiling_data) {
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
                            bool is_input = true, ge::Format format = ge::FORMAT_ND) {
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

template <typename T1, typename T2>
static bool compare_map(const std::unordered_map<T1, T2>& map1, const std::unordered_map<T1, T2>& map2) {
  if (map1.size() != map2.size()) {
    return false;
  }
  for (const auto& it : map1) {
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
    std::cout << "ERROR: is_const";
    return false;
  }
  return true;
}

TEST_F(TupleReduceTilingTest, ParseTest1) {
  std::string compileInfo = R"({
    "_reduce_axis": [0, 2, 3],
    "_is_const": true,
    "_fused_reduce_axis": [0, 2],
    "_fusible_code": [11, 10, 11, 11, 10],
    "_pattern": "TupleReduce",
    "_common_info": [32, 32, true, 2048],
    "_graph_info": [1, 2, 4, 4],
    "_runtime": false,
    "_buffer_size": [16224, 12976, 12976],
    "_dim_var_code": {"104": 0}
    })";

  nlohmann::json op_info = nlohmann::json::parse(compileInfo.c_str());

  optiling::TupleReduce::TupleReduceCompileInfo actual_struct("tuple_reduce", op_info);
  optiling::TupleReduce::TupleReduceCompileInfo expect_struct;
  expect_struct.reduce_axis = {0, 2, 3};
  expect_struct.is_const = true;
  expect_struct.fused_reduce_axis = {0, 2};
  expect_struct.fusible_code = {11, 10, 11, 11, 10};
  expect_struct.common_info = {32, 32, 1, 2048};
  expect_struct.graph_info = {1, 2, 4, 4};
  expect_struct.runtime = false;
  expect_struct.buffer_size = {16224, 12976, 12976};
  const auto& local_dim_var_code = op_info.at("_dim_var_code").get<std::unordered_map<std::string, int32_t>>();
  for (const auto& item: local_dim_var_code) {
    expect_struct.dim_var_code[std::stoi(item.first)] = item.second;
  }

  ASSERT_TRUE(compare_tuple_reduce_struct(actual_struct, expect_struct));
}

TEST_F(TupleReduceTilingTest, TilingTest1) {
  std::vector<std::vector<int64_t>> inputs{{32, 4, 112 * 112, 16}};
  std::vector<std::vector<int64_t>> outputs{{1, 4, 1, 16}, {1, 4, 1, 16}};
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

  std::string StrCompileInfo = R"({"_dim_var_code": {"104": 0}})";
  nlohmann::json op_info = nlohmann::json::parse(StrCompileInfo.c_str());

  optiling::TupleReduce::TupleReduceCompileInfo compileInfo;
  compileInfo.reduce_axis = {0, 2, 3};
  compileInfo.is_const = true;
  compileInfo.fused_reduce_axis = {0, 2};
  compileInfo.fusible_code = {11, 10, 11, 11, 10};
  compileInfo.common_info = {32, 32, 1, 2048};
  compileInfo.core_num = 32;
  compileInfo.block_size = 32;
  compileInfo.atomic_support = 32;
  compileInfo.atomic_threshold = 32;

  compileInfo.graph_info = {1, 2, 4, 4};
  compileInfo.inputs_num = 1;
  compileInfo.min_dtype_size = 2;
  compileInfo.max_dtype_size = 4;
  compileInfo.reduce_dtype_size = 4;

  compileInfo.runtime = false;
  compileInfo.buffer_size = {16224, 12976, 12976};
  const auto& local_dim_var_code = op_info.at("_dim_var_code").get<std::unordered_map<std::string, int32_t>>();
  for (const auto& item: local_dim_var_code) {
    compileInfo.dim_var_code[std::stoi(item.first)] = item.second;
  }

  optiling::TupleReduce::TupleReduce tupleReduce("TupleReduce", op_paras, compileInfo, runInfo);
  ASSERT_TRUE(tupleReduce.DoTiling());
}

TEST_F(TupleReduceTilingTest, TilingTest2) {
  std::vector<std::vector<int64_t>> inputs{{32, 1280, 64, 16}};
  std::vector<std::vector<int64_t>> outputs{{1, 1280, 1, 16}, {1, 1280, 1, 16}};
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

  std::string StrCompileInfo = R"({"_dim_var_code": {"104": 0}})";
  nlohmann::json op_info = nlohmann::json::parse(StrCompileInfo.c_str());

  optiling::TupleReduce::TupleReduceCompileInfo compileInfo;
  compileInfo.reduce_axis = {0, 2, 3};
  compileInfo.is_const = true;
  compileInfo.fused_reduce_axis = {0, 2};
  compileInfo.fusible_code = {11, 10, 11, 11, 10};
  compileInfo.common_info = {32, 32, 1, 2048};
  compileInfo.core_num = 32;
  compileInfo.block_size = 32;
  compileInfo.atomic_support = 32;
  compileInfo.atomic_threshold = 32;

  compileInfo.graph_info = {1, 2, 4, 4};
  compileInfo.inputs_num = 1;
  compileInfo.min_dtype_size = 2;
  compileInfo.max_dtype_size = 4;
  compileInfo.reduce_dtype_size = 4;

  compileInfo.runtime = false;
  compileInfo.buffer_size = {13000, 10832, 10832};
  const auto& local_dim_var_code = op_info.at("_dim_var_code").get<std::unordered_map<std::string, int32_t>>();
  for (const auto& item: local_dim_var_code) {
    compileInfo.dim_var_code[std::stoi(item.first)] = item.second;
  }

  optiling::TupleReduce::TupleReduce tupleReduce("TupleReduce", op_paras, compileInfo, runInfo);
  ASSERT_TRUE(tupleReduce.DoTiling());
}
