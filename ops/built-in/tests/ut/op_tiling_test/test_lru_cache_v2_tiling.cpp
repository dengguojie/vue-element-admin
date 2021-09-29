#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "selection_ops.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"
#include "register/op_tiling_registry.h"
#include "data_flow_ops.h"

using namespace std;
using namespace ge;

class LRUCacheV2Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "LRUCacheV2Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "LRUCacheV2Tiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream& tiling_data) {
  auto data = tiling_data.str();
  string result;
  int64_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }

  return result;
}

using namespace ge;
#include "test_common.h"
/*
REG_OP(LRUCacheV2)
    .INPUT(index_list, TensorType::BasicType())
    .INPUT(data, TensorType::BasicType())
    .INPUT(cache, TensorType::BasicType())
    .INPUT(tag, TensorType::BasicType())
    .INPUT(is_last_call, TensorType::BasicType())
    .OUTPUT(data, TensorType::BasicType())
    .OUTPUT(cache, TensorType::BasicType())
    .OUTPUT(tag, TensorType::BasicType())
    .OUTPUT(index_offset_list, TensorType::BasicType())
    .OUTPUT(not_in_cache_index_list, TensorType::BasicType())
    .OUTPUT(not_in_cache_number, TensorType::BasicType())
    .REQUIRED_ATTR(pre_route_count, Int)
    .OP_END_FACTORY_REG(LRUCacheV2)
*/
TEST_F(LRUCacheV2Tiling, lru_cache_v2_tiling_0) {
  std::string op_name = "LRUCacheV2";
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  std::string compileInfo = "{\"vars\": {\"core_num\": 8}}";

  std::vector<int64_t> shape_a{-1};
  std::vector<int64_t> shape_b{1024, 256};
  std::vector<int64_t> shape_c{131072};
  std::vector<int64_t> shape_d{1};
  std::vector<int64_t> shape_e{512};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;

  TensorDesc index_list_desc;
  index_list_desc.SetShape(ge::Shape(shape_a));
  index_list_desc.SetDataType(dtypeB);
  TensorDesc data_desc;
  data_desc.SetShape(ge::Shape(shape_b));
  data_desc.SetDataType(dtypeA);
  TensorDesc cache_desc;
  cache_desc.SetShape(ge::Shape(shape_c));
  cache_desc.SetDataType(dtypeA);
  TensorDesc tag_desc;
  tag_desc.SetShape(ge::Shape(shape_e));
  tag_desc.SetDataType(dtypeB);
  TensorDesc is_last_call_desc;
  is_last_call_desc.SetShape(ge::Shape(shape_d));
  is_last_call_desc.SetDataType(dtypeB);

  auto opParas = op::LRUCacheV2("LRUCacheV2");
  TENSOR_INPUT(opParas, index_list_desc, index_list);
  TENSOR_INPUT(opParas, data_desc, data);
  TENSOR_INPUT(opParas, cache_desc, cache);
  TENSOR_INPUT(opParas, tag_desc, tag);
  TENSOR_INPUT(opParas, is_last_call_desc, is_last_call);
  TENSOR_OUTPUT(opParas, data_desc, data);
  TENSOR_OUTPUT(opParas, cache_desc, cache);
  TENSOR_OUTPUT(opParas, tag_desc, tag);
  TENSOR_OUTPUT(opParas, index_list_desc, not_in_cache_index_list);
  TENSOR_OUTPUT(opParas, index_list_desc, index_offset_list);
  TENSOR_OUTPUT(opParas, is_last_call_desc, not_in_cache_number);
  string expectTilingData = "0 -1 ";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), expectTilingData);
  int64_t profiling_test_num = 100;
  for (int64_t i = 0; i < profiling_test_num; i++) {
    iter->second(opParas, op_compile_info, runInfo);
  }
}