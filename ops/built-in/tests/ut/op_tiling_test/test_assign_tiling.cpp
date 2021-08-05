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

using namespace std;
using namespace ge;

class AssignTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AssignTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AssignTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream &tiling_data) {
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

TEST_F(AssignTiling, Assign_tiling1) {
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find("Assign");
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  auto opParas = op::Assign("Assign");

  vector<vector<int64_t>> input{
      {4, 4, 4, 4},
      {4, 4, 4, 4},
  };
  TensorDesc tensor_input1(ge::Shape(input[0]), FORMAT_ND, DT_FLOAT16);
  auto data1 = op::Data("ref");
  data1.update_input_desc_x(tensor_input1);
  data1.update_output_desc_y(tensor_input1);
  TensorDesc tensor_input2(ge::Shape(input[1]), FORMAT_ND, DT_FLOAT16);
  auto data2 = op::Data("value");
  data2.update_input_desc_x(tensor_input2);
  data2.update_output_desc_y(tensor_input2);
  
  opParas.set_input_ref(data1);
  opParas.set_input_value(data2);
  std::vector<Operator> inputs{data1, data2};
  std::vector<Operator> outputs{opParas};

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 256000}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo.c_str());
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "4 4 4 ");
}

