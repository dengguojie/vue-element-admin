//
// Created by xukaiwei on 7/1/21.
//
#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#define private public
#define private public
#include "register/op_tiling_registry.h"

#include "graph/graph.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "op_tiling/reduce_tiling_v3.h"
#include "op_tiling/tiling_handler.h"

#include "reduce_ops.h"
#include "array_ops.h"
#include "test_common.h"


using namespace std;
using namespace ge;
using namespace optiling;

class ReduceTilingV3 : public testing::Test {
protected:
   static void SetUpTestCase() {
     std::cout << "ReduceTilingV3 SetUp" << std::endl;
   }

   static void TearDownTestCase() {
     std::cout << "ReduceTilingV3 TearDown" << std::endl;
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

template<typename T1, typename T2>
static bool compare_var_attr_map(const std::unordered_map<T1, T2>& map1, const std::unordered_map<T1, T2>& map2) {
  if (map1.size() != map2.size()) {
     std::cout << "map1.size is :" << map1.size() << endl;
     std::cout << "map2.size is :" << map2.size() << endl;
     std::cout << "map1.size";
    return false;
  }
  for (const auto& it: map1) {
    if (map2.count(it.first) == 0) {
      std::cout << "map2.count";
      return false;
    }

    std::vector<VarAttr> var_attr_1_list = map1.at(it.first);
    std::vector<VarAttr> var_attr_2_list = map2.at(it.first);

    if (var_attr_1_list.size() != var_attr_2_list.size()) {
     std::cout << "var_attr_1_list.size()";
      return false;
    }

    for (int i = 0; i < var_attr_1_list.size(); i ++){
      VarAttr var_attr_1 = var_attr_1_list[i];
      VarAttr var_attr_2 = var_attr_2_list[i];
      if (var_attr_1.name != var_attr_2.name) {
        std::cout << "var_attr_1.name";
        return false;
      }
      if (var_attr_1.length != var_attr_2.length) {
        std::cout << "ar_attr_1.length";
        return false;
      }
      if (var_attr_1.type != var_attr_2.type) {
       std::cout << "var_attr_1.type";
        return false;
      }
      if (var_attr_1.src_type != var_attr_2.src_type) {
      std::cout << "var_attr_1.src_type";
        return false;
      }
    }
  }
  return true;
}
static bool compare_reduce_struct(const optiling::v3::ReduceCompileInfo ptr1,
                                  const optiling::v3::ReduceCompileInfo ptr2) {
  if (ptr1.is_const != ptr2.is_const) {
   std::cout << "ERROR1";
    return false;
  }
  if (ptr1.is_const_post != ptr2.is_const_post) {
  std::cout << "ERROR2";
    return false;
  }
  if (ptr1.atomic != ptr2.atomic) {
  std::cout << "ERROR3" <<std::endl;
  std::cout << ptr1.atomic <<std::endl;
  std::cout << ptr2.atomic <<std::endl;
    return false;
  }
  if (ptr1.is_keep_dims != ptr2.is_keep_dims) {
  std::cout << "ERROR4";
    return false;
  }
  if (ptr1.idx_before_reduce != ptr2.idx_before_reduce) {
  std::cout << "ERROR5";
    return false;
  }
  if (ptr1.zero_ub_factor != ptr2.zero_ub_factor) {
   std::cout << "ERROR6";
    return false;
  }
  if (ptr1.core_num != ptr2.core_num) {
  std::cout << "ERROR7";
    return false;
  }
  if (ptr1.min_block_size != ptr2.min_block_size) {
  std::cout << "ERROR8";
    return false;
  }
  if (ptr1.coef != ptr2.coef) {
  std::cout << "ERROR9";
    return false;
  }
  if (ptr1.pattern_info != ptr2.pattern_info) {
  std::cout << "ERROR10";
    return false;
  }
  if (ptr1.ub_info_rf != ptr2.ub_info_rf) {
  std::cout << "ERROR11";
    return false;
  }
  if (ptr1.ub_info != ptr2.ub_info) {
  std::cout << "ERROR12";
    return false;
  }
  if (ptr1.ori_axis.first != ptr1.ori_axis.first) {
  std::cout << "ERROR13";
    return false;
  }
  if (ptr1.ori_axis.second != ptr1.ori_axis.second) {
  std::cout << "ERROR14";
    return false;
  }
  if (ptr1.axes_idx.first != ptr1.axes_idx.first) {
  std::cout << "ERROR15";
    return false;
  }
  if (ptr1.axes_idx.second != ptr1.axes_idx.second) {
  std::cout << "ERROR16";
    return false;
  }
  if (ptr1.compile_pattern.first != ptr1.compile_pattern.first) {
  std::cout << "ERROR17";
    return false;
  }
  if (ptr1.compile_pattern.second != ptr1.compile_pattern.second) {
  std::cout << "ERROR18";
    return false;
  }
  if (!compare_map(ptr1.block_dim_map, ptr2.block_dim_map)) {
  std::cout << "ERROR19";
    return false;
  }
  if (!compare_map(ptr1.atomic_flags_map, ptr2.atomic_flags_map)) {
  std::cout << "ERROR20";
    return false;
  }

  return true;
}


TEST_F(ReduceTilingV3, ReduceParseTest1) {
  std::string compileInfo = R"({"_ori_axis": [0],
                                "_pattern": "CommReduce",
                                "_common_info": [32, 1, 8, 1, 1],
                                "_pattern_info": [5],
                                "axes_idx":0,
                                "_compile_pattern":0,
                                "_ub_info": [16256],
                                "_ub_info_rf": [16256],
                                "_block_dims": {"40000400": 0},
                                "_atomic_flags": {"40000400": false}})";
  nlohmann::json json_info = nlohmann::json::parse(compileInfo.c_str());
  v3::ReduceCompileInfo actualCompileInfo = v3::ReduceCompileInfo("CommonReduce", json_info);
  v3::ReduceCompileInfo expectCompileInfo;
  expectCompileInfo.is_const = false;
  expectCompileInfo.is_const_post = false;
  expectCompileInfo.idx_before_reduce = 0;
  expectCompileInfo.zero_ub_factor = -1;
  expectCompileInfo.core_num = 32;
  expectCompileInfo.is_keep_dims = true;
  expectCompileInfo.min_block_size = 8;
  expectCompileInfo.atomic = true;
  expectCompileInfo.coef = 1;
  expectCompileInfo.pattern_info = {5};
  expectCompileInfo.ub_info_rf = {16256};
  expectCompileInfo.ub_info = {16256};
  expectCompileInfo.ori_axis.first = true;
  expectCompileInfo.ori_axis.second = {0};
  expectCompileInfo.axes_idx.first = false;
  expectCompileInfo.compile_pattern.first = false;
  expectCompileInfo.block_dim_map = {{"40000400", 0}};
  expectCompileInfo.atomic_flags_map = {{"40000400",false}};

  ASSERT_TRUE(compare_reduce_struct(actualCompileInfo, actualCompileInfo));
}

TEST_F(ReduceTilingV3, ReduceParseTest2) {
  std::string compileInfo = R"({"_ori_axis": [0],
                                "_pattern": "CommReduce",
                                "_common_info": [32, 1, 8, 1],
                                "_pattern_info": [5],
                                "axes_idx":0,
                                "_compile_pattern":0,
                                "_ub_info": [16256],
                                "_ub_info_rf": [16256],
                                "_block_dims": {"40000400": 0},
                                "_atomic_flags": {"40000400": false}})";
  nlohmann::json json_info = nlohmann::json::parse(compileInfo.c_str());
  auto compileInfo_ptr = CreateReduceTilingHandler("reduce","CommReduce",json_info);

  bool result = compileInfo_ptr ? true : false;
  ASSERT_FALSE(compileInfo_ptr);
}

//TEST_F(ReduceTilingV3, ReduceParseTest3) {
//  std::string compileInfo = R"({"_ori_axis": [0],
//                                "_pattern": "CommReduce",
//                                "_var_attr_mode":1,
//                                "_var_attrs": {"40000400":[{"length":1,"name":"alpha","type":"float16",
//                                               "src_type":"float16"}],
//                                               "3100000":[{"length":2,"name":"beta","type":"float32",
//                                               "src_type":"float16"}]},
//                                "_common_info": [32, 1, 8, 1, 1],
//                                "_pattern_info": [5],
//                                "axes_idx":0,
//                                "_compile_pattern":0,
//                                "_ub_info": [16256],
//                                "_ub_info_rf": [16256],
//                                "_block_dims": {"40000400": 0},
//                                "_atomic_flags": {"40000400": false}})";
//  nlohmann::json json_info = nlohmann::json::parse(compileInfo.c_str());
//  v3::ReduceCompileInfo actualCompileInfo = v3::ReduceCompileInfo("CommonReduce", json_info);
//  unordered_map<std::uint64_t, std::vector<VarAttr>> expect_var_attr_map;
//  VarAttr var1 = VarAttr("alpha","float16","float16",1);
//  std::vector<VarAttr> var1_attr_list;
//  var1_attr_list.push_back(var1);
//  expect_var_attr_map[40000400] = var1_attr_list;
//
//  VarAttr var1 = VarAttr("beta","float32","float16",2);
//  std::vector<VarAttr> var2_attr_list;
//  var2_attr_list.push_back(var2);
//  expect_var_attr_map[3100000] = var2_attr_list;
//  ASSERT_TRUE(true);
////  ASSERT_TRUE(compare_var_attr_map(expect_var_attr_map, actualCompileInfo.varAttrWrap.var_attr_map));
//}


TEST_F(ReduceTilingV3, ReduceParseTest5) {
  std::string compileInfo = R"({"_ori_axis": [0],
                                "_pattern": "CommReduce",
                                "_common_info": [32, 1, 8, 1, 1, 0],
                                "_pattern_info": [5],
                                "axes_idx":0,
                                "_compile_pattern":0,
                                "_ub_info": [16256],
                                "_ub_info_rf": [16256],
                                "_block_dims": {"40000400": 0},
                                "_atomic_flags": {"40000400": false}})";
  nlohmann::json json_info = nlohmann::json::parse(compileInfo.c_str());
  auto compileInfo_ptr = CreateReduceTilingHandler("reduce","CommReduce",json_info);

  bool result = compileInfo_ptr ? true : false;
  ASSERT_FALSE(compileInfo_ptr);
}

/* Test Case
 * **/
TEST_F(ReduceTilingV3, ReduceTiling1) {
  using namespace optiling;

  std::vector<int64_t> input{1};
  std::vector<int64_t> output{1};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto _op = op::ReduceSumD("ReduceSumD_1");
  _op.set_input_x(x1);
  _op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{_op};
  ge::Graph graph("ReduceTiling1");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string compileInfo = R"({ "_ori_axis": [0], "_pattern": "CommReduce","push_status": 0,"_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5], "_ub_info": [16256], "_ub_info_rf": [16256], "_vars": {"-1000500": ["_dim_1_0", "_block_factor", "_ub_factor"]}})";

  optiling::utils::OpRunInfo runInfo;
  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compileInfo);
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateReduceTilingHandler(this->test_info_->name(),
                              "CommReduce",
                              nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(_op, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 1 ");
}

TEST_F(ReduceTilingV3, ReduceTiling2) {
  using namespace optiling;

  std::vector<int64_t> input{2, 39, 0};
  std::vector<int64_t> output{2, 39, 1};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto _op = op::ReduceSumD("ReduceSumD_2");
  _op.set_input_x(x1);
  _op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{_op};
  ge::Graph graph("ReduceTiling2");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string compileInfo = R"({ "_ori_axis": [2], "_pattern": "CommReduce", "push_status": 0,
                                               "_zero_ub_factor": 25600, "_vars": {"10": ["_dim_1", "_ub_factor"]}})";

  optiling::utils::OpRunInfo runInfo;

  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compileInfo);
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateReduceTilingHandler(this->test_info_->name(),
                              "CommReduce",
                              nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(_op, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "78 25600 ");
}

TEST_F(ReduceTilingV3, ReduceTiling3) {
  using namespace optiling;

  std::vector<int64_t> input{2, 39, 0};
  std::vector<int64_t> output{};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto _op = op::ReduceSumD("ReduceSumD_3");
  _op.set_input_x(x1);
  _op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{_op};
  ge::Graph graph("ReduceTiling3");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string compileInfo = R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "_zero_ub_factor": 32128, "_vars": {"110": ["_dim_2", "_ub_factor"]}})";

  optiling::utils::OpRunInfo runInfo;

  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compileInfo);
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateReduceTilingHandler(this->test_info_->name(),
                              "CommReduce",
                              nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(_op, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "2 128 ");
}

TEST_F(ReduceTilingV3, ReduceTiling4) {
  using namespace optiling;

  std::vector<int64_t> input{64, 64};
  std::vector<int64_t> output{1,64};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto _op = op::ReduceSumD("ReduceSumD_4");
  _op.set_input_x(x1);
  _op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{_op};
  ge::Graph graph("ReduceTiling4");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string compileInfo = R"({"_ori_axis": [0],"_pattern": "CommReduce", "push_status": 0, "_zero_ub_factor": 32512, "_common_info": [32,1,8,1,1], "_pattern_info": [1], "_ub_info":[32512], "_ub_info_rf": [32512], "_reduce_shape_known": true, "_const_shape_post": true, "_compile_pattern": 1, "_block_dims":{"1":32},
     "_atomic_flags":{"1": true},
     "_vars": {"1": []}})";

  optiling::utils::OpRunInfo runInfo;

  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compileInfo);
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateReduceTilingHandler(this->test_info_->name(),
                              "CommReduce",
                              nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(_op, runInfo));
}

TEST_F(ReduceTilingV3, ReduceTiling5) {
  using namespace optiling;

  std::vector<int64_t> input{1};
  std::vector<int64_t> output{1};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto _op = op::ReduceSumD("ReduceSumD_5");
  _op.set_input_x(x1);
  _op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{_op};
  ge::Graph graph("ReduceTiling5");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string compileInfo = R"({ "_ori_axis": [0], "_pattern": "CommReduce","push_status": 0,"common_info": [32, 1, 8, 1, 1], "pattern_info": [20000], "ub_info": [16256], "_ub_info_rf": [16256], "_vars": {"-1000500": ["dim_1_0", "block_factor", "ub_factor"]}})";

  optiling::utils::OpRunInfo runInfo;

  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compileInfo);
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateReduceTilingHandler(this->test_info_->name(),
                              "CommReduce",
                              nlohmann::json::parse(compileInfo));
  ASSERT_FALSE(outer_compile_info->DoTiling(_op, runInfo));
}

TEST_F(ReduceTilingV3, ReduceTiling6) {
  using namespace optiling;

  std::vector<int64_t> input{1};
  std::vector<int64_t> output{1};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto _op = op::ReduceSumD("ReduceSumD_6");
  _op.set_input_x(x1);
  _op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{_op};
  ge::Graph graph("ReduceTiling6");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string compileInfo = R"({ "axes_idx": 0, "_pattern": "CommReduce","push_status": 0,"common_info": [32, 1, 8, 1, 1], "pattern_info": [20000], "ub_info": [16256], "_ub_info_rf": [16256], "_vars": {"-1000500": ["dim_1_0", "block_factor", "ub_factor"]}})";

  optiling::utils::OpRunInfo runInfo;

  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compileInfo);
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateReduceTilingHandler(this->test_info_->name(),
                              "CommReduce",
                              nlohmann::json::parse(compileInfo));
  ASSERT_FALSE(outer_compile_info->DoTiling(_op, runInfo));
}

// ReduceTiling7 const_tensor

// FineTuning tune0
TEST_F(ReduceTilingV3, ReduceTiling8) {
  using namespace optiling;

  std::vector<int64_t> input{10000, 9, 80};
  std::vector<int64_t> output{1, 9, 80};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT16);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto _op = op::ReduceSumD("ReduceSumD_8");
  _op.set_input_x(x1);
  _op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{_op};
  ge::Graph graph("ReduceTiling8");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string compileInfo = R"({"_ori_axis": [0],"_pattern": "CommReduce", "push_status": 0,
                               "_zero_ub_factor": 32512, "_common_info": [32,1,16,0,1],
                               "_pattern_info": [5,4,9], "_ub_info":[21632, 21376, 21632],
                               "_ub_info_rf": [21632,16000,21632],
                               "_pattern": "CommReduce",
                               "_vars": {"1": []}})";

  optiling::utils::OpRunInfo runInfo;

  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compileInfo);
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateReduceTilingHandler(this->test_info_->name(),
                              "CommReduce",
                              nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(_op, runInfo));
}

// FineTuning tune1
TEST_F(ReduceTilingV3, ReduceTiling9) {
  using namespace optiling;

  std::vector<int64_t> input{16, 1, 8, 38, 1, 16, 16};
  std::vector<int64_t> output{1, 16};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto _op = op::ReduceSumD("ReduceSumD_9");
  _op.set_input_x(x1);
  _op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{_op};
  ge::Graph graph("ReduceTiling9");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string compileInfo = R"({"_ori_axis": [0,2,3,4,5],"_pattern": "CommReduce",
                               "_common_info": [32,1,8,1,1],
                               "_pattern_info": [5,4,9], "_ub_info":[32512, 32128, 16128],
                               "_ub_info_rf": [32512, 21376, 32512],
                               "_pattern": "CommReduce",
                               "_vars": {"1": []}})";

  optiling::utils::OpRunInfo runInfo;

  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compileInfo);
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateReduceTilingHandler(this->test_info_->name(),
                              "CommReduce",
                              nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(_op, runInfo));
}

// for new interface
TEST_F(ReduceTilingV3, ReduceTiling10) {
  using namespace optiling;

  std::vector<int64_t> input{64, 64};
  std::vector<int64_t> output{1,64};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto _op = op::ReduceSumD("ReduceSumD_4");
  _op.set_input_x(x1);
  _op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{_op};
  ge::Graph graph("ReduceTiling4");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);


  std::string compileInfo = R"({"_ori_axis": [0],"_pattern": "CommReduce", "push_status": 0, "_zero_ub_factor": 32512, "_common_info": [32,1,8,1,1], "_pattern_info": [1], "_ub_info":[32512], "_ub_info_rf": [32512], "_reduce_shape_known": true, "_const_shape_post": true, "_compile_pattern": 1, "_block_dims":{"1":32},
       "_atomic_flags":{"1": true},
       "_vars": {"1": []}})";

  // new interface
  optiling::utils::OpRunInfo runInfo;
  std::vector<std::vector<int64_t>> input_shapes{input,};
  optiling::OpInfo c_op_info(input_shapes, DT_FLOAT);
  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compileInfo);
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateReduceTilingHandler(this->test_info_->name(),
                              "CommReduce",
                              nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(_op, runInfo, c_op_info));
}


TEST_F(ReduceTilingV3, ReduceTiling11) {
  using namespace optiling;

  std::vector<int64_t> input{1};
  std::vector<int64_t> output{1};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto _op = op::ReduceSumD("ReduceSumD_11");
  _op.set_input_x(x1);
  _op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{_op};
  ge::Graph graph("ReduceTiling11");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);


  std::string compileInfo = R"({"_ori_axis": [0],
                                "_pattern": "CommReduce",
                                "_common_info": [32, 1, 8, 1, 1],
                                "_pattern_info": [5],
                                "_ub_info": [16256],
                                "_ub_info_rf": [16256],
                                "_vars": {"-1000500": ["_dim_1_0", "_block_factor", "_ub_factor"]}})";

  // new interface
  optiling::utils::OpRunInfo runInfo;
  std::vector<std::vector<int64_t>> input_shapes{};
  optiling::OpInfo c_op_info(input_shapes, DT_FLOAT);
  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compileInfo);
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateReduceTilingHandler(this->test_info_->name(),
                              "CommReduce",
                              nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(_op, runInfo, c_op_info));
}

TEST_F(ReduceTilingV3, ReduceTiling12) {
  using namespace optiling;

  std::vector<int64_t> input{64, 64};
  std::vector<int64_t> output{1,64};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto _op = op::ReduceSumD("ReduceSumD_12");
  _op.set_input_x(x1);
  _op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{_op};
  ge::Graph graph("ReduceTiling12");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);


  std::string compileInfo = R"({"_ori_axis": [-2],"_pattern": "CommReduce", "_zero_ub_factor": 32512, "_common_info": [32,1,8,1,1], "_pattern_info": [1], "_ub_info":[32512], "_ub_info_rf": [32512], "_reduce_shape_known": true, "_compile_pattern": 1, "_block_dims":{"1":32},
         "_atomic_flags":{"1": true},
         "_vars": {"1": []}})";

  // new interface
  optiling::utils::OpRunInfo runInfo;
  std::vector<std::vector<int64_t>> input_shapes{input,};
  optiling::OpInfo c_op_info(input_shapes, DT_FLOAT);
  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compileInfo);
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateReduceTilingHandler(this->test_info_->name(),
                              "CommReduce",
                              nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(_op, runInfo, c_op_info));
}

TEST_F(ReduceTilingV3, ReduceTiling13) {
  using namespace optiling;

  std::vector<int64_t> input{12456, 15};
  std::vector<int64_t> output{12456,1};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto _op = op::ReduceSumD("ReduceSumD_13");
  _op.set_input_x(x1);
  _op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{_op};
  ge::Graph graph("ReduceTiling13");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);


  std::string compileInfo = R"({"_ori_axis": [-1],"_pattern": "CommReduce", "_zero_ub_factor": 32512, "_common_info":
  [32,1,8,1,1,256], "_pattern_info": [1], "_ub_info":[32512], "_ub_info_rf": [32512],"_ub_info_pad": [20462],
  "_reduce_shape_known": true,
  "_compile_pattern": 1, "_block_dims":{"1":32},
         "_atomic_flags":{"1": true},
         "_vars": {"1": []}})";

  // new interface
  optiling::utils::OpRunInfo runInfo;
  std::vector<std::vector<int64_t>> input_shapes{input,};
  optiling::OpInfo c_op_info(input_shapes, DT_FLOAT);
  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compileInfo);
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateReduceTilingHandler(this->test_info_->name(),
                              "CommReduce",
                              nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(_op, runInfo, c_op_info));

}


TEST_F(ReduceTilingV3, ReduceTiling13_1) {
  using namespace optiling;

  std::vector<int64_t> input{12456, 15, 45};
  std::vector<int64_t> output{12456,1, 45};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto _op = op::ReduceSumD("ReduceSumD_13_1");
  _op.set_input_x(x1);
  _op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{_op};
  ge::Graph graph("ReduceTiling13_1");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);


  std::string compileInfo = R"({"_ori_axis": [1],"_pattern": "CommReduce", "_zero_ub_factor": 32512, "_common_info":
  [32,1,8,1,1,256], "_pattern_info": [1], "_ub_info":[32512], "_ub_info_rf": [32512],"_ub_info_pad": [20462],
  "_reduce_shape_known": true,
  "_compile_pattern": 1, "_block_dims":{"1":32},
         "_atomic_flags":{"1": true},
         "_vars": {"1": []}})";

  // new interface
  optiling::utils::OpRunInfo runInfo;
  std::vector<std::vector<int64_t>> input_shapes{input,};
  optiling::OpInfo c_op_info(input_shapes, DT_FLOAT);
  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compileInfo);
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateReduceTilingHandler(this->test_info_->name(),
                              "CommReduce",
                              nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(_op, runInfo, c_op_info));
}

TEST_F(ReduceTilingV3, ReduceTiling13_2) {
  using namespace optiling;

  std::vector<int64_t> input{12, 44444, 4};
  std::vector<int64_t> output{12, 1, 4};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto _op = op::ReduceSumD("ReduceSumD_13_2");
  _op.set_input_x(x1);
  _op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{_op};
  ge::Graph graph("ReduceTiling13_2");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);


  std::string compileInfo = R"({"_ori_axis": [1],"_pattern": "CommReduce", "_zero_ub_factor": 32512, "_common_info":
  [32,1,8,0,1,256], "_pattern_info": [1], "_ub_info":[32512], "_ub_info_rf": [32512],"_ub_info_pad": [20462],
  "_reduce_shape_known": true,
  "_compile_pattern": 1, "_block_dims":{"1":32},
         "_atomic_flags":{"1": true},
         "_vars": {"1": []}})";

  // new interface
  optiling::utils::OpRunInfo runInfo;
  std::vector<std::vector<int64_t>> input_shapes{input,};
  optiling::OpInfo c_op_info(input_shapes, DT_FLOAT);
  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compileInfo);
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateReduceTilingHandler(this->test_info_->name(),
                              "CommReduce",
                              nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(_op, runInfo, c_op_info));
}


TEST_F(ReduceTilingV3, ReduceTiling13_3) {
  using namespace optiling;

  std::vector<int64_t> input{3, 81920,2};
  std::vector<int64_t> output{3,1,2};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto _op = op::ReduceSumD("ReduceSumD_13_3");
  _op.set_input_x(x1);
  _op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{_op};
  ge::Graph graph("ReduceTiling13_3");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);


  std::string compileInfo = R"({"_ori_axis": [-1],"_pattern": "CommReduce", "_zero_ub_factor": 32512, "_common_info":
  [32,1,8,1,1,256], "_pattern_info": [1], "_ub_info":[32512], "_ub_info_rf": [32512],"_ub_info_pad": [20462],
  "_reduce_shape_known": true,
  "_compile_pattern": 1, "_block_dims":{"1":32},
         "_atomic_flags":{"1": true},
         "_vars": {"1": []}})";

  // new interface
  optiling::utils::OpRunInfo runInfo;
  std::vector<std::vector<int64_t>> input_shapes{input,};
  optiling::OpInfo c_op_info(input_shapes, DT_FLOAT);
  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compileInfo);
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateReduceTilingHandler(this->test_info_->name(),
                              "CommReduce",
                              nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(_op, runInfo, c_op_info));
}

TEST_F(ReduceTilingV3, ReduceTiling14) {
  using namespace optiling;

  std::vector<int64_t> input{12456, 15};
  std::vector<int64_t> output{12456,1};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto _op = op::ReduceSumD("ReduceSumD_14");
  _op.set_input_x(x1);
  _op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{_op};
  ge::Graph graph("ReduceTiling14");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string compileInfo = R"({"_ori_axis": [-1],"_pattern": "CommReduce", "_zero_ub_factor": 32512, "_common_info":
  [32,1,8,1,1,256,1], "_pattern_info": [1], "_ub_info":[32512], "_ub_info_rf": [32512],"_ub_info_pad": [20462],
  "_ub_info_transpose": [32512], "_reduce_shape_known": true, "_compile_pattern": 1, "_block_dims":{"1":32},
   "_atomic_flags":{"1": true}, "_vars": {"1": []}})";

  // new interface
  optiling::utils::OpRunInfo runInfo;
  std::vector<std::vector<int64_t>> input_shapes{input,};
  optiling::OpInfo c_op_info(input_shapes, DT_FLOAT);
  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compileInfo);
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateReduceTilingHandler(this->test_info_->name(),
                              "CommReduce",
                              nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(_op, runInfo, c_op_info));
}

static void ReduceSumCompute(std::vector<int64_t> inputA, std::vector<int64_t> inputB, std::vector<int32_t> axes,
                             std::vector<int64_t> output, ge::DataType dtypeA, ge::DataType dtypeB,
                             ge::DataType dtypeOutput, std::string compileInfo, bool isCustom, std::string caseName) {
  using namespace optiling;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputA));
  tensor_inputA.SetDataType(dtypeA);

  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputB));
  tensor_inputB.SetDataType(dtypeB);

  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(dtypeOutput);

  auto _op = op::ReduceSum(caseName.c_str());
  TENSOR_INPUT(_op, tensor_inputA, x);
  TENSOR_INPUT_CONST(_op, tensor_inputB, axes, (const uint8_t*)axes.data(), axes.size() * 4);
  TENSOR_OUTPUT(_op, tensor_output, y);

  optiling::utils::OpRunInfo runInfo;
  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compileInfo);
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateReduceTilingHandler(caseName,
                              "CommReduce",
                              nlohmann::json::parse(compileInfo));
  if (!isCustom) {
    ASSERT_TRUE(outer_compile_info->DoTiling(_op, runInfo));
  } else {
    std::vector<std::vector<int64_t>> input_shapes{inputA, inputB};
    optiling::OpInfo c_op_info(input_shapes, DT_FLOAT);
    ASSERT_TRUE(outer_compile_info->DoTiling(_op, runInfo, c_op_info));
  }
}

static void ReduceSumComputeInt64(std::vector<int64_t> inputA, std::vector<int64_t> inputB, std::vector<int64_t> axes,
                             std::vector<int64_t> output, ge::DataType dtypeA, ge::DataType dtypeB,
                             ge::DataType dtypeOutput, std::string compileInfo, bool isCustom, std::string caseName) {
  using namespace optiling;

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputA));
  tensor_inputA.SetDataType(dtypeA);

  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputB));
  tensor_inputB.SetDataType(dtypeB);

  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(dtypeOutput);

  auto _op = op::ReduceSum(caseName.c_str());
  TENSOR_INPUT(_op, tensor_inputA, x);
  TENSOR_INPUT_CONST(_op, tensor_inputB, axes, (const uint8_t*)axes.data(), axes.size() * 8);
  TENSOR_OUTPUT(_op, tensor_output, y);

  optiling::utils::OpRunInfo runInfo;
  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compileInfo);
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateReduceTilingHandler(caseName,
                              "CommReduce",
                              nlohmann::json::parse(compileInfo));
  if (!isCustom) {
    ASSERT_TRUE(outer_compile_info->DoTiling(_op, runInfo));
  } else {
    std::vector<std::vector<int64_t>> input_shapes{inputA, inputB};
    optiling::OpInfo c_op_info(input_shapes, DT_FLOAT);
    ASSERT_TRUE(outer_compile_info->DoTiling(_op, runInfo, c_op_info));
  }
}

TEST_F(ReduceTilingV3, ReduceSumTiling1) {
  std::string caseName = "ReduceSumTiling1";
  std::string compileInfo = R"({"_pattern": "CommReduce", "_common_info": [32,1,8,1,1], "_pattern_info": [5, 4, 9], "_ub_info_rf": [32512, 21376, 32512], "_ub_info": [32512, 21376, 16128], "_idx_before_reduce": 0, "_vars": {"4293966796": ["_dim_1", "_block_factor", "_ub_factor"], "4293866796": ["_dim_1", "_block_factor", "_ub_factor"], "500": ["_dim_1", "_block_factor", "_ub_factor"], "100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "4294966896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294866896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1000400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294966396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1000900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1100900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1200900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"]}, "_normal_vars": {"4293966796": ["_dim_1", "_block_factor", "_ub_factor"], "4293866796": ["_dim_1", "_block_factor", "_ub_factor"], "500": ["_dim_1", "_block_factor", "_ub_factor"], "100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "4294966896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294866896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1000400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294966396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1000900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1100900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1200900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"]}, "_attr_vars": {"4293966796": [], "4293866796": [], "500": [], "100500": [], "2147483647": [], "4294966896": [], "4294866896": [], "1000400": [], "1100400": [], "4294966396": [], "4294866396": [], "4294766396": [], "4292866396": [], "4292766396": [], "1000900": [], "1100900": [], "1200900": []}, "_custom_vars": {"4293966796": [], "4293866796": [], "500": [], "100500": [], "2147483647": [], "4294966896": [], "4294866896": [], "1000400": [], "1100400": [], "4294966396": [], "4294866396": [], "4294766396": [], "4292866396": [], "4292766396": [], "1000900": [], "1100900": [], "1200900": []}})";
  std::vector<int64_t> inputA{32,256};
  std::vector<int64_t> inputB{1};
  std::vector<int32_t> axes{0};
  std::vector<int64_t> output{1,256};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  bool isCustom = true;
  ReduceSumCompute(inputA, inputB, axes, output, dtypeA, dtypeB, dtypeOutput, compileInfo, isCustom, caseName);
}

TEST_F(ReduceTilingV3, ReduceSumTiling2) {
  std::string caseName = "ReduceSumTiling2";
  std::string compileInfo = R"({"_pattern": "CommReduce", "_common_info": [32,1,8,1,1], "_pattern_info": [5, 4, 9], "_ub_info_rf": [32512, 21376, 32512], "_ub_info": [32512, 21376, 16128], "_idx_before_reduce": 0, "_vars": {"4293966796": ["_dim_1", "_block_factor", "_ub_factor"], "4293866796": ["_dim_1", "_block_factor", "_ub_factor"], "500": ["_dim_1", "_block_factor", "_ub_factor"], "100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "4294966896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294866896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1000400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294966396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1000900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1100900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1200900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"]}, "_normal_vars": {"4293966796": ["_dim_1", "_block_factor", "_ub_factor"], "4293866796": ["_dim_1", "_block_factor", "_ub_factor"], "500": ["_dim_1", "_block_factor", "_ub_factor"], "100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "4294966896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294866896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1000400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294966396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1000900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1100900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1200900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"]}, "_attr_vars": {"4293966796": [], "4293866796": [], "500": [], "100500": [], "2147483647": [], "4294966896": [], "4294866896": [], "1000400": [], "1100400": [], "4294966396": [], "4294866396": [], "4294766396": [], "4292866396": [], "4292766396": [], "1000900": [], "1100900": [], "1200900": []}, "_custom_vars": {"4293966796": [], "4293866796": [], "500": [], "100500": [], "2147483647": [], "4294966896": [], "4294866896": [], "1000400": [], "1100400": [], "4294966396": [], "4294866396": [], "4294766396": [], "4292866396": [], "4292766396": [], "1000900": [], "1100900": [], "1200900": []}})";
  std::vector<int64_t> inputA{32,256};
  std::vector<int64_t> inputB{1};
  std::vector<int32_t> axes{0};
  std::vector<int64_t> output{1,256};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  bool isCustom = false;
  ReduceSumCompute(inputA, inputB, axes, output, dtypeA, dtypeB, dtypeOutput, compileInfo, isCustom, caseName);
}

TEST_F(ReduceTilingV3, ReduceSumTiling3) {
  std::string caseName = "ReduceSumTiling3";
  std::string compileInfo = R"({"_pattern": "CommReduce", "_common_info": [32,1,8,1,1], "_pattern_info": [5, 4, 9], "_ub_info_rf": [32512, 21376, 32512], "_ub_info": [32512, 21376, 16128], "_idx_before_reduce": 0, "_vars": {"4293966796": ["_dim_1", "_block_factor", "_ub_factor"], "4293866796": ["_dim_1", "_block_factor", "_ub_factor"], "500": ["_dim_1", "_block_factor", "_ub_factor"], "100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "4294966896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294866896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1000400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294966396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1000900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1100900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1200900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"]}, "_normal_vars": {"4293966796": ["_dim_1", "_block_factor", "_ub_factor"], "4293866796": ["_dim_1", "_block_factor", "_ub_factor"], "500": ["_dim_1", "_block_factor", "_ub_factor"], "100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "4294966896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294866896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1000400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294966396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1000900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1100900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1200900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"]}, "_attr_vars": {"4293966796": [], "4293866796": [], "500": [], "100500": [], "2147483647": [], "4294966896": [], "4294866896": [], "1000400": [], "1100400": [], "4294966396": [], "4294866396": [], "4294766396": [], "4292866396": [], "4292766396": [], "1000900": [], "1100900": [], "1200900": []}, "_custom_vars": {"4293966796": [], "4293866796": [], "500": [], "100500": [], "2147483647": [], "4294966896": [], "4294866896": [], "1000400": [], "1100400": [], "4294966396": [], "4294866396": [], "4294766396": [], "4292866396": [], "4292766396": [], "1000900": [], "1100900": [], "1200900": []}})";
  std::vector<int64_t> inputA{32,256};
  std::vector<int64_t> inputB{1};
  std::vector<int64_t> axes{0};
  std::vector<int64_t> output{1,256};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT64;
  ge::DataType dtypeOutput = dtypeA;
  bool isCustom = false;
  ReduceSumComputeInt64(inputA, inputB, axes, output, dtypeA, dtypeB, dtypeOutput, compileInfo, isCustom, caseName);
}

/* Test Case
 * **/
TEST_F(ReduceTilingV3, ReduceTiling_var_attr) {
  using namespace optiling;

  std::vector<int64_t> input{1};
  std::vector<int64_t> output{1};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto _op = op::ReduceSumD("ReduceSumD_1");
  _op.set_input_x(x1);
  _op.SetAttr("alpha", 12345);
  _op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{_op};
  ge::Graph graph("ReduceTiling1");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string compileInfo = R"({ "_ori_axis": [0],"_var_attr_mode":1,
                                 "_var_attrs": {"4293966796":[{"length":1,"name":"alpha","type":"int32",
                                               "src_type":"int32"}]},
                                 "_pattern": "CommReduce","push_status": 0,"_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5], "_ub_info": [16256], "_ub_info_rf": [16256], "_vars": {"-1000500": ["_dim_1_0", "_block_factor", "_ub_factor"]}})";

  optiling::utils::OpRunInfo runInfo;
  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compileInfo);
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateReduceTilingHandler(this->test_info_->name(),
                              "CommReduce",
                              nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(_op, runInfo));
  EXPECT_EQ(runInfo.GetBlockDim(), 1);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 1 12345 ");
}

/* Test Case
 * **/
TEST_F(ReduceTilingV3, ReduceTiling_var_attr_2) {
  using namespace optiling;

  std::vector<int64_t> input{64,64};
  std::vector<int64_t> output{1,64};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto _op = op::ReduceSumD("ReduceSumD_1_1");
  _op.set_input_x(x1);
  _op.SetAttr("alpha", 12345);
  _op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{_op};
  ge::Graph graph("ReduceTiling1_1");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string compileInfo = R"({"_ori_axis": [0],"_var_attr_mode":0,"_var_attrs": [{"length":1,"name":"alpha","type":"int32","src_type":"int32"}],
                            "_pattern": "CommReduce", "push_status": 0, "_zero_ub_factor": 32512, "_common_info": [32,1,8,1,1], "_pattern_info": [1], "_ub_info":[32512], "_ub_info_rf": [32512], "_reduce_shape_known": true, "_const_shape_post": true, "_compile_pattern": 1, "_block_dims":{"1":32},
                            "_atomic_flags":{"1": true}, "_vars": {"1": []}})";

  optiling::utils::OpRunInfo runInfo;
  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compileInfo);
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateReduceTilingHandler(this->test_info_->name(),
                              "CommReduce",
                              nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(_op, runInfo));
}


/* Test Case
 * **/
TEST_F(ReduceTilingV3, ReduceTiling_reduce_all) {
  using namespace optiling;

  std::vector<int64_t> input{500,64,64};
  std::vector<int64_t> output{500,1,64};

  TensorDesc tensor_input(ge::Shape(input), FORMAT_ND, DT_FLOAT);
  TensorDesc tensor_output(ge::Shape(output), FORMAT_ND, DT_FLOAT);

  auto x1 = op::Data("x1");
  x1.update_input_desc_x(tensor_input);
  x1.update_output_desc_y(tensor_input);

  auto _op = op::ReduceSumD("ReduceTiling_reduce_all");
  _op.set_input_x(x1);
  _op.update_output_desc_y(tensor_output);

  std::vector<Operator> inputs{x1};
  std::vector<Operator> outputs{_op};
  ge::Graph graph("ReduceTiling_reduce_all");
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  std::string compileInfo = R"({"_reduce_axes_type": 0, "_pattern": "CommReduce", "_zero_ub_factor":
  32512, "_common_info":[32,1,8,1,1,256], "_pattern_info": [5,4,9], "_ub_info":[32512, 32128, 16128],
  "_ub_info_rf": [32512, 21376, 32512],"_vars": {"1": []}})";

  optiling::utils::OpRunInfo runInfo;
  const nlohmann::json& parsed_compile_info = nlohmann::json::parse(compileInfo);
  std::shared_ptr<AutoTilingHandler> outer_compile_info = \
    CreateReduceTilingHandler(this->test_info_->name(),
                              "CommReduce",
                              nlohmann::json::parse(compileInfo));
  ASSERT_TRUE(outer_compile_info->DoTiling(_op, runInfo));
  EXPECT_EQ(runInfo.GetTilingKey(), 1100400);
}

