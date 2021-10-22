// #include <iostream>
// #include <fstream>
// #include <vector>

// #include <gtest/gtest.h>
// #include "graph/utils/op_desc_utils.h"
// #include "graph/utils/attr_utils.h"
// #define private public
// #define private public
#include "register/op_tiling_registry.h"

// using namespace std;

// class EletwiseTiling : public testing::Test {
//  protected:
//   static void SetUpTestCase() {
//     std::cout << "EletwiseTiling SetUp" << std::endl;
//   }

//   static void TearDownTestCase() {
//     std::cout << "EletwiseTiling TearDown" << std::endl;
//   }
// };

// static string to_string(const std::stringstream &tiling_data) {
//   auto data = tiling_data.str();
//   string result;
//   int32_t tmp = 0;
//   for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
//     memcpy(&tmp, data.c_str() + i, sizeof(tmp));
//     result += std::to_string(tmp);
//     result += " ";
//   }

//   return result;
// }

// TEST_F(EletwiseTiling, Eletwise_tiling1) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

//   // dynamic_op_add_267.static_op_add_269
//   std::string compileInfo = R"({ "_pattern": "ElemWise", "_fusion_index": [[0], [1]], "push_status": 0, "_flag_info": [false, false, true, true, true, false, false], "_base_info": {"320": [32, 4, 21840, 10920], "000": [32, 4, 21840, 10920]}, "_elewise_vars": { "232000000": [10001, 20000, 30000], "0": [10100], "1": [10100, 20000, 30000], "2": [10100, 20000, 30001], "4": [10100, 20001, 30001] }, "_vars": { "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_1_0"], "1": ["_dim_1_0", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_1_0", "_block_factor_0", "_ub_factor_1"], "4": ["_dim_1_0", "_block_factor_1", "_ub_factor_1"] } })";

//   std::vector<int64_t> inputA{1, 5824};
//   std::vector<int64_t> inputB{100, 1};
//   std::vector<int64_t> output{100, 5824};
//   std::string in_dtype = "float32";
//   std::string dtype = "float32";

//   TeOpTensor tensor_inputA;
//   tensor_inputA.shape = inputA;
//   tensor_inputA.dtype = in_dtype;
//   TeOpTensor tensor_inputB;
//   tensor_inputB.shape = inputB;
//   tensor_inputB.dtype = in_dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = dtype;
//   TeOpTensorArg tensor_argA;
//   tensor_argA.tensor.push_back(tensor_inputA);
//   tensor_argA.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_argB;
//   tensor_argB.tensor.push_back(tensor_inputB);
//   tensor_argB.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_output);
//   tensor_arg.arg_type = TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_argA);
//   opParas.inputs.push_back(tensor_argB);
//   opParas.outputs.push_back(tensor_arg);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "Eletwise_tiling1";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
//   EXPECT_EQ(runInfo.block_dim, 25);
//   EXPECT_EQ(to_string(runInfo.tiling_data), "5824 4 2 ");
// }

// TEST_F(EletwiseTiling, Eletwise_tiling2) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

//   // dynamic_op_exp_432.static_op_exp_433
//   std::string compileInfo = R"({ "_outs_uint1": false, "_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, false, true, false, false, false], "_base_info": {"100": [32, 4, 32768, 16384]}, "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";

//   std::vector<int64_t> inputA{1, 33, 1089};
//   std::vector<int64_t> output{1, 33, 1089};
//   std::string dtype = "float32";

//   TeOpTensor tensor_inputA;
//   tensor_inputA.shape = inputA;
//   tensor_inputA.dtype = dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = dtype;
//   TeOpTensorArg tensor_argA;
//   tensor_argA.tensor.push_back(tensor_inputA);
//   tensor_argA.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_output);
//   tensor_arg.arg_type = TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_argA);
//   opParas.outputs.push_back(tensor_arg);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "Eletwise_tiling2";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
//   EXPECT_EQ(runInfo.block_dim, 32);
//   EXPECT_EQ(to_string(runInfo.tiling_data), "35937 1152 1152 ");
// }

// TEST_F(EletwiseTiling, Eletwise_tiling3) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

//   // dynamic_op_cast_30.static_op_cast_30
//   std::string compileInfo = R"({ "_outs_uint1": false, "_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, false, true, false, false, false], "_base_info": {"100": [32, 4, 16384, 8192]}, "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ], "210010000": [ 10000, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ], "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";

//   std::vector<int64_t> inputA{128, 128, 128, 128};
//   std::vector<int64_t> output{128, 128, 128, 128};
//   std::string dtype = "uint32";

//   TeOpTensor tensor_inputA;
//   tensor_inputA.shape = inputA;
//   tensor_inputA.dtype = dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = dtype;
//   TeOpTensorArg tensor_argA;
//   tensor_argA.tensor.push_back(tensor_inputA);
//   tensor_argA.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_output);
//   tensor_arg.arg_type = TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_argA);
//   opParas.outputs.push_back(tensor_arg);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "Eletwise_tiling3";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
//   EXPECT_EQ(runInfo.block_dim, 32);
//   EXPECT_EQ(to_string(runInfo.tiling_data), "268435456 8388608 8192 ");
// }

// TEST_F(EletwiseTiling, Eletwise_tiling_mul1) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

//   // dynamic_op_cast_30.static_op_cast_30
//   std::string compileInfo = R"({ "_pattern": "Broadcast", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"100": [32, 4, 13104, 6552], "120": [32, 4, 10920, 5456]}, "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ], "210010000": [ 10000, 20000, 30000 ], "212000000": [10000, 10100, 10101, 10102], "212000001": [10000, 10100, 10101, 10102, 20000, 30000], "212000002": [10000, 10100, 10101, 10102, 20000, 30001], "212010002": [10000, 10100, 10101, 10102, 20000, 30001], "212000004": [10000, 10100, 10101, 10102, 20001, 30001], "212010004": [10000, 10100, 10101, 10102, 20001, 30001] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ], "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ], "210010000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ], "212000000": [ "_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_1_2"], "212000001": [ "_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_1_2", "_block_factor_0", "_ub_factor_0"], "212000002": [ "_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_1_2", "_block_factor_0", "_ub_factor_1"], "212010002": [ "_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_1_2", "_block_factor_0", "_ub_factor_1"], "212000004": [ "_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_1_2", "_block_factor_1", "_ub_factor_1"], "212010004": [ "_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_1_2", "_block_factor_1", "_ub_factor_1"]} })";

//   std::vector<std::vector<int64_t>> inputs {
//     {100, 1},
//     {100, 1860},
//     {100, 1}
//   };
//   std::vector<int64_t> output {100, 1860};
//   std::vector<std::string> input_types{"uint8", "float32", "float32"};
//   std::string output_dtypoe = "float32";

//   TeOpParas opParas;
//   for (size_t i = 0; i < inputs.size(); i++) {
//     TeOpTensor tensor_input;
//     TeOpTensorArg tensor_arg;
//     tensor_input.shape = inputs[i];
//     tensor_input.dtype = input_types[i];
//     tensor_arg.tensor.push_back(tensor_input);
//     tensor_arg.arg_type = TA_SINGLE;
//     opParas.inputs.push_back(tensor_arg);
//   }
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = output_dtypoe;
//   TeOpTensorArg tensor_output_arg;
//   tensor_output_arg.tensor.push_back(tensor_output);
//   tensor_output_arg.arg_type = TA_SINGLE;
//   opParas.outputs.push_back(tensor_output_arg);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "Eletwise_tiling_mul1";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
//   EXPECT_EQ(runInfo.block_dim, 25);
//   EXPECT_EQ(to_string(runInfo.tiling_data), "100 1 1860 1 4 4 ");
// }

// TEST_F(EletwiseTiling, Eletwise_tiling4) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

//   // dynamic_op_exp_432.static_op_exp_433
//   std::string compileInfo = R"({ "_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, "2", true, true, false, false, false], "_base_info": {"101": [32, 4, 32768, 16384]}, "_elewise_vars": { "210000000": [ 10001, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";

//   std::vector<int64_t> inputA{1, 33, 1089};
//   std::vector<int64_t> output{1, 33, 1089};
//   std::string dtype = "float32";

//   TeOpTensor tensor_inputA;
//   tensor_inputA.shape = inputA;
//   tensor_inputA.dtype = dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = dtype;
//   TeOpTensorArg tensor_argA;
//   tensor_argA.tensor.push_back(tensor_inputA);
//   tensor_argA.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_output);
//   tensor_arg.arg_type = TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_argA);
//   opParas.outputs.push_back(tensor_arg);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "Eletwise_tiling4";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(!iter->second.tiling_func_(opParas, op_compile_info, runInfo));
// }

// TEST_F(EletwiseTiling, Eletwise_tiling5) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

//   // dynamic_op_exp_432.static_op_exp_433
//   std::string compileInfo = R"({ "_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"110": [32, 4, 32768, 16384]}, "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";

//   std::vector<int64_t> inputA{1, 33, 1};
//   std::vector<int64_t> inputB{1, 33, 1089};
//   std::vector<int64_t> output{1, 33, 1089};
//   std::string dtype = "float32";

//   TeOpTensor tensor_inputA;
//   tensor_inputA.shape = inputA;
//   tensor_inputA.dtype = dtype;
//   TeOpTensor tensor_inputB;
//   tensor_inputB.shape = inputB;
//   tensor_inputB.dtype = dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = dtype;
//   TeOpTensorArg tensor_argA;
//   tensor_argA.tensor.push_back(tensor_inputA);
//   tensor_argA.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_argB;
//   tensor_argB.tensor.push_back(tensor_inputB);
//   tensor_argB.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_output);
//   tensor_arg.arg_type = TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_argA);
//   opParas.inputs.push_back(tensor_argB);
//   opParas.outputs.push_back(tensor_arg);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "Eletwise_tiling5";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(!iter->second.tiling_func_(opParas, op_compile_info, runInfo));
// }

// TEST_F(EletwiseTiling, Eletwise_tiling6) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

//   // dynamic_op_exp_432.static_op_exp_433
//   std::string compileInfo = R"({ "_outs_uint1": false, "_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, false, true, false, false, false], "_base_info": {"100": [32, 4, 32768, 16384]}, "_elewise_vars": { "2101000000": [ 10000, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";

//   std::vector<int64_t> inputA{1, 33, 1089};
//   std::vector<int64_t> output{1, 33, 1089};
//   std::string dtype = "float32";

//   TeOpTensor tensor_inputA;
//   tensor_inputA.shape = inputA;
//   tensor_inputA.dtype = dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = dtype;
//   TeOpTensorArg tensor_argA;
//   tensor_argA.tensor.push_back(tensor_inputA);
//   tensor_argA.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_output);
//   tensor_arg.arg_type = TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_argA);
//   opParas.outputs.push_back(tensor_arg);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "Eletwise_tiling6";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(!iter->second.tiling_func_(opParas, op_compile_info, runInfo));
// }

// TEST_F(EletwiseTiling, Eletwise_tiling7) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

//   // dynamic_op_exp_432.static_op_exp_433
//   std::string compileInfo = R"({ "_outs_uint1": false, "_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, false, true, false, false, false], "_base_info": {"100": [32, 4, 32768, 16384]}, "_elewise_vars": { "211000000": [ 10000, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";

//   std::vector<int64_t> inputA{1, 33, 1};
//   std::vector<int64_t> inputB{1, 33, 1089};
//   std::vector<int64_t> output{1, 33, 1089};
//   std::string dtype = "float32";

//   TeOpTensor tensor_inputA;
//   tensor_inputA.shape = inputA;
//   tensor_inputA.dtype = dtype;
//   TeOpTensor tensor_inputB;
//   tensor_inputB.shape = inputB;
//   tensor_inputB.dtype = dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = dtype;
//   TeOpTensorArg tensor_argA;
//   tensor_argA.tensor.push_back(tensor_inputA);
//   tensor_argA.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_argB;
//   tensor_argB.tensor.push_back(tensor_inputB);
//   tensor_argB.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_output);
//   tensor_arg.arg_type = TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_argA);
//   opParas.inputs.push_back(tensor_argB);
//   opParas.outputs.push_back(tensor_arg);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "Eletwise_tiling7";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(!iter->second.tiling_func_(opParas, op_compile_info, runInfo));
// }

// TEST_F(EletwiseTiling, Eletwise_tiling8) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

//   // dynamic_op_exp_432.static_op_exp_433
//   std::string compileInfo = R"({"_fusion_index":[1], "push_status": 0, "_pattern": "ElemWise", "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"120": [32, 4, 32768, 16384]}, "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";

//   std::vector<int64_t> inputA{1, 33, 1};
//   std::vector<int64_t> inputB{3, 33, 1089};
//   std::vector<int64_t> output{3, 33, 1089};
//   std::string dtype = "float32";

//   TeOpTensor tensor_inputA;
//   tensor_inputA.shape = inputA;
//   tensor_inputA.dtype = dtype;
//   TeOpTensor tensor_inputB;
//   tensor_inputB.shape = inputB;
//   tensor_inputB.dtype = dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = dtype;
//   TeOpTensorArg tensor_argA;
//   tensor_argA.tensor.push_back(tensor_inputA);
//   tensor_argA.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_argB;
//   tensor_argB.tensor.push_back(tensor_inputB);
//   tensor_argB.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_output);
//   tensor_arg.arg_type = TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_argA);
//   opParas.inputs.push_back(tensor_argB);
//   opParas.outputs.push_back(tensor_arg);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "Eletwise_tiling8";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(!iter->second.tiling_func_(opParas, op_compile_info, runInfo));
// }

// TEST_F(EletwiseTiling, Eletwise_tiling9) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

//   // dynamic_op_exp_432.static_op_exp_433
//   std::string compileInfo = R"({"_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, true, true, true, false, false, false], "_base_info": {"sdf": [32, 4, 32768, 16384]}, "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";

//   std::vector<int64_t> inputA{1, 33, 1};
//   std::vector<int64_t> inputB{3, 33, 1089};
//   std::vector<int64_t> output{3, 33, 1089};
//   std::string dtype = "float32";

//   TeOpTensor tensor_inputA;
//   tensor_inputA.shape = inputA;
//   tensor_inputA.dtype = dtype;
//   TeOpTensor tensor_inputB;
//   tensor_inputB.shape = inputB;
//   tensor_inputB.dtype = dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = dtype;
//   TeOpTensorArg tensor_argA;
//   tensor_argA.tensor.push_back(tensor_inputA);
//   tensor_argA.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_argB;
//   tensor_argB.tensor.push_back(tensor_inputB);
//   tensor_argB.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_output);
//   tensor_arg.arg_type = TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_argA);
//   opParas.inputs.push_back(tensor_argB);
//   opParas.outputs.push_back(tensor_arg);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "Eletwise_tiling9";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(!iter->second.tiling_func_(opParas, op_compile_info, runInfo));
// }

// TEST_F(EletwiseTiling, Eletwise_tiling10) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

//   // dynamic_op_exp_432.static_op_exp_433
//   std::string compileInfo = R"({"_fusion_index": [[0, 1]],"_const_shapes":[[64,64]],"_const_block_dims":[32],"_pattern": "ElemWise","_flag_info": [false, true, true, true, false, false, false],"_vars": { "100000000": [] },"push_status": 0})";
//   std::vector<int64_t> inputA{64,64};
//   std::vector<int64_t> inputB{64,64};
//   std::vector<int64_t> output{64,64};
//   std::string dtype = "float32";

//   TeOpTensor tensor_inputA;
//   tensor_inputA.shape = inputA;
//   tensor_inputA.dtype = dtype;
//   TeOpTensor tensor_inputB;
//   tensor_inputB.shape = inputB;
//   tensor_inputB.dtype = dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = dtype;
//   TeOpTensorArg tensor_argA;
//   tensor_argA.tensor.push_back(tensor_inputA);
//   tensor_argA.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_argB;
//   tensor_argB.tensor.push_back(tensor_inputB);
//   tensor_argB.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_output);
//   tensor_arg.arg_type = TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_argA);
//   opParas.inputs.push_back(tensor_argB);
//   opParas.outputs.push_back(tensor_arg);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "Eletwise_tiling10";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
// }

// TEST_F(EletwiseTiling, Eletwise_tiling11) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

//   // dynamic_op_exp_432.static_op_exp_433
//   std::string compileInfo = R"({ "push_status": 0, "_pattern": "ElemWise", "_flag_info": [false, false, false, true, false, false, false], "_base_info": {"100": [32, 4, 32768, 16384]}, "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";

//   std::vector<int64_t> inputA{1, 33, 0};
//   std::vector<int64_t> output{1, 33, 0};
//   std::string dtype = "float32";

//   TeOpTensor tensor_inputA;
//   tensor_inputA.shape = inputA;
//   tensor_inputA.dtype = dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = dtype;
//   TeOpTensorArg tensor_argA;
//   tensor_argA.tensor.push_back(tensor_inputA);
//   tensor_argA.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_output);
//   tensor_arg.arg_type = TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_argA);
//   opParas.outputs.push_back(tensor_arg);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "Eletwise_tiling11";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
//   EXPECT_EQ(runInfo.block_dim, 1);
//   EXPECT_EQ(to_string(runInfo.tiling_data), "");
// }

// TEST_F(EletwiseTiling, Eletwise_tiling12) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

//   // dynamic real div const tiling
//   std::string compileInfo = R"({ "_outs_uint1": false, "_flag_info": [true],"_base_info": {"000": [32, 2, 43680, 21840]}, "_broadcast_axis": [false], "_fusion_index": [[0], [1], [2]], "_pattern": "ElemWise", "push_status": 0})";
//   std::vector<int64_t> inputA{6643712,};
//   std::vector<int64_t> inputB{6643712,};
//   std::vector<int64_t> output{6643712,};
//   std::string dtype = "float16";

//   TeOpTensor tensor_inputA;
//   tensor_inputA.shape = inputA;
//   tensor_inputA.dtype = dtype;
//   TeOpTensor tensor_inputB;
//   tensor_inputB.shape = inputB;
//   tensor_inputB.dtype = dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = dtype;
//   TeOpTensorArg tensor_argA;
//   tensor_argA.tensor.push_back(tensor_inputA);
//   tensor_argA.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_argB;
//   tensor_argB.tensor.push_back(tensor_inputB);
//   tensor_argB.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_output);
//   tensor_arg.arg_type = TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_argA);
//   opParas.inputs.push_back(tensor_argB);
//   opParas.outputs.push_back(tensor_arg);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "Eletwise_tiling12";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
// }


// TEST_F(EletwiseTiling, Eletwise_tiling13) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

//   // dynamic real div const tiling
//   std::string compileInfo = R"({ "push_status": 0, "_pattern": "Broadcast", "_flag_info": [false, false, true, true, false, false, false], "_base_info": {"210": [32, 4, 21832, 10912]}, "_elewise_vars": { "221000001": [10000, 10001, 10100, 20000, 30000], "221000002": [10000, 10001, 10100, 20000, 30001] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] } })";
//   std::vector<int64_t> inputA{35, 45, 223};
//   std::vector<int64_t> inputB{45, 223};
//   std::vector<int64_t> output{35, 45, 223};
//   std::string dtype = "float16";

//   TeOpTensor tensor_inputA;
//   tensor_inputA.shape = inputA;
//   tensor_inputA.dtype = dtype;
//   TeOpTensor tensor_inputB;
//   tensor_inputB.shape = inputB;
//   tensor_inputB.dtype = dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = "uint8";
//   TeOpTensorArg tensor_argA;
//   tensor_argA.tensor.push_back(tensor_inputA);
//   tensor_argA.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_argB;
//   tensor_argB.tensor.push_back(tensor_inputB);
//   tensor_argB.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_output);
//   tensor_arg.arg_type = TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_argA);
//   opParas.inputs.push_back(tensor_argB);
//   opParas.outputs.push_back(tensor_arg);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "Eletwise_tiling13";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
// }

// namespace optiling {
//   class VarAttrHelper {
//    public:
//     static void InitTeOpVarAttr(ge::OpDescPtr &op_desc, optiling::TeOpVarAttrArgs &attr);
//   };
// }


// TEST_F(EletwiseTiling, Eletwise_tiling14) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

//   // dynamic_op_exp_432.static_op_exp_433
//   std::string compileInfo = R"({ "_outs_uint1": false, "_pattern": "ElemWise", "push_status": 0, "_flag_info": [false, false, false, true, false, false, false], "_base_info": {"100": [32, 4, 32768, 16384]}, "_elewise_vars": { "210000000": [ 10000, 20000, 30000 ] }, "_vars": { "210000000": [ "_dim_0_0", "_block_factor_0", "_ub_factor_0" ] }, "_attr_vars": { "210000000": [{"name": "tyun", "type":"Int32"}] } })";

//   std::vector<int64_t> inputA{1, 33, 1089};
//   std::vector<int64_t> output{1, 33, 1089};
//   std::string dtype = "float32";

//   TeOpTensor tensor_inputA;
//   tensor_inputA.shape = inputA;
//   tensor_inputA.dtype = dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = dtype;
//   TeOpTensorArg tensor_argA;
//   tensor_argA.tensor.push_back(tensor_inputA);
//   tensor_argA.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_output);
//   tensor_arg.arg_type = TA_SINGLE;
//   ge::OpDesc op_desc;
//   ge::OpDescPtr op_desc_ptr = std::make_shared<ge::OpDesc>(op_desc);
//   ge::AttrUtils::SetInt(op_desc_ptr, "tyun", 11);
//   TeOpVarAttrArgs var_attrs;
//   VarAttrHelper::InitTeOpVarAttr(op_desc_ptr, var_attrs);
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_argA);
//   opParas.outputs.push_back(tensor_arg);
//   opParas.op_type = op_name;
//   opParas.var_attrs = var_attrs;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "Eletwise_tiling14";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
//   EXPECT_EQ(runInfo.block_dim, 32);
//   EXPECT_EQ(to_string(runInfo.tiling_data), "35937 1152 1152 11 ");
// }

// TEST_F(EletwiseTiling, Eletwise_tiling15) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

//   // dynamic_op_add_267.static_op_add_269
//   std::string compileInfo = R"({ "_pattern": "ElemWise", "_fusion_index": [[0], [1]], "push_status": 0, "_flag_info": [false, false, true, true, true, false, false], "_base_info": {"320": [32, 4, 21840, 10920], "000": [32, 4, 21840, 10920]}, "_elewise_vars": { "232000000": [10001, 20000, 30000], "0": [10100], "1": [10100, 20000, 30000], "2": [10100, 20000, 30001], "4": [10100, 20001, 30001] }, "_vars": { "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_1_0"], "1": ["_dim_1_0", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_1_0", "_block_factor_0", "_ub_factor_1"], "4": ["_dim_1_0", "_block_factor_1", "_ub_factor_1"] }, "_attr_vars": { "232000000": [{"name": "tyun", "type":"Int32"}], "0": [{"name": "tyun", "type":"Int32"}], "1": [{"name": "tyun", "type":"Int32"}], "2": [{"name": "tyun", "type":"Int32"}], "4": [{"name": "tyun", "type":"Int32"}]} })";

//   std::vector<int64_t> inputA{1, 5824};
//   std::vector<int64_t> inputB{100, 1};
//   std::vector<int64_t> output{100, 5824};
//   std::string in_dtype = "float32";
//   std::string dtype = "float32";

//   TeOpTensor tensor_inputA;
//   tensor_inputA.shape = inputA;
//   tensor_inputA.dtype = in_dtype;
//   TeOpTensor tensor_inputB;
//   tensor_inputB.shape = inputB;
//   tensor_inputB.dtype = in_dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = dtype;
//   TeOpTensorArg tensor_argA;
//   tensor_argA.tensor.push_back(tensor_inputA);
//   tensor_argA.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_argB;
//   tensor_argB.tensor.push_back(tensor_inputB);
//   tensor_argB.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_output);
//   tensor_arg.arg_type = TA_SINGLE;
//   ge::OpDesc op_desc;
//   ge::OpDescPtr op_desc_ptr = std::make_shared<ge::OpDesc>(op_desc);
//   ge::AttrUtils::SetInt(op_desc_ptr, "tyun", 100);
//   TeOpVarAttrArgs var_attrs;
//   VarAttrHelper::InitTeOpVarAttr(op_desc_ptr, var_attrs);
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_argA);
//   opParas.inputs.push_back(tensor_argB);
//   opParas.outputs.push_back(tensor_arg);
//   opParas.op_type = op_name;
//   opParas.var_attrs = var_attrs;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "Eletwise_tiling15";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
//   EXPECT_EQ(runInfo.block_dim, 25);
//   EXPECT_EQ(to_string(runInfo.tiling_data), "5824 4 2 100 ");
// }

// TEST_F(EletwiseTiling, Eletwise_tiling16) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

//   // dynamic real div const tiling
//   std::string compileInfo = R"({"_fusion_index": [[0]], "_pattern": "Broadcast", "_outs_uint1": false, "_flag_info": [false, true, true, true, false, false, false], "_const_shapes": [[1], [48985]], "_const_block_dims": [32, 32], "_vars": {"100000000": [], "100000001": []}, "_normal_vars": {"100000000": [], "100000001": []}, "_attr_vars": {"100000000": [{"name": "tyun", "type":"Int32"}], "100000001": [{"name": "tyun", "type":"Int32"}]}, "_custom_vars": {"100000000": [], "100000001": []}, "_elewise_vars": {"100000000": [], "100000001": []}})";
//   std::vector<int64_t> inputA{48985};
//   std::vector<int64_t> inputB{48985};
//   std::vector<int64_t> output{48985};
//   std::string dtype = "float16";

//   TeOpTensor tensor_inputA;
//   tensor_inputA.shape = inputA;
//   tensor_inputA.dtype = dtype;
//   TeOpTensor tensor_inputB;
//   tensor_inputB.shape = inputB;
//   tensor_inputB.dtype = dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = dtype;
//   TeOpTensorArg tensor_argA;
//   tensor_argA.tensor.push_back(tensor_inputA);
//   tensor_argA.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_argB;
//   tensor_argB.tensor.push_back(tensor_inputB);
//   tensor_argB.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_output);
//   tensor_arg.arg_type = TA_SINGLE;
//   ge::OpDesc op_desc;
//   ge::OpDescPtr op_desc_ptr = std::make_shared<ge::OpDesc>(op_desc);
//   ge::AttrUtils::SetInt(op_desc_ptr, "tyun", 101);
//   TeOpVarAttrArgs var_attrs;
//   VarAttrHelper::InitTeOpVarAttr(op_desc_ptr, var_attrs);
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_argA);
//   opParas.inputs.push_back(tensor_argB);
//   opParas.outputs.push_back(tensor_arg);
//   opParas.op_type = op_name;
//   opParas.var_attrs = var_attrs;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "Eletwise_tiling16";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
//   EXPECT_EQ(to_string(runInfo.tiling_data), "101 ");
// }

// TEST_F(EletwiseTiling, Eletwise_tiling17) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

//   // dynamic real div const tiling
//   std::string compileInfo = R"({"_fusion_index": [[0], [1], [2], [3], [4]], "_pattern": "Broadcast", "_outs_uint1": false, "_flag_info": [false, false, true, true, true, false, true], "_base_info": {"100": [32, 2, 43680, 21840], "120": [32, 2, 28656, 14320], "121": [32, 2, 30704, 15344], "210": [32, 2, 30704, 15344], "320": [32, 2, 42320, 21152], "230": [32, 2, 42320, 21152], "000": [32, 2, 30704, 15344], "999": [32, 2, 30704, 15344]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212000004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212010004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "212100005": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "212100006": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "212100009": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_2"], "4": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_3"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_4"], "7": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_1"], "8": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_3"], "10": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_4"], "13": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_2"], "14": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_3"], "15": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_4"], "19": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_3", "_ub_factor_3"], "20": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_3", "_ub_factor_4"], "25": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_4", "_ub_factor_4"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_1"], "299900003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_2"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_3"], "299900006": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_1"], "299900007": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_2"], "299900008": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_3"], "299900011": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_2", "_ub_factor_2"], "299900012": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_2", "_ub_factor_3"], "299900016": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_3", "_ub_factor_3"]}, "_normal_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212000004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212010004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "212100005": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "212100006": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "212100009": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_2"], "4": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_3"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_4"], "7": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_1"], "8": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_3"], "10": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_4"], "13": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_2"], "14": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_3"], "15": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_4"], "19": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_3", "_ub_factor_3"], "20": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_3", "_ub_factor_4"], "25": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_4", "_ub_factor_4"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_1"], "299900003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_2"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_3"], "299900006": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_1"], "299900007": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_2"], "299900008": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_3"], "299900011": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_2", "_ub_factor_2"], "299900012": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_2", "_ub_factor_3"], "299900016": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_3", "_ub_factor_3"]}, "_attr_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "212000004": [], "212010004": [], "212100000": [], "212100001": [], "212100002": [], "212100003": [], "212100005": [], "212100006": [], "212100009": [], "221000000": [], "221000001": [], "221000002": [], "221000004": [], "232000000": [], "223000000": [], "0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "7": [], "8": [], "9": [], "10": [], "13": [], "14": [], "15": [], "19": [], "20": [], "25": [], "299900000": [], "299900001": [], "299900002": [], "299900003": [], "299900004": [], "299900006": [], "299900007": [], "299900008": [], "299900011": [], "299900012": [], "299900016": []}, "_custom_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "212000004": [], "212010004": [], "212100000": [], "212100001": [], "212100002": [], "212100003": [], "212100005": [], "212100006": [], "212100009": [], "221000000": [], "221000001": [], "221000002": [], "221000004": [], "232000000": [], "223000000": [], "0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "7": [], "8": [], "9": [], "10": [], "13": [], "14": [], "15": [], "19": [], "20": [], "25": [], "299900000": [], "299900001": [], "299900002": [], "299900003": [], "299900004": [], "299900006": [], "299900007": [], "299900008": [], "299900011": [], "299900012": [], "299900016": []}, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "212000000": [10000, 10100, 10101], "212000001": [10000, 10100, 10101, 20000, 30000], "212000002": [10000, 10100, 10101, 20000, 30001], "212010002": [10000, 10100, 10101, 20000, 30001], "212000004": [10000, 10100, 10101, 20001, 30001], "212010004": [10000, 10100, 10101, 20001, 30001], "212100000": [10000, 10100, 10101, 10200], "212100001": [10000, 10100, 10101, 10200, 20000, 30000], "212100002": [10000, 10100, 10101, 10200, 20000, 30001], "212100003": [10000, 10100, 10101, 10200, 20000, 30002], "212100005": [10000, 10100, 10101, 10200, 20001, 30001], "212100006": [10000, 10100, 10101, 10200, 20001, 30002], "212100009": [10000, 10100, 10101, 10200, 20002, 30002], "221000000": [10000, 10001, 10100], "221000001": [10000, 10001, 10100, 20000, 30000], "221000002": [10000, 10001, 10100, 20000, 30001], "221000004": [10000, 10001, 10100, 20001, 30001], "232000000": [10001, 20000, 30000], "223000000": [10000, 20000, 30000], "0": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401], "1": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30000], "2": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30001], "3": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30002], "4": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30003], "5": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30004], "7": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30001], "8": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30002], "9": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30003], "10": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30004], "13": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20002, 30002], "14": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20002, 30003], "15": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20002, 30004], "19": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20003, 30003], "20": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20003, 30004], "25": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20004, 30004], "299900000": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301], "299900001": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30000], "299900002": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30001], "299900003": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30002], "299900004": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30003], "299900006": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20001, 30001], "299900007": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20001, 30002], "299900008": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20001, 30003], "299900011": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20002, 30002], "299900012": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20002, 30003], "299900016": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20003, 30003]}})";
//   std::vector<int64_t> inputA{28, 1, 35, 45, 223};
//   std::vector<int64_t> inputB{28, 5, 35, 1, 1};
//   std::vector<int64_t> output{28, 5, 35, 45, 223};
//   std::string dtype = "float16";

//   TeOpTensor tensor_inputA;
//   tensor_inputA.shape = inputA;
//   tensor_inputA.dtype = dtype;
//   TeOpTensor tensor_inputB;
//   tensor_inputB.shape = inputB;
//   tensor_inputB.dtype = dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = "float16";
//   TeOpTensorArg tensor_argA;
//   tensor_argA.tensor.push_back(tensor_inputA);
//   tensor_argA.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_argB;
//   tensor_argB.tensor.push_back(tensor_inputB);
//   tensor_argB.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_output);
//   tensor_arg.arg_type = TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_argA);
//   opParas.inputs.push_back(tensor_argB);
//   opParas.outputs.push_back(tensor_arg);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "Eletwise_tiling17";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
//   EXPECT_EQ(runInfo.block_dim, 28);
//   EXPECT_EQ(to_string(runInfo.tiling_data), "28 28 1 5 35 35 10035 1 5 3 ");
// }

// TEST_F(EletwiseTiling, Eletwise_tiling18) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

//   // dynamic real div const tiling
//   std::string compileInfo = R"({"_fusion_index": [[0], [1], [2], [3], [4]], "_pattern": "Broadcast", "_outs_uint1": false, "_flag_info": [false, false, true, true, true, false, true], "_base_info": {"100": [32, 2, 43680, 21840], "120": [32, 2, 28656, 14320], "121": [32, 2, 30704, 15344], "210": [32, 2, 30704, 15344], "320": [32, 2, 42320, 21152], "230": [32, 2, 42320, 21152], "000": [32, 2, 30704, 15344], "999": [32, 2, 30704, 15344]}, "_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212000004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212010004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "212100005": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "212100006": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "212100009": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_2"], "4": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_3"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_4"], "7": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_1"], "8": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_3"], "10": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_4"], "13": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_2"], "14": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_3"], "15": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_4"], "19": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_3", "_ub_factor_3"], "20": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_3", "_ub_factor_4"], "25": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_4", "_ub_factor_4"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_1"], "299900003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_2"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_3"], "299900006": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_1"], "299900007": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_2"], "299900008": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_3"], "299900011": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_2", "_ub_factor_2"], "299900012": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_2", "_ub_factor_3"], "299900016": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_3", "_ub_factor_3"]}, "_normal_vars": {"210000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "210010000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "212000000": ["_dim_0_0", "_dim_1_0", "_dim_1_1"], "212000001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_0"], "212000002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212010002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_0", "_ub_factor_1"], "212000004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212010004": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_block_factor_1", "_ub_factor_1"], "212100000": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0"], "212100001": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_0"], "212100002": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_1"], "212100003": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_0", "_ub_factor_2"], "212100005": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_1"], "212100006": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_1", "_ub_factor_2"], "212100009": ["_dim_0_0", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_block_factor_2", "_ub_factor_2"], "221000000": ["_dim_0_0", "_dim_0_1", "_dim_1_0"], "221000001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_0"], "221000002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_0", "_ub_factor_1"], "221000004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_block_factor_1", "_ub_factor_1"], "232000000": ["_dim_0_1", "_block_factor_0", "_ub_factor_0"], "223000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1"], "1": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_0"], "2": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_1"], "3": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_2"], "4": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_3"], "5": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_0", "_ub_factor_4"], "7": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_1"], "8": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_2"], "9": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_3"], "10": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_1", "_ub_factor_4"], "13": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_2"], "14": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_3"], "15": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_2", "_ub_factor_4"], "19": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_3", "_ub_factor_3"], "20": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_3", "_ub_factor_4"], "25": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_dim_4_0", "_dim_4_1", "_block_factor_4", "_ub_factor_4"], "299900000": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1"], "299900001": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_0"], "299900002": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_1"], "299900003": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_2"], "299900004": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_0", "_ub_factor_3"], "299900006": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_1"], "299900007": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_2"], "299900008": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_1", "_ub_factor_3"], "299900011": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_2", "_ub_factor_2"], "299900012": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_2", "_ub_factor_3"], "299900016": ["_dim_0_0", "_dim_0_1", "_dim_1_0", "_dim_1_1", "_dim_2_0", "_dim_2_1", "_dim_3_0", "_dim_3_1", "_block_factor_3", "_ub_factor_3"]}, "_attr_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "212000004": [], "212010004": [], "212100000": [], "212100001": [], "212100002": [], "212100003": [], "212100005": [], "212100006": [], "212100009": [], "221000000": [], "221000001": [], "221000002": [], "221000004": [], "232000000": [], "223000000": [], "0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "7": [], "8": [], "9": [], "10": [], "13": [], "14": [], "15": [], "19": [], "20": [], "25": [], "299900000": [], "299900001": [], "299900002": [], "299900003": [], "299900004": [], "299900006": [], "299900007": [], "299900008": [], "299900011": [], "299900012": [], "299900016": []}, "_custom_vars": {"210000000": [], "210010000": [], "212000000": [], "212000001": [], "212000002": [], "212010002": [], "212000004": [], "212010004": [], "212100000": [], "212100001": [], "212100002": [], "212100003": [], "212100005": [], "212100006": [], "212100009": [], "221000000": [], "221000001": [], "221000002": [], "221000004": [], "232000000": [], "223000000": [], "0": [], "1": [], "2": [], "3": [], "4": [], "5": [], "7": [], "8": [], "9": [], "10": [], "13": [], "14": [], "15": [], "19": [], "20": [], "25": [], "299900000": [], "299900001": [], "299900002": [], "299900003": [], "299900004": [], "299900006": [], "299900007": [], "299900008": [], "299900011": [], "299900012": [], "299900016": []}, "_elewise_vars": {"210000000": [10000, 20000, 30000], "210010000": [10000, 20000, 30000], "212000000": [10000, 10100, 10101], "212000001": [10000, 10100, 10101, 20000, 30000], "212000002": [10000, 10100, 10101, 20000, 30001], "212010002": [10000, 10100, 10101, 20000, 30001], "212000004": [10000, 10100, 10101, 20001, 30001], "212010004": [10000, 10100, 10101, 20001, 30001], "212100000": [10000, 10100, 10101, 10200], "212100001": [10000, 10100, 10101, 10200, 20000, 30000], "212100002": [10000, 10100, 10101, 10200, 20000, 30001], "212100003": [10000, 10100, 10101, 10200, 20000, 30002], "212100005": [10000, 10100, 10101, 10200, 20001, 30001], "212100006": [10000, 10100, 10101, 10200, 20001, 30002], "212100009": [10000, 10100, 10101, 10200, 20002, 30002], "221000000": [10000, 10001, 10100], "221000001": [10000, 10001, 10100, 20000, 30000], "221000002": [10000, 10001, 10100, 20000, 30001], "221000004": [10000, 10001, 10100, 20001, 30001], "232000000": [10001, 20000, 30000], "223000000": [10000, 20000, 30000], "0": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401], "1": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30000], "2": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30001], "3": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30002], "4": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30003], "5": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20000, 30004], "7": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30001], "8": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30002], "9": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30003], "10": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20001, 30004], "13": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20002, 30002], "14": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20002, 30003], "15": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20002, 30004], "19": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20003, 30003], "20": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20003, 30004], "25": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 10400, 10401, 20004, 30004], "299900000": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301], "299900001": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30000], "299900002": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30001], "299900003": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30002], "299900004": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20000, 30003], "299900006": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20001, 30001], "299900007": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20001, 30002], "299900008": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20001, 30003], "299900011": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20002, 30002], "299900012": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20002, 30003], "299900016": [10000, 10001, 10100, 10101, 10200, 10201, 10300, 10301, 20003, 30003]}})";
//   std::vector<int64_t> inputA{1, 1, 1, 112, 22};
//   std::vector<int64_t> inputB{32, 5, 25, 1, 22};
//   std::vector<int64_t> output{32, 5, 25, 112, 22};
//   std::string dtype = "float16";

//   TeOpTensor tensor_inputA;
//   tensor_inputA.shape = inputA;
//   tensor_inputA.dtype = dtype;
//   TeOpTensor tensor_inputB;
//   tensor_inputB.shape = inputB;
//   tensor_inputB.dtype = dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = "float16";
//   TeOpTensorArg tensor_argA;
//   tensor_argA.tensor.push_back(tensor_inputA);
//   tensor_argA.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_argB;
//   tensor_argB.tensor.push_back(tensor_inputB);
//   tensor_argB.arg_type = TA_SINGLE;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_output);
//   tensor_arg.arg_type = TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_argA);
//   opParas.inputs.push_back(tensor_argB);
//   opParas.outputs.push_back(tensor_arg);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "Eletwise_tiling18";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
//   EXPECT_EQ(runInfo.block_dim, 32);
//   EXPECT_EQ(to_string(runInfo.tiling_data), "1 4000 14 1 8 1 22 22 125 12 ");
// }