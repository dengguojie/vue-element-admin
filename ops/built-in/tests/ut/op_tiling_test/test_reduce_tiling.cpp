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

// class ReduceTiling : public testing::Test {
// protected:
//     static void SetUpTestCase() {
//       std::cout << "ReduceTiling SetUp" << std::endl;
//     }

//     static void TearDownTestCase() {
//       std::cout << "ReduceTiling TearDown" << std::endl;
//     }
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

// TEST_F(ReduceTiling, ReduceTiling1) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());


//   std::string compileInfo = R"({ "_ori_axis": [0], "_pattern": "CommReduce","push_status": 0,"_common_info": [32, 1, 8, 1, 1], "_pattern_info": [5], "_ub_info": [16256], "_ub_info_rf": [16256], "_vars": {"-1000500": ["_dim_1_0", "_block_factor", "_ub_factor"]}})";

//   std::vector<int64_t> input{1};
//   std::vector<int64_t> output{1};
//   std::string in_dtype = "float32";

//   TeOpTensor tensor_input;
//   tensor_input.shape = input;
//   tensor_input.dtype = in_dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = in_dtype;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_input);
//   tensor_arg.arg_type = TensorArgType::TA_SINGLE;
//   TeOpTensorArg tensor_arg_out;
//   tensor_arg_out.tensor.push_back(tensor_output);
//   tensor_arg_out.arg_type = TensorArgType::TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_arg);
//   opParas.outputs.push_back(tensor_arg_out);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "REDUCE__COUNTER__1";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
//   EXPECT_EQ(runInfo.block_dim, 1);
//   EXPECT_EQ(to_string(runInfo.tiling_data), "1 1 1 ");
// }

// TEST_F(ReduceTiling, ReduceTiling2) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());


//   std::string compileInfo = R"({ "_ori_axis": [2], "_pattern": "CommReduce", "push_status": 0, "_zero_ub_factor": 25600, "_vars": {"10": ["_dim_1", "_ub_factor"]}})";

//   std::vector<int64_t> input{2, 39, 0};
//   std::vector<int64_t> output{2, 39, 1};
//   std::string in_dtype = "float32";

//   TeOpTensor tensor_input;
//   tensor_input.shape = input;
//   tensor_input.dtype = in_dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = in_dtype;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_input);
//   tensor_arg.arg_type = TensorArgType::TA_SINGLE;
//   TeOpTensorArg tensor_arg_out;
//   tensor_arg_out.tensor.push_back(tensor_output);
//   tensor_arg_out.arg_type = TensorArgType::TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_arg);
//   opParas.outputs.push_back(tensor_arg_out);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "REDUCE__COUNTER__2";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
//   EXPECT_EQ(runInfo.block_dim, 1);
//   EXPECT_EQ(to_string(runInfo.tiling_data), "78 25600 ");
// }

// TEST_F(ReduceTiling, ReduceTiling3) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());


//   std::string compileInfo = R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "_zero_ub_factor": 32128, "_vars": {"110": ["_dim_2", "_ub_factor"]}})";

//   std::vector<int64_t> input{2, 39, 0};
//   std::vector<int64_t> output{};
//   std::string in_dtype = "float32";

//   TeOpTensor tensor_input;
//   tensor_input.shape = input;
//   tensor_input.dtype = in_dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = in_dtype;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_input);
//   tensor_arg.arg_type = TensorArgType::TA_SINGLE;
//   TeOpTensorArg tensor_arg_out;
//   tensor_arg_out.tensor.push_back(tensor_output);
//   tensor_arg_out.arg_type = TensorArgType::TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_arg);
//   opParas.outputs.push_back(tensor_arg_out);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "REDUCE__COUNTER__3";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
//   EXPECT_EQ(runInfo.block_dim, 1);
//   EXPECT_EQ(to_string(runInfo.tiling_data), "2 128 ");
// }

// TEST_F(ReduceTiling, ReduceTiling4) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

//   std::string compileInfo = R"({"_ori_axis": [0],"_pattern": "CommReduce", "push_status": 0, "_zero_ub_factor": 32512, "_common_info": [32,1,8,1,1], "_pattern_info": [1], "_ub_info":[32512], "_ub_info_rf": [32512], "_reduce_shape_known": true, "_const_shape_post": true, "_compile_pattern": 1, "_block_dims":{"1":32},
//       "_atomic_flags":{"1": true},
//       "_vars": {"1": []}})";
//   std::vector<int64_t> input{64,64};
//   std::vector<int64_t> output{1,64};
//   std::string in_dtype = "float32";
//   TeOpTensor tensor_input;
//   tensor_input.shape = input;
//   tensor_input.dtype = in_dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = in_dtype;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_input);
//   tensor_arg.arg_type = TensorArgType::TA_SINGLE;
//   TeOpTensorArg tensor_arg_out;
//   tensor_arg_out.tensor.push_back(tensor_output);
//   tensor_arg_out.arg_type = TensorArgType::TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_arg);
//   opParas.outputs.push_back(tensor_arg_out);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "REDUCE__COUNTER__4";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
// }

// TEST_F(ReduceTiling, ReduceTiling5) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());


//   std::string compileInfo = R"({ "_ori_axis": [0], "_pattern": "CommReduce","push_status": 0,"common_info": [32, 1, 8, 1, 1], "pattern_info": [20000], "ub_info": [16256], "_ub_info_rf": [16256], "_vars": {"-1000500": ["dim_1_0", "block_factor", "ub_factor"]}})";

//   std::vector<int64_t> input{1};
//   std::vector<int64_t> output{1};
//   std::string in_dtype = "float32";

//   TeOpTensor tensor_input;
//   tensor_input.shape = input;
//   tensor_input.dtype = in_dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = in_dtype;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_input);
//   tensor_arg.arg_type = TensorArgType::TA_SINGLE;
//   TeOpTensorArg tensor_arg_out;
//   tensor_arg_out.tensor.push_back(tensor_output);
//   tensor_arg_out.arg_type = TensorArgType::TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_arg);
//   opParas.outputs.push_back(tensor_arg_out);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "REDUCE__COUNTER__5";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(!iter->second.tiling_func_(opParas, op_compile_info, runInfo));
// }

// TEST_F(ReduceTiling, ReduceTiling6) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());


//   std::string compileInfo = R"({ "axes_idx": 0, "_pattern": "CommReduce","push_status": 0,"common_info": [32, 1, 8, 1, 1], "pattern_info": [20000], "ub_info": [16256], "_ub_info_rf": [16256], "_vars": {"-1000500": ["dim_1_0", "block_factor", "ub_factor"]}})";

//   std::vector<int64_t> input{1};
//   std::vector<int64_t> output{1};
//   std::string in_dtype = "float32";

//   TeOpTensor tensor_input;
//   tensor_input.shape = input;
//   tensor_input.dtype = in_dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = in_dtype;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_input);
//   tensor_arg.arg_type = TensorArgType::TA_SINGLE;
//   TeOpTensorArg tensor_arg_out;
//   tensor_arg_out.tensor.push_back(tensor_output);
//   tensor_arg_out.arg_type = TensorArgType::TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_arg);
//   opParas.outputs.push_back(tensor_arg_out);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "REDUCE__COUNTER__6";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(!iter->second.tiling_func_(opParas, op_compile_info, runInfo));
// }

// TEST_F(ReduceTiling, ReduceTiling7) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());


//   std::string compileInfo = R"({"_idx_before_reduce": 0, "_pattern": "CommReduce", "_common_info": [32, 1, 8, 1, 1], "_pattern_info": [-1], "_ub_info": [32512], "_ub_info_rf": [32512], "_reduce_shape_known": true, "_const_shape_post": true, "_compile_pattern": -1, "_block_dims": {"-1": 32}, "_atomic_flags": {"-1": false}})";

//   std::vector<int64_t> input{1200, 10};
//   std::vector<int64_t> output{1200, 10};
//   std::vector<int64_t> input_axis;
//   std::vector<int32_t> axis;
//   std::string in_dtype = "float32";

//   TeOpTensor tensor_input;
//   tensor_input.shape = input;
//   tensor_input.dtype = in_dtype;
//   TeOpTensor tensor_input_axis;
//   tensor_input_axis.shape = input_axis;
//   tensor_input_axis.dtype = "int32";
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = in_dtype;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_input);
//   tensor_arg.arg_type = TensorArgType::TA_SINGLE;
//   TeOpTensorArg tensor_input_axis_arg;
//   tensor_input_axis_arg.tensor.push_back(tensor_input_axis);
//   tensor_input_axis_arg.arg_type = TensorArgType::TA_SINGLE;
//   TeOpTensorArg tensor_arg_out;
//   tensor_arg_out.tensor.push_back(tensor_output);
//   tensor_arg_out.arg_type = TensorArgType::TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_arg);
//   ge::TensorDesc tensorDesc1;
//   tensorDesc1.SetDataType(ge::DT_INT32);
//   auto ge_tensor = ge::Tensor(tensorDesc1);
//   opParas.const_inputs["axes"] =
//   std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)axis.data(), axis.size() * 4, ge_tensor);
//   opParas.inputs.push_back(tensor_input_axis_arg);
//   opParas.outputs.push_back(tensor_arg_out);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "REDUCE__COUNTER__7";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
// }

// // FineTuning tune0
// TEST_F(ReduceTiling, ReduceTiling8) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

//   std::string compileInfo = R"({"_ori_axis": [0],"_pattern": "CommReduce", "push_status": 0,
//                                 "_zero_ub_factor": 32512, "_common_info": [32,1,16,0,1],
//                                 "_pattern_info": [5,4,9], "_ub_info":[21632, 21376, 21632],
//                                 "_ub_info_rf": [21632,16000,21632],
//                                 "_pattern": "CommReduce",
//                                 "_vars": {"1": []}})";
//   std::vector<int64_t> input{10000, 9, 80};
//   std::vector<int64_t> output{1, 9, 80};
//   std::string in_dtype = "float16";
//   TeOpTensor tensor_input;
//   tensor_input.shape = input;
//   tensor_input.dtype = in_dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = in_dtype;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_input);
//   tensor_arg.arg_type = TensorArgType::TA_SINGLE;
//   TeOpTensorArg tensor_arg_out;
//   tensor_arg_out.tensor.push_back(tensor_output);
//   tensor_arg_out.arg_type = TensorArgType::TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_arg);
//   opParas.outputs.push_back(tensor_arg_out);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "REDUCE__COUNTER__8";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
// }

// // FineTuning tune1
// TEST_F(ReduceTiling, ReduceTiling9) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

//   std::string compileInfo = R"({"_ori_axis": [0,2,3,4,5],"_pattern": "CommReduce",
//                                 "_common_info": [32,1,8,1,1],
//                                 "_pattern_info": [5,4,9], "_ub_info":[32512, 32128, 16128],
//                                 "_ub_info_rf": [32512, 21376, 32512],
//                                 "_pattern": "CommReduce",
//                                 "_vars": {"1": []}})";
//   std::vector<int64_t> input{16, 1, 8, 38, 1, 16, 16};
//   std::vector<int64_t> output{1, 16};
//   std::string in_dtype = "float32";
//   TeOpTensor tensor_input;
//   tensor_input.shape = input;
//   tensor_input.dtype = in_dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = in_dtype;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_input);
//   tensor_arg.arg_type = TensorArgType::TA_SINGLE;
//   TeOpTensorArg tensor_arg_out;
//   tensor_arg_out.tensor.push_back(tensor_output);
//   tensor_arg_out.arg_type = TensorArgType::TA_SINGLE;
//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_arg);
//   opParas.outputs.push_back(tensor_arg_out);
//   opParas.op_type = op_name;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "REDUCE__COUNTER__9";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
// }


// namespace optiling {
//     class VarAttrHelper {
//     public:
//         static void InitTeOpVarAttr(ge::OpDescPtr &op_desc, optiling::TeOpVarAttrArgs &attr);
//     };
// }

// TEST_F(ReduceTiling, ReduceTiling10) {
//   using namespace optiling;
//   std::string op_name = "AutoTiling";
//   auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
//   ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

//   std::string compileInfo = R"({"_ori_axis": [0],"_pattern": "CommReduce",
//                                 "_zero_ub_factor": 32512, "_common_info": [32,1,8,1,1],
//                                 "_pattern_info": [1], "_ub_info":[32512], "_ub_info_rf": [32512],
//                                 "_reduce_shape_known": true, "_const_shape_post": true,
//                                 "_compile_pattern": 1, "_block_dims":{"1":32},
//                                 "_atomic_flags":{"1": true},
//                                 "_attr_vars": { "1": [{"name": "Test", "type":"Int32"}] },
//                                 "_vars": {"1": []}})";
//   std::vector<int64_t> input{64,64};
//   std::vector<int64_t> output{1,64};
//   std::string in_dtype = "float32";
//   TeOpTensor tensor_input;
//   tensor_input.shape = input;
//   tensor_input.dtype = in_dtype;
//   TeOpTensor tensor_output;
//   tensor_output.shape = output;
//   tensor_output.dtype = in_dtype;
//   TeOpTensorArg tensor_arg;
//   tensor_arg.tensor.push_back(tensor_input);
//   tensor_arg.arg_type = TensorArgType::TA_SINGLE;
//   TeOpTensorArg tensor_arg_out;
//   tensor_arg_out.tensor.push_back(tensor_output);
//   tensor_arg_out.arg_type = TensorArgType::TA_SINGLE;

//   ge::OpDesc op_desc;
//   ge::OpDescPtr op_desc_ptr = std::make_shared<ge::OpDesc>(op_desc);
//   ge::AttrUtils::SetInt(op_desc_ptr, "Test", 11);
//   TeOpVarAttrArgs var_attrs;
//   VarAttrHelper::InitTeOpVarAttr(op_desc_ptr, var_attrs);

//   TeOpParas opParas;
//   opParas.inputs.push_back(tensor_arg);
//   opParas.outputs.push_back(tensor_arg_out);
//   opParas.op_type = op_name;
//   opParas.var_attrs = var_attrs;
//   OpCompileInfo op_compile_info;
//   op_compile_info.str = compileInfo;
//   op_compile_info.key = "REDUCE__COUNTER__10";
//   OpRunInfo runInfo;
//   ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
// }
