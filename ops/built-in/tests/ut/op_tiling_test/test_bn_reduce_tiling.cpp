#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class BNReduceTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "BNReduceTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "BNReduceTiling TearDown" << std::endl;
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

static std::string g_compile_info = R"({ "ori_axis": [0, 2, 3], "_ori_axis": [0, 2, 3], "_pattern": "bn_reduce", "push_status": 1,
    "_common_info": [8192, 32, 1, 8, 1, 1], "_vars": {
        "-5310": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "block_factor_1", "ub_factor_0"],
        "-2205311": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "block_factor_1", "ub_factor_0"],
        "-1005311": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "block_factor_1", "ub_factor_0"],
        "-4205310": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "block_factor_1", "ub_factor_0"],
        "-205310": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "block_factor_1", "ub_factor_0"],
        "1000900": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "block_factor_1", "ub_factor_0"],
        "3304100": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "block_factor_1", "ub_factor_0"],
        "3204100": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "block_factor_1", "ub_factor_0"]
     }})";


// customised cut N
TEST_F(BNReduceTiling, BNReduceTiling_cut_n) {
    using namespace optiling;
    std::string op_name = "BNTrainingReduce";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::vector<int64_t> input{64, 1, 10, 2, 16};
    std::vector<int64_t> output{1, 1, 1, 1, 16};
    std::string in_dtype = "float32";

    TeOpTensor tensor_input;
    tensor_input.shape = input;
    tensor_input.dtype = in_dtype;
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = in_dtype;
    TeOpTensorArg tensor_arg;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_arg_out;
    tensor_arg_out.tensor.push_back(tensor_output);
    tensor_arg_out.arg_type = TensorArgType::TA_SINGLE;
    TeOpParas opParas;
    opParas.inputs.push_back(tensor_arg);
    opParas.outputs.push_back(tensor_arg_out);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = g_compile_info;
    op_compile_info.key = "REDUCE__COUNTER__";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 32);
    EXPECT_EQ(to_string(runInfo.tiling_data), "64 1 10 2 32 2 ");
}


// customised cut h twice
TEST_F(BNReduceTiling, BNReduceTilingcut_h_twice) {
    using namespace optiling;
    std::string op_name = "BNTrainingReduce";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::vector<int64_t> input{2, 1, 64, 8, 16};
    std::vector<int64_t> output{1, 1, 1, 1, 16};
    std::string in_dtype = "float32";

    TeOpTensor tensor_input;
    tensor_input.shape = input;
    tensor_input.dtype = in_dtype;
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = in_dtype;
    TeOpTensorArg tensor_arg;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_arg_out;
    tensor_arg_out.tensor.push_back(tensor_output);
    tensor_arg_out.arg_type = TensorArgType::TA_SINGLE;
    TeOpParas opParas;
    opParas.inputs.push_back(tensor_arg);
    opParas.outputs.push_back(tensor_arg_out);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = g_compile_info;
    op_compile_info.key = "REDUCE__COUNTER__";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 32);
    EXPECT_EQ(to_string(runInfo.tiling_data), "2 1 64 8 32 4 ");
}


// customised cut h fuse 
TEST_F(BNReduceTiling, BNReduceTilingcut_h_fuse) {
    using namespace optiling;
    std::string op_name = "BNTrainingReduce";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::vector<int64_t> input{2, 1, 64, 8, 16};
    std::vector<int64_t> output{1, 1, 1, 1, 16};
    std::string in_dtype = "float32";

    TeOpTensor tensor_input;
    tensor_input.shape = input;
    tensor_input.dtype = in_dtype;
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = in_dtype;
    TeOpTensorArg tensor_arg;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_arg_out;
    tensor_arg_out.tensor.push_back(tensor_output);
    tensor_arg_out.arg_type = TensorArgType::TA_SINGLE;
    TeOpParas opParas;
    opParas.inputs.push_back(tensor_arg);
    opParas.outputs.push_back(tensor_arg_out);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = g_compile_info;
    op_compile_info.key = "REDUCE__COUNTER__";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 32);
    EXPECT_EQ(to_string(runInfo.tiling_data), "2 1 64 8 32 4 ");
}


// customised cut c1
TEST_F(BNReduceTiling, BNReduceTilingcut_c1) {
    using namespace optiling;
    std::string op_name = "BNTrainingReduce";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::vector<int64_t> input{16, 128, 16, 16, 16};
    std::vector<int64_t> output{1, 128, 1, 1, 16};
    std::string in_dtype = "float32";

    TeOpTensor tensor_input;
    tensor_input.shape = input;
    tensor_input.dtype = in_dtype;
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = in_dtype;
    TeOpTensorArg tensor_arg;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_arg_out;
    tensor_arg_out.tensor.push_back(tensor_output);
    tensor_arg_out.arg_type = TensorArgType::TA_SINGLE;
    TeOpParas opParas;
    opParas.inputs.push_back(tensor_arg);
    opParas.outputs.push_back(tensor_arg_out);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = g_compile_info;
    op_compile_info.key = "REDUCE__COUNTER__";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 32);
    EXPECT_EQ(to_string(runInfo.tiling_data), "16 128 16 16 4 2 ");
}

// customised cut default
TEST_F(BNReduceTiling, BNReduceTiling_default) {
    using namespace optiling;
    std::string op_name = "BNTrainingReduce";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::vector<int64_t> input{1, 9, 4, 4, 16};
    std::vector<int64_t> output{1, 9, 1, 1, 16};
    std::string in_dtype = "float32";

    TeOpTensor tensor_input;
    tensor_input.shape = input;
    tensor_input.dtype = in_dtype;
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = in_dtype;
    TeOpTensorArg tensor_arg;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_arg_out;
    tensor_arg_out.tensor.push_back(tensor_output);
    tensor_arg_out.arg_type = TensorArgType::TA_SINGLE;
    TeOpParas opParas;
    opParas.inputs.push_back(tensor_arg);
    opParas.outputs.push_back(tensor_arg_out);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = g_compile_info;
    op_compile_info.key = "REDUCE__COUNTER__";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 9);
    EXPECT_EQ(to_string(runInfo.tiling_data), "1 9 4 4 9 4 ");
}


// atomic B1U0
TEST_F(BNReduceTiling, BNReduceTiling_B1U0) {
    using namespace optiling;
    std::string op_name = "BNTrainingReduce";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::vector<int64_t> input{4, 1, 24, 24, 16};
    std::vector<int64_t> output{1, 1, 1, 1, 16};
    std::string in_dtype = "float32";

    TeOpTensor tensor_input;
    tensor_input.shape = input;
    tensor_input.dtype = in_dtype;
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = in_dtype;
    TeOpTensorArg tensor_arg;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_arg_out;
    tensor_arg_out.tensor.push_back(tensor_output);
    tensor_arg_out.arg_type = TensorArgType::TA_SINGLE;
    TeOpParas opParas;
    opParas.inputs.push_back(tensor_arg);
    opParas.outputs.push_back(tensor_arg_out);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = g_compile_info;
    op_compile_info.key = "REDUCE__COUNTER__";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 1);
    EXPECT_EQ(to_string(runInfo.tiling_data), "4 1 24 24 1 12 ");
}


// atomic B3U3
TEST_F(BNReduceTiling, BNReduceTiling_B3U3) {
    using namespace optiling;
    std::string op_name = "BNTrainingReduce";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::vector<int64_t> input{16, 4, 147, 147, 16};
    std::vector<int64_t> output{1, 4, 1, 1, 16};
    std::string in_dtype = "float32";

    TeOpTensor tensor_input;
    tensor_input.shape = input;
    tensor_input.dtype = in_dtype;
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = in_dtype;
    TeOpTensorArg tensor_arg;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_arg_out;
    tensor_arg_out.tensor.push_back(tensor_output);
    tensor_arg_out.arg_type = TensorArgType::TA_SINGLE;
    TeOpParas opParas;
    opParas.inputs.push_back(tensor_arg);
    opParas.outputs.push_back(tensor_arg_out);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = g_compile_info;
    op_compile_info.key = "REDUCE__COUNTER__";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 32);
    EXPECT_EQ(to_string(runInfo.tiling_data), "1 16 4 21609 16 10805 512 ");
}


// atomic B3U2
TEST_F(BNReduceTiling, BNReduceTiling_B3U2) {
    using namespace optiling;
    std::string op_name = "BNTrainingReduce";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::vector<int64_t> input{4, 16, 56, 56, 16};
    std::vector<int64_t> output{1, 16, 1, 1, 16};
    std::string in_dtype = "float32";

    TeOpTensor tensor_input;
    tensor_input.shape = input;
    tensor_input.dtype = in_dtype;
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = in_dtype;
    TeOpTensorArg tensor_arg;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_arg_out;
    tensor_arg_out.tensor.push_back(tensor_output);
    tensor_arg_out.arg_type = TensorArgType::TA_SINGLE;
    TeOpParas opParas;
    opParas.inputs.push_back(tensor_arg);
    opParas.outputs.push_back(tensor_arg_out);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = g_compile_info;
    op_compile_info.key = "REDUCE__COUNTER__";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 32);
    EXPECT_EQ(to_string(runInfo.tiling_data), "1 4 16 3136 16 392 1 ");
}


// const
TEST_F(BNReduceTiling, BNReduceTiling_const) {
    std::string compile_info = R"({ "ori_axis": [0, 2, 3], "_ori_axis": [0, 2, 3], "_pattern": "bn_reduce", "push_status": 1,
        "_common_info": [8192, 32, 1, 8, 1, 1], "_reduce_shape_known": true, "_vars": {
            "-5310": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "block_factor_1", "ub_factor_0"],
            "-2205311": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "block_factor_1", "ub_factor_0"],
            "-1005311": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "block_factor_1", "ub_factor_0"],
            "-4205310": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "block_factor_1", "ub_factor_0"],
            "-205310": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "block_factor_1", "ub_factor_0"],
            "1000900": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "block_factor_1", "ub_factor_0"],
            "3304100": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "block_factor_1", "ub_factor_0"],
            "3204100": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "block_factor_1", "ub_factor_0"]
         }})";

    using namespace optiling;
    std::string op_name = "BNTrainingReduce";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::vector<int64_t> input{8, 8, 56, 56, 16};
    std::vector<int64_t> output{1, 8, 1, 1, 16};
    std::string in_dtype = "float32";

    TeOpTensor tensor_input;
    tensor_input.shape = input;
    tensor_input.dtype = in_dtype;
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = in_dtype;
    TeOpTensorArg tensor_arg;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_arg_out;
    tensor_arg_out.tensor.push_back(tensor_output);
    tensor_arg_out.arg_type = TensorArgType::TA_SINGLE;
    TeOpParas opParas;
    opParas.inputs.push_back(tensor_arg);
    opParas.outputs.push_back(tensor_arg_out);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str =compile_info;
    op_compile_info.key = "REDUCE__COUNTER__CONST";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 32);
    EXPECT_EQ(to_string(runInfo.tiling_data), "3 784 3 392 0 0 ");
}

// const_post
TEST_F(BNReduceTiling, BNReduceTiling_const_post) {
    std::string compile_info = R"({ "ori_axis": [0, 2, 3], "_ori_axis": [0, 2, 3], "_pattern": "bn_reduce", "push_status": 1,
        "_common_info": [8192, 32, 1, 8, 1, 1], "_reduce_shape_known": true, "_const_shape_post": true,
        "_block_dims" : {"134": 32}, "_atomic_flags" : {"134": true}, 
        "_vars": {
            "3304100": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "block_factor_1", "ub_factor_0"],
            "3204100": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "block_factor_1", "ub_factor_0"]
         }})";

    using namespace optiling;
    std::string op_name = "BNTrainingReduce";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::vector<int64_t> input{8, 8, 56, 56, 16};
    std::vector<int64_t> output{1, 8, 1, 1, 16};
    std::string in_dtype = "float32";

    TeOpTensor tensor_input;
    tensor_input.shape = input;
    tensor_input.dtype = in_dtype;
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = in_dtype;
    TeOpTensorArg tensor_arg;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_arg_out;
    tensor_arg_out.tensor.push_back(tensor_output);
    tensor_arg_out.arg_type = TensorArgType::TA_SINGLE;
    TeOpParas opParas;
    opParas.inputs.push_back(tensor_arg);
    opParas.outputs.push_back(tensor_arg_out);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str =compile_info;
    op_compile_info.key = "REDUCE__COUNTER__CONST_POST";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(runInfo.block_dim, 32);
    EXPECT_EQ(to_string(runInfo.tiling_data), "134 ");
}


// const_post_failed
TEST_F(BNReduceTiling, BNReduceTiling_const_post_failed) {
    std::string compile_info = R"({ "ori_axis": [0, 2, 3], "_ori_axis": [0, 2, 3], "_pattern": "bn_reduce", "push_status": 1,
        "_common_info": [8192, 32, 1, 8, 1, 1], "_reduce_shape_known": true, "_const_shape_post": true,
        "_block_dims" : {"133": 32}, "_atomic_flags" : {"134": true}, 
        "_vars": {
            "3304100": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "block_factor_1", "ub_factor_0"],
            "3204100": ["_dim_0", "_dim_1", "_dim_2", "_dim_3", "block_factor_1", "ub_factor_0"]
         }})";


    using namespace optiling;
    std::string op_name = "BNTrainingReduce";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::vector<int64_t> input{8, 8, 56, 56, 16};
    std::vector<int64_t> output{1, 8, 1, 1, 16};
    std::string in_dtype = "float32";

    TeOpTensor tensor_input;
    tensor_input.shape = input;
    tensor_input.dtype = in_dtype;
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = in_dtype;
    TeOpTensorArg tensor_arg;
    tensor_arg.tensor.push_back(tensor_input);
    tensor_arg.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_arg_out;
    tensor_arg_out.tensor.push_back(tensor_output);
    tensor_arg_out.arg_type = TensorArgType::TA_SINGLE;
    TeOpParas opParas;
    opParas.inputs.push_back(tensor_arg);
    opParas.outputs.push_back(tensor_arg_out);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str =compile_info;
    op_compile_info.key = "REDUCE__COUNTER__FAILED";
    OpRunInfo runInfo;
    ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}


