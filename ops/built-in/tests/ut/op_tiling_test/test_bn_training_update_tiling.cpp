#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class BNTrainingUpdateTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "BNTrainingUpdateTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "BNTrainingUpdateTiling TearDown" << std::endl;
    }
};

static string to_string(const std::stringstream &tiling_data)
{
    auto data = tiling_data.str();
    string result;
    int32_t tmp = 0;
    for (size_t i = 0; i < data.length(); i += sizeof(int32_t))
    {
        memcpy(&tmp, data.c_str() + i, sizeof(tmp));
        result += std::to_string(tmp);
        result += " ";
    }

    return result;
}

TEST_F(BNTrainingUpdateTiling, BNTrainingUpdate_tiling_test_1)
{
    using namespace optiling;
    std::string op_name = "BNTrainingUpdate";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
    std::string compileInfo = R"({
                        "bn_update_num_rec_dtype": "float32", 
                        "bn_update_batch_var_scaler_dtype": "float32", 
                        "_pattern": "BNTrainingUpdate", 
                        "max_ub_count": 10920, 
                        "block_dim": 32, 
                        "_flag_info": [false, false, null, true, false],
                        "_base_info": {"000": [262144, 4, 3, 32]},
                        "elewise_vars": {
                        "2": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "dim0_4", "num_rec", "batch_var_scaler"], 
                        "3": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "dim0_4", "num_rec", "batch_var_scaler"], 
                        "4": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "dim0_4", "num_rec", "batch_var_scaler"], 
                        "8": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "dim0_4", "num_rec", "batch_var_scaler"], 
                        "14": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "dim0_4", "num_rec", "batch_var_scaler"]}, 
                        "_vars": {
                        "2": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "dim0_4", "num_rec", "batch_var_scaler", "block_factor", "ub_factor"], 
                        "3": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "dim0_4", "num_rec", "batch_var_scaler", "block_factor", "ub_factor"], 
                        "4": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "dim0_4", "num_rec", "batch_var_scaler", "block_factor", "ub_factor"], 
                        "8": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "dim0_4", "num_rec", "batch_var_scaler", "block_factor", "ub_factor"], 
                        "9": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "dim0_4", "num_rec", "batch_var_scaler", "block_factor", "ub_factor"], 
                        "14": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "dim0_4", "num_rec", "batch_var_scaler", "block_factor", "ub_factor"]}, 
                        "_normal_vars": {"2": [], "3": [], "8": [], "9": [], "14": []}, 
                        "_attr_vars": {"2": [], "3": [], "8": [], "9": [], "14": []}, 
                        "_custom_vars": {
                        "2": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "dim0_4", "num_rec", "batch_var_scaler", "block_factor", "ub_factor"], 
                        "3": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "dim0_4", "num_rec", "batch_var_scaler", "block_factor", "ub_factor"], 
                        "4": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "dim0_4", "num_rec", "batch_var_scaler", "block_factor", "ub_factor"], 
                        "8": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "dim0_4", "num_rec", "batch_var_scaler", "block_factor", "ub_factor"], 
                        "9": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "dim0_4", "num_rec", "batch_var_scaler", "block_factor", "ub_factor"], 
                        "14": ["dim0_0", "dim0_1", "dim0_2", "dim0_3", "dim0_4", "num_rec", "batch_var_scaler", "block_factor", "ub_factor"]}})";

    std::vector<std::vector<int64_t>> inputs{
        {128, 6, 73, 73, 16},
        {1, 6, 1, 1, 16},
        {1, 6, 1, 1, 16},
        {1, 6, 1, 1, 16},
        {1, 6, 1, 1, 16},
        {1, 6, 1, 1, 16},
        {1, 6, 1, 1, 16}};

    std::vector<std::vector<int64_t>> outputs{
        {128, 6, 73, 73, 16},
        {1, 6, 1, 1, 16},
        {1, 6, 1, 1, 16},
        {1, 6, 1, 1, 16},
        {1, 6, 1, 1, 16}};

    std::vector<std::string> input_types{"float32", "float32", "float32", "float32", "float32", "float32", "float32"};
    std::vector<std::string> output_types{"float32", "float32", "float32", "float32", "float32"};
    std::string data_format = "NC1HWC0";

    TeOpParas opParas;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        TeOpTensor tensor_input;
        TeOpTensorArg tensor_arg;
        tensor_input.shape = inputs[i];
        tensor_input.dtype = input_types[i];
        tensor_input.format = data_format;
        tensor_arg.tensor.push_back(tensor_input);
        tensor_arg.arg_type = TensorArgType::TA_SINGLE;
        opParas.inputs.push_back(tensor_arg);
    }
    for (size_t i = 0; i < outputs.size(); i++)
    {
        TeOpTensor tensor_output;
        TeOpTensorArg tensor_arg;
        tensor_output.shape = outputs[i];
        tensor_output.dtype = output_types[i];
        tensor_output.format = data_format;
        tensor_arg.tensor.push_back(tensor_output);
        tensor_arg.arg_type = TensorArgType::TA_SINGLE;
        opParas.outputs.push_back(tensor_arg);
    }
    opParas.op_type = op_name;

    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "BNTrainingUpdate_tiling_test_1";

    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}