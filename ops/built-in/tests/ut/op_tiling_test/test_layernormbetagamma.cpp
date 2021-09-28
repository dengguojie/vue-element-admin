#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"
#include <iostream>
using namespace std;

class LayerNormBetaGammaTiling : public testing::Test {
 protected:
    static void SetUpTestCase() {
        std::cout << "LayerNormBetaGammaTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "LayerNormBetaGammaTiling TearDown" << std::endl;
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

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_0) {
    using namespace optiling;
    std::string op_name = "LayerNormBetaGammaBackprop";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("LayerNormBetaGammaBackprop");
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());
    
    std::string compileInfo = "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"batch_cols_padding\":4835, \"k_num\":139}}";

    std::vector<int64_t> input_tensor_shape{2,3,512};

    TeOpTensor input_tensor;
    input_tensor.shape = input_tensor_shape;
    input_tensor.dtype = "float16";

    TeOpTensorArg tensor_argA;
    tensor_argA.tensor.push_back(input_tensor);
    tensor_argA.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_argA);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "aa";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_1) {
    using namespace optiling;
    std::string op_name = "LayerNormBetaGammaBackpropV2";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("LayerNormBetaGammaBackpropV2");
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    std::string compileInfo = "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, \"shape_gamma\":[512],\"dynamic_reduce\":true,\"dynamic_normal\":false}";

    std::vector<int64_t> input_tensor_shape{2,3,512};

    TeOpTensor input_tensor;
    input_tensor.shape = input_tensor_shape;
    input_tensor.dtype = "float16";

    TeOpTensorArg tensor_argA;
    tensor_argA.tensor.push_back(input_tensor);
    tensor_argA.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_argA);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "tiling1";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_2) {
    using namespace optiling;
    std::string op_name = "LayerNormBetaGammaBackpropV2";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("LayerNormBetaGammaBackpropV2");
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    std::string compileInfo = "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, \"shape_gamma\":[512],\"dynamic_reduce\":true,\"dynamic_normal\":false}";

    std::vector<int64_t> input_tensor_shape{32,10,512};

    TeOpTensor input_tensor;
    input_tensor.shape = input_tensor_shape;
    input_tensor.dtype = "float16";

    TeOpTensorArg tensor_argA;
    tensor_argA.tensor.push_back(input_tensor);
    tensor_argA.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_argA);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "aa2";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_3) {
    using namespace optiling;
    std::string op_name = "LayerNormBetaGammaBackpropV2";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("LayerNormBetaGammaBackpropV2");
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    std::string compileInfo = "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, \"shape_gamma\":[512],\"dynamic_reduce\":true,\"dynamic_normal\":false}";

    std::vector<int64_t> input_tensor_shape{32,51,512};

    TeOpTensor input_tensor;
    input_tensor.shape = input_tensor_shape;
    input_tensor.dtype = "float16";

    TeOpTensorArg tensor_argA;
    tensor_argA.tensor.push_back(input_tensor);
    tensor_argA.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_argA);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "aa3";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_4) {
    using namespace optiling;
    std::string op_name = "LayerNormBetaGammaBackpropV2";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("LayerNormBetaGammaBackpropV2");
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    std::string compileInfo = "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, \"shape_gamma\":[4096],\"dynamic_reduce\":true,\"dynamic_normal\":false}";

    std::vector<int64_t> input_tensor_shape{32,51,4096};

    TeOpTensor input_tensor;
    input_tensor.shape = input_tensor_shape;
    input_tensor.dtype = "float16";

    TeOpTensorArg tensor_argA;
    tensor_argA.tensor.push_back(input_tensor);
    tensor_argA.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_argA);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "aa4";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_5) {
    using namespace optiling;
    std::string op_name = "LayerNormBetaGammaBackpropV2";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("LayerNormBetaGammaBackpropV2");
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    std::string compileInfo = "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, \"shape_gamma\":[512],\"dynamic_reduce\":false,\"dynamic_normal\":true}";

    std::vector<int64_t> input_tensor_shape{2,3,512};

    TeOpTensor input_tensor;
    input_tensor.shape = input_tensor_shape;
    input_tensor.dtype = "float16";

    TeOpTensorArg tensor_argA;
    tensor_argA.tensor.push_back(input_tensor);
    tensor_argA.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_argA);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "aa5";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}
 
TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_6) {
    using namespace optiling;
    std::string op_name = "LayerNormBetaGammaBackpropV2";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("LayerNormBetaGammaBackpropV2");
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    std::string compileInfo = "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, \"shape_gamma\":[512],\"dynamic_reduce\":false,\"dynamic_normal\":true}";

    std::vector<int64_t> input_tensor_shape{32,10,512};

    TeOpTensor input_tensor;
    input_tensor.shape = input_tensor_shape;
    input_tensor.dtype = "float16";

    TeOpTensorArg tensor_argA;
    tensor_argA.tensor.push_back(input_tensor);
    tensor_argA.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_argA);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "aa6";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_7) {
    std::cout<<"tiling7 start."<<std::endl;
    using namespace optiling;
    std::string op_name = "LayerNormBetaGammaBackpropV2";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("LayerNormBetaGammaBackpropV2");
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    std::string compileInfo = "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, \"shape_gamma\":[512],\"dynamic_reduce\":false,\"dynamic_normal\":true}";

    std::vector<int64_t> input_tensor_shape{32,100,512};

    TeOpTensor input_tensor;
    input_tensor.shape = input_tensor_shape;
    input_tensor.dtype = "float16";

    TeOpTensorArg tensor_argA;
    tensor_argA.tensor.push_back(input_tensor);
    tensor_argA.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_argA);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "aa7";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_8) {
    std::cout<<"tiling8 start."<<std::endl;
    using namespace optiling;
    std::string op_name = "LayerNormBetaGammaBackpropV2";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("LayerNormBetaGammaBackpropV2");
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    std::string compileInfo = "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, \"shape_gamma\":[4096],\"dynamic_reduce\":false,\"dynamic_normal\":true}";

    std::vector<int64_t> input_tensor_shape{32,51,4096};

    TeOpTensor input_tensor;
    input_tensor.shape = input_tensor_shape;
    input_tensor.dtype = "float16";

    TeOpTensorArg tensor_argA;
    tensor_argA.tensor.push_back(input_tensor);
    tensor_argA.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_argA);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "aa8";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_9) {
    std::cout<<"tiling9 start."<<std::endl;
    using namespace optiling;
    std::string op_name = "LayerNormBetaGammaBackpropV2";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("LayerNormBetaGammaBackpropV2");
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    std::string compileInfo = "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, \"shape_gamma\":[512],\"dynamic_reduce\":true,\"dynamic_normal\":true}";

    std::vector<int64_t> input_tensor_shape{2,3,512};

    TeOpTensor input_tensor;
    input_tensor.shape = input_tensor_shape;
    input_tensor.dtype = "float16";

    TeOpTensorArg tensor_argA;
    tensor_argA.tensor.push_back(input_tensor);
    tensor_argA.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_argA);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "aa9";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_10) {
    std::cout<<"tiling10 start."<<std::endl;
    using namespace optiling;
    std::string op_name = "LayerNormBetaGammaBackpropV2";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("LayerNormBetaGammaBackpropV2");
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    std::string compileInfo = "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, \"shape_gamma\":[512],\"dynamic_reduce\":true,\"dynamic_normal\":true}";

    std::vector<int64_t> input_tensor_shape{32,10,512};

    TeOpTensor input_tensor;
    input_tensor.shape = input_tensor_shape;
    input_tensor.dtype = "float16";

    TeOpTensorArg tensor_argA;
    tensor_argA.tensor.push_back(input_tensor);
    tensor_argA.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_argA);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "aa10";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_11) {
    std::cout<<"tiling11 start."<<std::endl;
    using namespace optiling;
    std::string op_name = "LayerNormBetaGammaBackpropV2";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("LayerNormBetaGammaBackpropV2");
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    std::string compileInfo = "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, \"shape_gamma\":[512],\"dynamic_reduce\":true,\"dynamic_normal\":true}";

    std::vector<int64_t> input_tensor_shape{32,100,512};

    TeOpTensor input_tensor;
    input_tensor.shape = input_tensor_shape;
    input_tensor.dtype = "float16";

    TeOpTensorArg tensor_argA;
    tensor_argA.tensor.push_back(input_tensor);
    tensor_argA.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_argA);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "aa11";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_12) {
    std::cout<<"tiling12 start."<<std::endl;
    using namespace optiling;
    std::string op_name = "LayerNormBetaGammaBackpropV2";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("LayerNormBetaGammaBackpropV2");
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    std::string compileInfo = "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, \"shape_gamma\":[4096],\"dynamic_reduce\":true,\"dynamic_normal\":true}";

    std::vector<int64_t> input_tensor_shape{32,51,4096};

    TeOpTensor input_tensor;
    input_tensor.shape = input_tensor_shape;
    input_tensor.dtype = "float16";

    TeOpTensorArg tensor_argA;
    tensor_argA.tensor.push_back(input_tensor);
    tensor_argA.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_argA);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "aa12";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_13) {
    std::cout<<"tiling13 start."<<std::endl;
    using namespace optiling;
    std::string op_name = "LayerNormBetaGammaBackpropV2";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("LayerNormBetaGammaBackpropV2");
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    std::string compileInfo = "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, \"shape_gamma\":[2,3,512],\"dynamic_reduce\":true,\"dynamic_normal\":false}";

    std::vector<int64_t> input_tensor_shape{2,3,512};

    TeOpTensor input_tensor;
    input_tensor.shape = input_tensor_shape;
    input_tensor.dtype = "float16";

    TeOpTensorArg tensor_argA;
    tensor_argA.tensor.push_back(input_tensor);
    tensor_argA.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_argA);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "aa13";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_14) {
    std::cout<<"tiling14 start."<<std::endl;
    using namespace optiling;
    std::string op_name = "LayerNormBetaGammaBackpropV2";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("LayerNormBetaGammaBackpropV2");
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    std::string compileInfo = "{\"core_num\": 32, \"max_reduce_factor\":50, \"max_last_factor\":2048, \"shape_gamma\":[32,100,512],\"dynamic_reduce\":true,\"dynamic_normal\":false}";

    std::vector<int64_t> input_tensor_shape{32,100,512};

    TeOpTensor input_tensor;
    input_tensor.shape = input_tensor_shape;
    input_tensor.dtype = "float16";

    TeOpTensorArg tensor_argA;
    tensor_argA.tensor.push_back(input_tensor);
    tensor_argA.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_argA);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "aa14";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(LayerNormBetaGammaTiling, layernormbetagamma_tiling_15) {
    std::cout<<"tiling15 start."<<std::endl;
    using namespace optiling;
    std::string op_name = "LayerNormBetaGammaBackpropV2";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("LayerNormBetaGammaBackpropV2");
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    std::string compileInfo = "{\"core_num\": 0, \"max_reduce_factor\":50, \"max_last_factor\":2048, \"shape_gamma\":[32,100,512],\"dynamic_reduce\":true,\"dynamic_normal\":false}";

    std::vector<int64_t> input_tensor_shape{32,100,512};

    TeOpTensor input_tensor;
    input_tensor.shape = input_tensor_shape;
    input_tensor.dtype = "float16";

    TeOpTensorArg tensor_argA;
    tensor_argA.tensor.push_back(input_tensor);
    tensor_argA.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_argA);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "aa15";
    OpRunInfo runInfo;
    ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

