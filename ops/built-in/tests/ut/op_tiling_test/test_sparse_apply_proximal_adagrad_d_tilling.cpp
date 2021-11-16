#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class SparseApplyProximalAdagradDTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "SparseApplyProximalAdagradDTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "SparseApplyProximalAdagradDTiling TearDown" << std::endl;
    }
};
static string to_string(const std::stringstream& tiling_data) {
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
TEST_F(SparseApplyProximalAdagradDTiling, sparseApplyProximalAdagradDTiling_0) {
  using namespace optiling;
  std::string op_name = "SparseApplyProximalAdagradD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 253952, \"ub_tensor_num\": 2}}";
  
  std::vector<int64_t> inputA{10, 20, 32};
  std::vector<int64_t> inputB{10, 20, 32};
  std::vector<int64_t> inputC{1,};
  std::vector<int64_t> inputD{1,};
  std::vector<int64_t> inputE{1,};
  std::vector<int64_t> inputF{10, 20, 32};
  std::vector<int64_t> inputG{10, 20};
  std::vector<int64_t> outputA{10, 20, 32};
  std::vector<int64_t> outputB{10, 20, 32};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "float32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "float32";
  TeOpTensor tensor_inputD;
  tensor_inputD.shape = inputD;
  tensor_inputD.dtype = "float32";
  TeOpTensor tensor_inputE;
  tensor_inputE.shape = inputE;
  tensor_inputE.dtype = "float32";
  TeOpTensor tensor_inputF;
  tensor_inputF.shape = inputF;
  tensor_inputF.dtype = "float32";
  TeOpTensor tensor_inputG;
  tensor_inputG.shape = inputG;
  tensor_inputG.dtype = "float32";
  TeOpTensor tensor_outputA;
  tensor_outputA.shape = outputA;
  tensor_outputA.dtype = "float32";
  TeOpTensor tensor_outputB;
  tensor_outputB.shape = outputB;
  tensor_outputB.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argD;
  tensor_argD.tensor.push_back(tensor_inputD);
  tensor_argD.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argE;
  tensor_argE.tensor.push_back(tensor_inputE);
  tensor_argE.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argF;
  tensor_argF.tensor.push_back(tensor_inputF);
  tensor_argF.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argG;
  tensor_argG.tensor.push_back(tensor_inputG);
  tensor_argG.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg1;
  tensor_arg1.tensor.push_back(tensor_outputA);
  tensor_arg1.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg2;
  tensor_arg2.tensor.push_back(tensor_outputB);
  tensor_arg2.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.inputs.push_back(tensor_argD);
  opParas.inputs.push_back(tensor_argE);
  opParas.inputs.push_back(tensor_argF);
  opParas.inputs.push_back(tensor_argG);
  opParas.outputs.push_back(tensor_arg1);
  opParas.outputs.push_back(tensor_arg2);

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "sapadt0";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 1 1 968 200 121 25 4 32 1 ");
}
TEST_F(SparseApplyProximalAdagradDTiling, sparseApplyProximalAdagradDTiling_1) {
  using namespace optiling;
  std::string op_name = "SparseApplyProximalAdagradD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = "{\"vars\": {\"ub_size\": 253952, \"ub_tensor_num\": 2}}";
  
  std::vector<int64_t> inputA{10, 20, 32};
  std::vector<int64_t> inputB{10, 20, 32};
  std::vector<int64_t> inputC{1,};
  std::vector<int64_t> inputD{1,};
  std::vector<int64_t> inputE{1,};
  std::vector<int64_t> inputF{10, 20, 32};
  std::vector<int64_t> inputG{10, 20};
  std::vector<int64_t> outputA{10, 20, 32};
  std::vector<int64_t> outputB{10, 20, 32};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "float32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "float32";
  TeOpTensor tensor_inputD;
  tensor_inputD.shape = inputD;
  tensor_inputD.dtype = "float32";
  TeOpTensor tensor_inputE;
  tensor_inputE.shape = inputE;
  tensor_inputE.dtype = "float32";
  TeOpTensor tensor_inputF;
  tensor_inputF.shape = inputF;
  tensor_inputF.dtype = "float32";
  TeOpTensor tensor_inputG;
  tensor_inputG.shape = inputG;
  tensor_inputG.dtype = "float32";
  TeOpTensor tensor_outputA;
  tensor_outputA.shape = outputA;
  tensor_outputA.dtype = "float32";
  TeOpTensor tensor_outputB;
  tensor_outputB.shape = outputB;
  tensor_outputB.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argD;
  tensor_argD.tensor.push_back(tensor_inputD);
  tensor_argD.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argE;
  tensor_argE.tensor.push_back(tensor_inputE);
  tensor_argE.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argF;
  tensor_argF.tensor.push_back(tensor_inputF);
  tensor_argF.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argG;
  tensor_argG.tensor.push_back(tensor_inputG);
  tensor_argG.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg1;
  tensor_arg1.tensor.push_back(tensor_outputA);
  tensor_arg1.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg2;
  tensor_arg2.tensor.push_back(tensor_outputB);
  tensor_arg2.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.inputs.push_back(tensor_argD);
  opParas.inputs.push_back(tensor_argE);
  opParas.inputs.push_back(tensor_argF);
  opParas.inputs.push_back(tensor_argG);
  opParas.outputs.push_back(tensor_arg1);
  opParas.outputs.push_back(tensor_arg2);

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "sapadt1";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}
TEST_F(SparseApplyProximalAdagradDTiling, sparseApplyProximalAdagradDTiling_2) {
  using namespace optiling;
  std::string op_name = "SparseApplyProximalAdagradD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_tensor_num\": 2}}";
  
  std::vector<int64_t> inputA{10, 20, 32};
  std::vector<int64_t> inputB{10, 20, 32};
  std::vector<int64_t> inputC{1,};
  std::vector<int64_t> inputD{1,};
  std::vector<int64_t> inputE{1,};
  std::vector<int64_t> inputF{10, 20, 32};
  std::vector<int64_t> inputG{10, 20};
  std::vector<int64_t> outputA{10, 20, 32};
  std::vector<int64_t> outputB{10, 20, 32};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "float32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "float32";
  TeOpTensor tensor_inputD;
  tensor_inputD.shape = inputD;
  tensor_inputD.dtype = "float32";
  TeOpTensor tensor_inputE;
  tensor_inputE.shape = inputE;
  tensor_inputE.dtype = "float32";
  TeOpTensor tensor_inputF;
  tensor_inputF.shape = inputF;
  tensor_inputF.dtype = "float32";
  TeOpTensor tensor_inputG;
  tensor_inputG.shape = inputG;
  tensor_inputG.dtype = "float32";
  TeOpTensor tensor_outputA;
  tensor_outputA.shape = outputA;
  tensor_outputA.dtype = "float32";
  TeOpTensor tensor_outputB;
  tensor_outputB.shape = outputB;
  tensor_outputB.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argD;
  tensor_argD.tensor.push_back(tensor_inputD);
  tensor_argD.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argE;
  tensor_argE.tensor.push_back(tensor_inputE);
  tensor_argE.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argF;
  tensor_argF.tensor.push_back(tensor_inputF);
  tensor_argF.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argG;
  tensor_argG.tensor.push_back(tensor_inputG);
  tensor_argG.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg1;
  tensor_arg1.tensor.push_back(tensor_outputA);
  tensor_arg1.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg2;
  tensor_arg2.tensor.push_back(tensor_outputB);
  tensor_arg2.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.inputs.push_back(tensor_argD);
  opParas.inputs.push_back(tensor_argE);
  opParas.inputs.push_back(tensor_argF);
  opParas.inputs.push_back(tensor_argG);
  opParas.outputs.push_back(tensor_arg1);
  opParas.outputs.push_back(tensor_arg2);

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "sapadt2";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}
TEST_F(SparseApplyProximalAdagradDTiling, sparseApplyProximalAdagradDTiling_3) {
  using namespace optiling;
  std::string op_name = "SparseApplyProximalAdagradD";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\": 253952}}";
  
  std::vector<int64_t> inputA{10, 20, 32};
  std::vector<int64_t> inputB{10, 20, 32};
  std::vector<int64_t> inputC{1,};
  std::vector<int64_t> inputD{1,};
  std::vector<int64_t> inputE{1,};
  std::vector<int64_t> inputF{10, 20, 32};
  std::vector<int64_t> inputG{10, 20};
  std::vector<int64_t> outputA{10, 20, 32};
  std::vector<int64_t> outputB{10, 20, 32};

  TeOpTensor tensor_inputA;
  tensor_inputA.shape = inputA;
  tensor_inputA.dtype = "float32";
  TeOpTensor tensor_inputB;
  tensor_inputB.shape = inputB;
  tensor_inputB.dtype = "float32";
  TeOpTensor tensor_inputC;
  tensor_inputC.shape = inputC;
  tensor_inputC.dtype = "float32";
  TeOpTensor tensor_inputD;
  tensor_inputD.shape = inputD;
  tensor_inputD.dtype = "float32";
  TeOpTensor tensor_inputE;
  tensor_inputE.shape = inputE;
  tensor_inputE.dtype = "float32";
  TeOpTensor tensor_inputF;
  tensor_inputF.shape = inputF;
  tensor_inputF.dtype = "float32";
  TeOpTensor tensor_inputG;
  tensor_inputG.shape = inputG;
  tensor_inputG.dtype = "float32";
  TeOpTensor tensor_outputA;
  tensor_outputA.shape = outputA;
  tensor_outputA.dtype = "float32";
  TeOpTensor tensor_outputB;
  tensor_outputB.shape = outputB;
  tensor_outputB.dtype = "float32";

  TeOpTensorArg tensor_argA;
  tensor_argA.tensor.push_back(tensor_inputA);
  tensor_argA.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argB;
  tensor_argB.tensor.push_back(tensor_inputB);
  tensor_argB.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argC;
  tensor_argC.tensor.push_back(tensor_inputC);
  tensor_argC.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argD;
  tensor_argD.tensor.push_back(tensor_inputD);
  tensor_argD.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argE;
  tensor_argE.tensor.push_back(tensor_inputE);
  tensor_argE.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argF;
  tensor_argF.tensor.push_back(tensor_inputF);
  tensor_argF.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_argG;
  tensor_argG.tensor.push_back(tensor_inputG);
  tensor_argG.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg1;
  tensor_arg1.tensor.push_back(tensor_outputA);
  tensor_arg1.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_arg2;
  tensor_arg2.tensor.push_back(tensor_outputB);
  tensor_arg2.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensor_argA);
  opParas.inputs.push_back(tensor_argB);
  opParas.inputs.push_back(tensor_argC);
  opParas.inputs.push_back(tensor_argD);
  opParas.inputs.push_back(tensor_argE);
  opParas.inputs.push_back(tensor_argF);
  opParas.inputs.push_back(tensor_argG);
  opParas.outputs.push_back(tensor_arg1);
  opParas.outputs.push_back(tensor_arg2);

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "sapadt3";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
}