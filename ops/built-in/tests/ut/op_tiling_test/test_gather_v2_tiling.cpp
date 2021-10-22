#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "selection_ops.h"
#include "array_ops.h"
#include "test_common.h"

using namespace std;
using namespace ge;

class GatherV2Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GatherV2Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GatherV2Tiling TearDown" << std::endl;
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

static void Compute(vector<int64_t> inputA, vector<int64_t> inputB, vector<int64_t> inputC, vector<int32_t> axis,
                    vector<int64_t> output, ge::DataType dtypeA, ge::DataType dtypeB, ge::DataType dtypeC,
                    ge::DataType dtypeOutput, string infoKey, string compileInfo, string expectTilingData) {
  std::string op_name = "GatherV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("GatherV2");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  TensorDesc tensor_inputA;
  tensor_inputA.SetShape(ge::Shape(inputA));
  tensor_inputA.SetDataType(dtypeA);
  TensorDesc tensor_inputB;
  tensor_inputB.SetShape(ge::Shape(inputB));
  tensor_inputB.SetOriginShape(ge::Shape(inputB));
  tensor_inputB.SetDataType(dtypeB);
  TensorDesc tensor_inputC;
  tensor_inputC.SetShape(ge::Shape(inputC));
  tensor_inputC.SetDataType(dtypeC);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(dtypeOutput);

  auto opParas = op::GatherV2("GatherV2");
  TENSOR_INPUT(opParas, tensor_inputA, x);
  TENSOR_INPUT(opParas, tensor_inputB, indices);
  TENSOR_INPUT_CONST(opParas, tensor_inputC, axis, (const uint8_t*)axis.data(), axis.size() * 4);
  TENSOR_OUTPUT(opParas, tensor_output, y);

  optiling::utils::OpCompileInfo op_compile_info(infoKey.c_str(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), expectTilingData);
  int64_t profiling_test_num = 100;
  for (int64_t i = 0; i < profiling_test_num; i++) {
    iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  }
}

TEST_F(GatherV2Tiling, gather_v2_tiling_0) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":2}}";
  vector<int64_t> inputA{
      87552,
  };
  vector<int64_t> inputB{174, 1};
  vector<int64_t> inputC{1};
  vector<int32_t> axis{0};
  vector<int64_t> output{174, 1};
  ge::DataType dtypeA = ge::DT_FLOAT16;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "13 1 87552 1 174 0 8 0 21 6 0 32512 21 65024 32512 0 65024 21 0 87552 0 0 0 0 1 1 0 1 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_1) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":2}}";
  ;
  std::vector<int64_t> inputA{81, 6, 3};
  std::vector<int64_t> inputB{
      6,
  };
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{81, 6, 3};
  ge::DataType dtypeA = ge::DT_FLOAT16;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "8 81 6 3 6 0 32 0 6 0 0 19712 6 13136 6576 1 13136 6 0 1458 0 0 2 17 1 1 0 1 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_2) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":2}}";
  std::vector<int64_t> inputA{81, 6, 32};
  std::vector<int64_t> inputB{
      6,
  };
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{81, 6, 32};
  ge::DataType dtypeA = ge::DT_FLOAT16;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "10 81 6 32 6 0 32 0 6 0 0 19712 6 1232 0 16 1232 6 0 15552 0 0 2 17 1 1 0 1 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_3) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":2}}";
  std::vector<int64_t> inputA{81, 6, 32};
  std::vector<int64_t> inputB{
      6,
  };
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{0};
  std::vector<int64_t> output{6, 6, 32};
  ge::DataType dtypeA = ge::DT_FLOAT16;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "6 1 81 192 6 0 6 0 1 0 0 19712 1 205 32 96 205 1 0 15552 0 0 0 0 1 1 0 1 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_4) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":2}}";
  std::vector<int64_t> inputA{16, 8, 16, 32};
  std::vector<int64_t> inputB{
      32,
  };
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{0};
  std::vector<int64_t> output{32, 8, 16, 32};
  ge::DataType dtypeA = ge::DT_FLOAT16;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "3 1 16 4096 32 0 32 0 1 0 0 32512 1 15 7 2167 15 1 0 65536 0 0 0 0 1 1 0 1 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_5) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":2}}";
  std::vector<int64_t> inputA{16, 8, 16, 32};
  std::vector<int64_t> inputB{
      320,
  };
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{0};
  std::vector<int64_t> output{320, 8, 16, 32};
  ge::DataType dtypeA = ge::DT_FLOAT16;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "7 1 16 4096 320 0 32 0 10 0 0 32512 10 15 7 2167 15 10 0 65536 0 0 0 0 1 1 0 1 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_6) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":2}}";
  std::vector<int64_t> inputA{180, 4};
  std::vector<int64_t> inputB{
      4,
  };
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{0};
  std::vector<int64_t> output{4, 4};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "4 1 180 4 4 0 1 0 4 0 0 19712 4 9856 0 2 9856 4 0 720 0 0 0 0 1 1 0 1 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_7) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":2}}";
  std::vector<int64_t> inputA{180, 400000};
  std::vector<int64_t> inputB{
      4,
  };
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{0};
  std::vector<int64_t> output{4, 400000};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "5 1 180 400000 4 0 4 0 1 0 0 32512 1 0 0 0 0 0 0 72000000 6 9856 0 0 1 1 0 1 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_10) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":2}}";
  std::vector<int64_t> inputA{64, 8, 16, 32};
  std::vector<int64_t> inputB{
      32,
  };
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{64, 32, 16, 32};
  ge::DataType dtypeA = ge::DT_FLOAT16;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "11 64 8 512 32 0 32 0 32 0 0 32512 32 127 0 256 127 32 0 262144 0 0 2 0 1 1 0 1 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_11) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":2}}";
  std::vector<int64_t> inputA{64, 8, 3, 16};
  std::vector<int64_t> inputB{
      32,
  };
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{64, 32, 3, 16};
  ge::DataType dtypeA = ge::DT_FLOAT16;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "10 64 8 48 32 0 32 0 32 0 0 19712 32 821 8 24 821 32 0 24576 0 0 2 0 1 1 0 1 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_20) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{10, 16, 3};
  std::vector<int64_t> inputB{10, 40};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{10, 40, 3};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "20 1 16 3 400 0 10 0 40 0 0 40 0 6568 40 0 0 0 0 48 0 0 0 0 40 1 0 10 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_20_02) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{100, 16, 3};
  std::vector<int64_t> inputB{100, 800};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{100, 800, 3};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "20 1 16 3 80000 0 32 0 2400 3200 0 800 0 6568 800 0 0 0 0 144 0 0 0 0 800 3 4 100 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_21) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{100, 16, 3};
  std::vector<int64_t> inputB{100, 8000};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{100, 8000, 3};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "21 1 16 3 800000 0 32 0 24000 32000 0 8000 0 6568 1432 1 0 0 0 144 0 0 0 0 8000 3 4 100 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_22) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{10, 16, 3};
  std::vector<int64_t> inputB{10, 40000};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{10, 40000, 3};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "22 1 16 3 400000 0 10 0 40000 0 2 19712 576 6568 8 3 6568 576 0 48 0 0 0 0 40000 1 0 10 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_23) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{10, 16000, 3};
  std::vector<int64_t> inputB{10, 40};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{10, 40, 3};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "23 1 16000 3 400 0 10 0 40 0 0 40 0 10832 40 0 0 0 0 48000 0 0 0 0 40 1 0 10 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_23_02) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{100, 16000, 3};
  std::vector<int64_t> inputB{100, 800};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{100, 800, 3};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "23 1 16000 3 80000 0 32 0 2400 3200 0 800 0 10832 800 0 0 0 0 144000 0 0 0 0 800 3 4 100 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_24) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{100, 16000, 3};
  std::vector<int64_t> inputB{100, 18000};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{100, 18000, 3};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData =
      "24 1 16000 3 1800000 0 32 0 54000 72000 0 18000 0 10832 "
      "7168 1 0 0 0 144000 0 0 0 0 18000 3 4 100 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_25) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{10, 16000, 3};
  std::vector<int64_t> inputB{10, 40000};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{10, 40000, 3};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData =
      "25 1 16000 3 400000 0 10 0 40000 0 1 32512 7488 10832 16 3 10832 7488 0 48000 0 0 0 0 40000 1 0 10 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_26) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{10, 16, 33};
  std::vector<int64_t> inputB{10, 40};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{10, 40, 33};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "26 1 16 33 400 0 10 0 40 0 0 40 0 985 40 0 0 0 0 528 0 0 0 0 40 1 0 10 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_27) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{100, 16, 33};
  std::vector<int64_t> inputB{100, 18000};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{100, 18000, 33};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData =
      "27 1 16 33 1800000 0 32 0 54000 72000 0 18000 0 985 270 18 0 0 0 1584 0 0 0 0 18000 3 4 100 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_28) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{10, 16, 33};
  std::vector<int64_t> inputB{10, 40000};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{10, 40000, 33};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData =
      "28 1 16 33 400000 0 10 0 40000 0 1 32512 7488 985 7 33 985 593 7 528 0 0 0 0 40000 1 0 10 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_29) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{10, 16, 32};
  std::vector<int64_t> inputB{10, 4};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{10, 4, 32};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "29 1 16 32 40 0 10 0 4 0 0 4 0 616 4 0 0 0 0 512 0 0 0 0 4 1 0 10 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_30) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{1000, 16, 32};
  std::vector<int64_t> inputB{1000, 800};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{1000, 800, 32};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "30 1 16 32 800000 0 32 0 24800 6400 0 800 0 616 184 1 0 0 0 15872 0 0 0 0 800 31 8 1000 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_31) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{10, 6, 5, 32};
  std::vector<int64_t> inputB{10, 40000};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{2};
  std::vector<int64_t> output{10, 6, 40000, 32};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "31 6 5 32 400000 0 10 0 40000 0 2 19712 576 616 0 32 616 576 0 960 0 0 0 0 40000 1 0 10 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_32) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{10, 16000, 32};
  std::vector<int64_t> inputB{10, 4};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{10, 4, 32};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "32 1 16000 32 40 0 10 0 4 0 0 4 0 1016 4 0 0 0 0 512000 0 0 0 0 4 1 0 10 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_33) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{1000, 16000, 8};
  std::vector<int64_t> inputB{1000, 1800};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{1000, 1800, 8};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData =
      "33 1 16000 8 1800000 0 32 0 55800 14400 0 1800 0 4064 "
      "1800 0 0 0 0 3968000 0 0 0 0 1800 31 8 1000 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_34) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{10, 6, 500, 32};
  std::vector<int64_t> inputB{10, 40000};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{2};
  std::vector<int64_t> output{10, 6, 40000, 32};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData =
      "34 6 500 32 400000 0 10 0 40000 0 1 32512 7488 1016 0 "
      "32 1016 376 7 96000 0 0 0 0 40000 1 0 10 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_35) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{10, 16, 33000};
  std::vector<int64_t> inputB{10, 40};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{10, 40, 33000};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "35 1 16 33000 400 0 10 0 40 0 0 40 0 0 0 0 0 0 0 528000 1 488 0 0 40 1 0 10 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_36) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 2, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{4, 16, 33000};
  std::vector<int64_t> inputB{4, 19999};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{4, 19999, 33000};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "36 1 16 33000 79996 0 2 0 39998 0 0 19999 0 0 0 0 0 0 0 1056000 1 488 0 0 19999 2 0 4 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_37) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{2, 16, 33000};
  std::vector<int64_t> inputB{2, 40000};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{2, 40000, 33000};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "37 1 16 33000 80000 0 2 0 40000 0 1 32512 7488 0 0 0 0 0 0 528000 1 488 0 0 40000 1 0 2 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_38) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{2, 160, 2};
  std::vector<int64_t> inputB{2, 2};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{2};
  std::vector<int64_t> output{2, 160, 2};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "38 160 2 1 4 0 32 31 10 0 0 10 0 19712 10 0 0 0 0 640 0 0 0 0 2 5 0 2 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_39) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{2, 16000, 2};
  std::vector<int64_t> inputB{2, 2};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{2};
  std::vector<int64_t> output{2, 16000, 2};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "39 16000 2 1 4 0 32 31 1000 0 0 1000 0 32512 1000 0 0 0 0 64000 0 0 0 0 2 500 0 2 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_40) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{100, 16, 2};
  std::vector<int64_t> inputB{100, 2};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{100, 2, 2};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "40 1 16 2 200 0 32 31 6 0 0 6 0 9856 6 0 0 0 0 96 0 0 0 0 2 3 4 100 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_41) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}}";
  std::vector<int64_t> inputA{2, 16000, 2};
  std::vector<int64_t> inputB{2, 2};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{2, 2, 2};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  string expectTilingData = "41 1 16000 2 4 0 1 0 4 0 0 4 0 16256 4 0 0 0 0 64000 0 0 0 0 2 2 0 2 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData);
}
