#include <iostream>
#include <vector>
#include <fstream>

#include <gtest/gtest.h>

#define private public

#include "register/op_tiling_registry.h"
#include "selection_ops.h"
#include "array_ops.h"
#include "test_common.h"
#include "common/utils/ut_op_util.h"
#include "common_unittest.h"
#include "gatherv2.h"


#include "graph/utils/op_desc_utils.h"
#include "graph/graph.h"
#include "op_tiling/tiling_handler.h"

#include "common_autotiling_util.h"

#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"

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

static void Compute(vector <int64_t> inputA, vector <int64_t> inputB, vector <int64_t> inputC, vector <int32_t> axis,
                    vector <int64_t> output, ge::DataType dtypeA, ge::DataType dtypeB, ge::DataType dtypeC,
                    ge::DataType dtypeOutput, string infoKey, string compileInfo, string expectTilingData, int64_t value) {
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
  TENSOR_INPUT_CONST(opParas, tensor_inputC, axis, (const uint8_t *) axis.data(), axis.size() * 4);
  TENSOR_OUTPUT(opParas, tensor_output, y);
  opParas.SetAttr("batch_dims", value);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), expectTilingData);
  optiling::GatherV2CompileInfo info;
  int64_t tiling_len = sizeof(optiling::GatherV2TilingParams);
  TILING_PARSE_JSON_TO_COMPILEINFO("GatherV2",compileInfo,info);
  vector<bool> input_const={false, false, true};
  vector<string> attrs={"batch_dims"};
  ATTACH_OPERATOR_TO_HOLDER_CONST(holder,opParas, input_const, attrs, tiling_len, info);
  HOLDER_DO_TILING(holder,"GatherV2",ge::GRAPH_SUCCESS);
  TILING_DATA_VERIFY_BYTYPE(holder, int64_t, expectTilingData);

}

TEST_F(GatherV2Tiling, gather_v2_tiling_0) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":2, \"batch_dims\":0}, \"is_tik\": true,\"is_gather_v2\": true}";
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
  int64_t batch_dims = 0;
  string expectTilingData = "13 1 87552 1 174 0 8 0 21 6 0 32512 21 65024 32512 0 65024 21 0 87552 0 0 0 0 1 1 0 1 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_1) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":2, \"batch_dims\":0}, \"is_tik\": true,\"is_gather_v2\": true}";
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
  int64_t batch_dims = 0;
  string expectTilingData = "8 81 6 3 6 0 32 0 6 0 0 19712 6 13136 6576 1 13136 6 0 1458 0 0 2 17 1 1 0 1 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_2) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":2, \"batch_dims\":0}, \"is_tik\": true,\"is_gather_v2\": true}";
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
  int64_t batch_dims = 0;
  string expectTilingData = "10 81 6 32 6 0 32 0 6 0 0 19712 6 1232 0 16 1232 6 0 15552 0 0 2 17 1 1 0 1 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_3) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":2, \"batch_dims\":0}, \"is_tik\": true,\"is_gather_v2\": true}";
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
  int64_t batch_dims = 0;
  string expectTilingData = "6 1 81 192 6 0 6 0 1 0 0 19712 1 205 32 96 205 1 0 15552 0 0 0 0 1 1 0 1 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_4) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":2, \"batch_dims\":0}, \"is_tik\": true,\"is_gather_v2\": true}";
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
  int64_t batch_dims = 0;
  string expectTilingData = "3 1 16 4096 32 0 32 0 1 0 0 32512 1 15 7 2167 15 1 0 65536 0 0 0 0 1 1 0 1 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_5) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":2, \"batch_dims\":0}, \"is_tik\": true,\"is_gather_v2\": true}";
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
  int64_t batch_dims = 0;
  string expectTilingData = "7 1 16 4096 320 0 32 0 10 0 0 32512 10 15 7 2167 15 10 0 65536 0 0 0 0 1 1 0 1 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_6) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":2, \"batch_dims\":0}, \"is_tik\": true,\"is_gather_v2\": true}";
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
  int64_t batch_dims = 0;
  string expectTilingData = "4 1 180 4 4 0 1 0 4 0 0 19712 4 9856 0 2 9856 4 0 720 0 0 0 0 1 1 0 1 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_7) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":2, \"batch_dims\":0}, \"is_tik\": true,\"is_gather_v2\": true}";
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
  int64_t batch_dims = 0;
  string expectTilingData = "5 1 180 400000 4 0 4 0 1 0 0 32512 1 0 0 0 0 0 0 72000000 6 9856 0 0 1 1 0 1 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_10) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":2, \"batch_dims\":0}, \"is_tik\": true,\"is_gather_v2\": true}";
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
  int64_t batch_dims = 0;
  string expectTilingData = "11 64 8 512 32 0 32 0 32 0 0 32512 32 127 0 256 127 32 0 262144 0 0 2 0 1 1 0 1 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_11) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":2, \"batch_dims\":0}, \"is_tik\": true,\"is_gather_v2\": true}";
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
  int64_t batch_dims = 0;
  string expectTilingData = "10 64 8 48 32 0 32 0 32 0 0 19712 32 821 8 24 821 32 0 24576 0 0 2 0 1 1 0 1 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_14) {
  int32_t testIdx = 0;
  {
    std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":8, "
      "\"params_dsize\":4, \"batch_dims\":0, \"impl_mode\":\"high_performance\"}, \"is_tik\": true, "
      "\"is_gather_v2\": true}";
    std::vector<int64_t> inputA{822891, 16};
    std::vector<int64_t> inputB{5120, 820};
    std::vector<int64_t> inputC{1};
    std::vector<int32_t> axis{0};
    std::vector<int64_t> output{5120, 820, 16};
    ge::DataType dtypeA = ge::DT_FLOAT;
    ge::DataType dtypeB = ge::DT_INT64;
    ge::DataType dtypeC = ge::DT_INT32;
    ge::DataType dtypeOutput = dtypeA;
    int64_t batch_dims = 0;
    string expectTilingData = "14 1 822891 16 4198400 0 32 32 131200 131200 0 0 0 0 0 0 0 0 0 13166256 0 0 0 0 1 1 0 1 ";
    Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput,
            this->test_info_->name() + std::to_string(++testIdx),
            compileInfo, expectTilingData, batch_dims);
  }
  {
    std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":8, "
      "\"params_dsize\":4, \"batch_dims\":0, \"impl_mode\":\"high_performance\"}, \"is_tik\": true, "
      "\"is_gather_v2\": true}";
    std::vector<int64_t> inputA{822891, 11};
    std::vector<int64_t> inputB{5120, 820};
    std::vector<int64_t> inputC{1};
    std::vector<int32_t> axis{0};
    std::vector<int64_t> output{5120, 820, 11};
    ge::DataType dtypeA = ge::DT_FLOAT;
    ge::DataType dtypeB = ge::DT_INT64;
    ge::DataType dtypeC = ge::DT_INT32;
    ge::DataType dtypeOutput = dtypeA;
    int64_t batch_dims = 0;
    string expectTilingData = "14 1 822891 11 4198400 0 32 32 131200 131200 0 0 0 0 0 0 0 0 0 9051801 0 0 0 0 1 1 0 1 ";
    Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput,
            this->test_info_->name() + std::to_string(++testIdx),
            compileInfo, expectTilingData, batch_dims);
  }
  {
    std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, \"indices_dsize\":8, "
      "\"params_dsize\":4, \"batch_dims\":0, \"impl_mode\":\"high_performance\"}, \"is_tik\": true, "
      "\"is_gather_v2\": true}";
    std::vector<int64_t> inputA{822891, 5};
    std::vector<int64_t> inputB{5120, 820};
    std::vector<int64_t> inputC{1};
    std::vector<int32_t> axis{0};
    std::vector<int64_t> output{5120, 820, 5};
    ge::DataType dtypeA = ge::DT_FLOAT;
    ge::DataType dtypeB = ge::DT_INT64;
    ge::DataType dtypeC = ge::DT_INT32;
    ge::DataType dtypeOutput = dtypeA;
    int64_t batch_dims = 0;
    string expectTilingData = "14 1 822891 5 4198400 0 32 32 131200 131200 0 0 0 0 0 0 0 0 0 4114455 0 0 0 0 1 1 0 1 ";
    Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput,
            this->test_info_->name() + std::to_string(++testIdx),
            compileInfo, expectTilingData, batch_dims);
  }
}

TEST_F(GatherV2Tiling, gather_v2_tiling_20) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{10, 16, 3};
  std::vector<int64_t> inputB{10, 40};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{10, 40, 3};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData = "20 1 16 3 400 0 10 0 40 0 0 40 0 6568 40 0 0 0 0 48 0 0 0 0 40 1 0 10 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_20_02) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{100, 16, 3};
  std::vector<int64_t> inputB{100, 800};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{100, 800, 3};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData = "20 1 16 3 80000 0 32 0 2400 3200 0 800 0 6568 800 0 0 0 0 144 0 0 0 0 800 3 4 100 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_21) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{100, 16, 3};
  std::vector<int64_t> inputB{100, 8000};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{100, 8000, 3};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData = "21 1 16 3 800000 0 32 0 24000 32000 0 8000 0 6568 1432 1 0 0 0 144 0 0 0 0 8000 3 4 100 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_22) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{10, 16, 3};
  std::vector<int64_t> inputB{10, 40000};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{10, 40000, 3};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData = "22 1 16 3 400000 0 10 0 40000 0 2 19712 576 6568 8 3 6568 576 0 48 0 0 0 0 40000 1 0 10 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_23) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{10, 16000, 3};
  std::vector<int64_t> inputB{10, 40};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{10, 40, 3};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData = "23 1 16000 3 400 0 10 0 40 0 0 40 0 10832 40 0 0 0 0 48000 0 0 0 0 40 1 0 10 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_23_02) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{100, 16000, 3};
  std::vector<int64_t> inputB{100, 800};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{100, 800, 3};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData = "23 1 16000 3 80000 0 32 0 2400 3200 0 800 0 10832 800 0 0 0 0 144000 0 0 0 0 800 3 4 100 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_24) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{100, 16000, 3};
  std::vector<int64_t> inputB{100, 18000};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{100, 18000, 3};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData =
      "24 1 16000 3 1800000 0 32 0 54000 72000 0 18000 0 10832 "
      "7168 1 0 0 0 144000 0 0 0 0 18000 3 4 100 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_25) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{10, 16000, 3};
  std::vector<int64_t> inputB{10, 40000};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{10, 40000, 3};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData =
      "25 1 16000 3 400000 0 10 0 40000 0 1 32512 7488 10832 16 3 10832 7488 0 48000 0 0 0 0 40000 1 0 10 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_26) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{10, 16, 33};
  std::vector<int64_t> inputB{10, 40};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{10, 40, 33};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData = "26 1 16 33 400 0 10 0 40 0 0 40 0 985 40 0 0 0 0 528 0 0 0 0 40 1 0 10 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_27) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{100, 16, 33};
  std::vector<int64_t> inputB{100, 18000};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{100, 18000, 33};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData =
      "27 1 16 33 1800000 0 32 0 54000 72000 0 18000 0 985 270 18 0 0 0 1584 0 0 0 0 18000 3 4 100 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_28) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{10, 16, 33};
  std::vector<int64_t> inputB{10, 40000};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{10, 40000, 33};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData =
      "28 1 16 33 400000 0 10 0 40000 0 1 32512 7488 985 7 33 985 593 7 528 0 0 0 0 40000 1 0 10 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_29) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{10, 16, 32};
  std::vector<int64_t> inputB{10, 4};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{10, 4, 32};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData = "29 1 16 32 40 0 10 0 4 0 0 4 0 616 4 0 0 0 0 512 0 0 0 0 4 1 0 10 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_30) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{1000, 16, 32};
  std::vector<int64_t> inputB{1000, 800};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{1000, 800, 32};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData = "30 1 16 32 800000 0 32 0 24800 6400 0 800 0 616 184 1 0 0 0 15872 0 0 0 0 800 31 8 1000 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_31) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{10, 6, 5, 32};
  std::vector<int64_t> inputB{10, 40000};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{2};
  std::vector<int64_t> output{10, 6, 40000, 32};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData = "31 6 5 32 400000 0 10 0 40000 0 2 19712 576 616 0 32 616 576 0 960 0 0 0 0 40000 1 0 10 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_32) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{10, 16000, 32};
  std::vector<int64_t> inputB{10, 4};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{10, 4, 32};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData = "32 1 16000 32 40 0 10 0 4 0 0 4 0 1016 4 0 0 0 0 512000 0 0 0 0 4 1 0 10 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_33) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{1000, 16000, 8};
  std::vector<int64_t> inputB{1000, 1800};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{1000, 1800, 8};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData =
      "33 1 16000 8 1800000 0 32 0 55800 14400 0 1800 0 4064 "
      "1800 0 0 0 0 3968000 0 0 0 0 1800 31 8 1000 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_34) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{10, 6, 500, 32};
  std::vector<int64_t> inputB{10, 40000};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{2};
  std::vector<int64_t> output{10, 6, 40000, 32};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData =
      "34 6 500 32 400000 0 10 0 40000 0 1 32512 7488 1016 0 "
      "32 1016 376 7 96000 0 0 0 0 40000 1 0 10 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_35) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{10, 16, 33000};
  std::vector<int64_t> inputB{10, 40};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{10, 40, 33000};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData = "35 1 16 33000 400 0 10 0 40 0 0 40 0 0 0 0 0 0 0 528000 1 488 0 0 40 1 0 10 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_36) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 2, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{4, 16, 33000};
  std::vector<int64_t> inputB{4, 19999};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{4, 19999, 33000};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData = "36 1 16 33000 79996 0 2 0 39998 0 0 19999 0 0 0 0 0 0 0 1056000 1 488 0 0 19999 2 0 4 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_37) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{2, 16, 33000};
  std::vector<int64_t> inputB{2, 40000};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{2, 40000, 33000};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData = "37 1 16 33000 80000 0 2 0 40000 0 1 32512 7488 0 0 0 0 0 0 528000 1 488 0 0 40000 1 0 2 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_38) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{2, 160, 2};
  std::vector<int64_t> inputB{2, 2};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{2};
  std::vector<int64_t> output{2, 160, 2};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData = "38 160 2 1 4 0 32 31 10 0 0 10 0 19712 10 0 0 0 0 640 0 0 0 0 2 5 0 2 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_39) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{2, 16000, 2};
  std::vector<int64_t> inputB{2, 2};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{2};
  std::vector<int64_t> output{2, 16000, 2};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData = "39 16000 2 1 4 0 32 31 1000 0 0 1000 0 32512 1000 0 0 0 0 64000 0 0 0 0 2 500 0 2 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_40) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{100, 16, 2};
  std::vector<int64_t> inputB{100, 2};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{100, 2, 2};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData = "40 1 16 2 200 0 32 31 6 0 0 6 0 9856 6 0 0 0 0 96 0 0 0 0 2 3 4 100 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}

TEST_F(GatherV2Tiling, gather_v2_tiling_41) {
  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"l1_size\":2097152, "
      "\"indices_dsize\":4, \"params_dsize\":4, \"batch_dims\":1}, \"is_tik\": true,\"is_gather_v2\": true}";
  std::vector<int64_t> inputA{2, 16000, 2};
  std::vector<int64_t> inputB{2, 2};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> axis{1};
  std::vector<int64_t> output{2, 2, 2};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 1;
  string expectTilingData = "41 1 16000 2 4 0 1 0 4 0 0 4 0 16256 4 0 0 0 0 64000 0 0 0 0 2 2 0 2 ";
  Compute(inputA, inputB, inputC, axis, output, dtypeA, dtypeB, dtypeC, dtypeOutput, this->test_info_->name(),
          compileInfo, expectTilingData, batch_dims);
}


TEST_F(GatherV2Tiling, gather_v2_tiling_rt) {
  std::string compileInfo = R"({
    "attr_name": "batch_dims",
    "batch_dims_attr_idx": 0,
    "_pattern": "Gather",
    "_base_info": [
        32,
        262144,
        0,
        4,
        8
    ],
    "_custom_info": [
        32768,
        1,
        false,
        0
    ],
    "_tensor_sizes": {
        "7": [
            15392,
            1924
        ],
        "6": [
            26208,
            3276
        ],
        "1": [
            26208,
            3276
        ],
        "2": [
            26208,
            3276
        ],
        "5": [
            26208,
            3276
        ],
        "0": [
            52416,
            6552
        ]
    },
    "_gather_vars": {
        "900017000": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40000
        ],
        "900017001": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40001
        ],
        "900017002": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40002
        ],
        "900017003": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40003
        ],
        "900017005": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30001,
            40001
        ],
        "900017006": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30001,
            40002
        ],
        "900017007": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30001,
            40003
        ],
        "900017010": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30002,
            40002
        ],
        "900017011": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30002,
            40003
        ],
        "900017015": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30003,
            40003
        ],
        "900016000": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40000
        ],
        "900016001": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40001
        ],
        "900016002": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40002
        ],
        "900016003": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40003
        ],
        "900016005": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30001,
            40001
        ],
        "900016006": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30001,
            40002
        ],
        "900016007": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30001,
            40003
        ],
        "900016010": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30002,
            40002
        ],
        "900016011": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30002,
            40003
        ],
        "900016015": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30003,
            40003
        ],
        "900011000": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40000
        ],
        "900012000": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40000
        ],
        "900011001": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40001
        ],
        "900012001": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40001
        ],
        "900011002": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40002
        ],
        "900012002": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40002
        ],
        "900011003": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40003
        ],
        "900012003": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40003
        ],
        "900011005": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30001,
            40001
        ],
        "900012005": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30001,
            40001
        ],
        "900011006": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30001,
            40002
        ],
        "900012006": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30001,
            40002
        ],
        "900011007": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30001,
            40003
        ],
        "900012007": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30001,
            40003
        ],
        "900011010": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30002,
            40002
        ],
        "900012010": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30002,
            40002
        ],
        "900011011": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30002,
            40003
        ],
        "900012011": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30002,
            40003
        ],
        "900011015": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30003,
            40003
        ],
        "900012015": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30003,
            40003
        ],
        "900015000": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40000
        ],
        "900015001": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40001
        ],
        "900015002": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40002
        ],
        "900015003": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40003
        ],
        "900015005": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30001,
            40001
        ],
        "900015006": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30001,
            40002
        ],
        "900015007": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30001,
            40003
        ],
        "900015010": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30002,
            40002
        ],
        "900015011": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30002,
            40003
        ],
        "900015015": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30003,
            40003
        ],
        "900010000": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40000
        ],
        "900010001": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40001
        ],
        "900010002": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40002
        ],
        "900010003": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30000,
            40003
        ],
        "900010005": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30001,
            40001
        ],
        "900010006": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30001,
            40002
        ],
        "900010007": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30001,
            40003
        ],
        "900010010": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30002,
            40002
        ],
        "900010011": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30002,
            40003
        ],
        "900010015": [
            10000,
            10001,
            10002,
            10003,
            20001,
            30003,
            40003
        ]
    },
    "_vars": {
        "900017000": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_0"
        ],
        "900017001": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_1"
        ],
        "900017002": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_2"
        ],
        "900017003": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_3"
        ],
        "900017005": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_1"
        ],
        "900017006": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_2"
        ],
        "900017007": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_3"
        ],
        "900017010": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_2"
        ],
        "900017011": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_3"
        ],
        "900017015": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_3",
            "_ub_factor_3"
        ],
        "900016000": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_0"
        ],
        "900016001": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_1"
        ],
        "900016002": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_2"
        ],
        "900016003": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_3"
        ],
        "900016005": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_1"
        ],
        "900016006": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_2"
        ],
        "900016007": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_3"
        ],
        "900016010": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_2"
        ],
        "900016011": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_3"
        ],
        "900016015": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_3",
            "_ub_factor_3"
        ],
        "900011000": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_0"
        ],
        "900012000": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_0"
        ],
        "900011001": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_1"
        ],
        "900012001": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_1"
        ],
        "900011002": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_2"
        ],
        "900012002": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_2"
        ],
        "900011003": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_3"
        ],
        "900012003": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_3"
        ],
        "900011005": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_1"
        ],
        "900012005": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_1"
        ],
        "900011006": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_2"
        ],
        "900012006": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_2"
        ],
        "900011007": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_3"
        ],
        "900012007": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_3"
        ],
        "900011010": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_2"
        ],
        "900012010": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_2"
        ],
        "900011011": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_3"
        ],
        "900012011": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_3"
        ],
        "900011015": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_3",
            "_ub_factor_3"
        ],
        "900012015": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_3",
            "_ub_factor_3"
        ],
        "900015000": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_0"
        ],
        "900015001": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_1"
        ],
        "900015002": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_2"
        ],
        "900015003": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_3"
        ],
        "900015005": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_1"
        ],
        "900015006": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_2"
        ],
        "900015007": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_3"
        ],
        "900015010": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_2"
        ],
        "900015011": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_3"
        ],
        "900015015": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_3",
            "_ub_factor_3"
        ],
        "900010000": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_0"
        ],
        "900010001": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_1"
        ],
        "900010002": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_2"
        ],
        "900010003": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_3"
        ],
        "900010005": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_1"
        ],
        "900010006": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_2"
        ],
        "900010007": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_3"
        ],
        "900010010": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_2"
        ],
        "900010011": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_3"
        ],
        "900010015": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_3",
            "_ub_factor_3"
        ]
    },
    "_normal_vars": {
        "900017000": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_0"
        ],
        "900017001": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_1"
        ],
        "900017002": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_2"
        ],
        "900017003": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_3"
        ],
        "900017005": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_1"
        ],
        "900017006": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_2"
        ],
        "900017007": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_3"
        ],
        "900017010": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_2"
        ],
        "900017011": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_3"
        ],
        "900017015": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_3",
            "_ub_factor_3"
        ],
        "900016000": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_0"
        ],
        "900016001": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_1"
        ],
        "900016002": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_2"
        ],
        "900016003": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_3"
        ],
        "900016005": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_1"
        ],
        "900016006": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_2"
        ],
        "900016007": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_3"
        ],
        "900016010": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_2"
        ],
        "900016011": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_3"
        ],
        "900016015": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_3",
            "_ub_factor_3"
        ],
        "900011000": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_0"
        ],
        "900012000": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_0"
        ],
        "900011001": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_1"
        ],
        "900012001": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_1"
        ],
        "900011002": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_2"
        ],
        "900012002": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_2"
        ],
        "900011003": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_3"
        ],
        "900012003": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_3"
        ],
        "900011005": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_1"
        ],
        "900012005": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_1"
        ],
        "900011006": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_2"
        ],
        "900012006": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_2"
        ],
        "900011007": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_3"
        ],
        "900012007": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_3"
        ],
        "900011010": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_2"
        ],
        "900012010": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_2"
        ],
        "900011011": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_3"
        ],
        "900012011": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_3"
        ],
        "900011015": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_3",
            "_ub_factor_3"
        ],
        "900012015": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_3",
            "_ub_factor_3"
        ],
        "900015000": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_0"
        ],
        "900015001": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_1"
        ],
        "900015002": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_2"
        ],
        "900015003": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_3"
        ],
        "900015005": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_1"
        ],
        "900015006": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_2"
        ],
        "900015007": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_3"
        ],
        "900015010": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_2"
        ],
        "900015011": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_3"
        ],
        "900015015": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_3",
            "_ub_factor_3"
        ],
        "900010000": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_0"
        ],
        "900010001": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_1"
        ],
        "900010002": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_2"
        ],
        "900010003": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_0",
            "_ub_factor_3"
        ],
        "900010005": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_1"
        ],
        "900010006": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_2"
        ],
        "900010007": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_1",
            "_ub_factor_3"
        ],
        "900010010": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_2"
        ],
        "900010011": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_2",
            "_ub_factor_3"
        ],
        "900010015": [
            "_params_dim_0",
            "_params_dim_1",
            "_params_dim_2",
            "_params_dim_3",
            "_indices_dim_1",
            "_block_factor_3",
            "_ub_factor_3"
        ]
    },
    "_custom_vars": {
        "900017000": [],
        "900017001": [],
        "900017002": [],
        "900017003": [],
        "900017005": [],
        "900017006": [],
        "900017007": [],
        "900017010": [],
        "900017011": [],
        "900017015": [],
        "900016000": [],
        "900016001": [],
        "900016002": [],
        "900016003": [],
        "900016005": [],
        "900016006": [],
        "900016007": [],
        "900016010": [],
        "900016011": [],
        "900016015": [],
        "900011000": [],
        "900012000": [],
        "900011001": [],
        "900012001": [],
        "900011002": [],
        "900012002": [],
        "900011003": [],
        "900012003": [],
        "900011005": [],
        "900012005": [],
        "900011006": [],
        "900012006": [],
        "900011007": [],
        "900012007": [],
        "900011010": [],
        "900012010": [],
        "900011011": [],
        "900012011": [],
        "900011015": [],
        "900012015": [],
        "900015000": [],
        "900015001": [],
        "900015002": [],
        "900015003": [],
        "900015005": [],
        "900015006": [],
        "900015007": [],
        "900015010": [],
        "900015011": [],
        "900015015": [],
        "900010000": [],
        "900010001": [],
        "900010002": [],
        "900010003": [],
        "900010005": [],
        "900010006": [],
        "900010007": [],
        "900010010": [],
        "900010011": [],
        "900010015": []
    }
})";

  std::vector <int64_t> inputA{6400, 225};
  std::vector <int64_t> inputB{225};
  std::vector <int64_t> inputC{1};
  std::vector <int64_t> output{6400,224};
  std::vector <int32_t> axis{1};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT64;
  ge::DataType dtypeC = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  int64_t batch_dims = 0;
  string expectTilingData = "1 6400 225 1 225 200 8 ";
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
  TENSOR_INPUT_CONST(opParas, tensor_inputC, axis, (const uint8_t *) axis.data(), axis.size() * 4);
  TENSOR_OUTPUT(opParas, tensor_output, y);
  opParas.SetAttr("batch_dims", batch_dims);

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(ut_util::to_string_int32(runInfo.GetAllTilingData()), expectTilingData);
 
  optiling::GatherV2CompileInfo info;
  int64_t tiling_len = sizeof(optiling::GatherV2TilingParams);
  TILING_PARSE_JSON_TO_COMPILEINFO("GatherV2",compileInfo,info);
  vector<bool> input_const={false, false, true};
  vector<string> attrs={"batch_dims"};
  ATTACH_OPERATOR_TO_HOLDER_CONST(holder,opParas, input_const, attrs, tiling_len, info);
  HOLDER_DO_TILING(holder,"GatherV2",ge::GRAPH_SUCCESS);
  TILING_DATA_VERIFY_BYTYPE(holder, int32_t, "1 6400 225 1 225 200 8 ");

}
