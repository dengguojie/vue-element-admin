#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "common/utils/ut_profiling_reg.h"
#include "image_ops.h"
#include "array_ops.h"

using namespace std;

class ResizeBilinearV2Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ResizeBilinearV2Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ResizeBilinearV2Tiling TearDown" << std::endl;
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

using namespace ge;
#include "test_common.h"
/*
.INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16,
                               DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(align_corners, Bool, false)
    .ATTR(half_pixel_centers, Bool, false)
*/

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_tiling_0) {
  std::string op_name = "ResizeBilinearV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::ResizeBilinearV2("ResizeBilinearV2");

  std::string compileInfo =
      "{\"vars\": {\"max_w_len\": 1305, \"core_num\": 32, \"align_corners\": 0, \"half_pixel_centers\": 0, "
      "\"strides_h\": 1, \"strides_w\": 1, \"padding\": 0}}";

  std::vector<int64_t> input{16, 256, 7, 7, 16};
  std::vector<int64_t> output{16, 256, 33, 33, 16};

  TensorDesc tensor_input;
  tensor_input.SetShape(ge::Shape(input));
  tensor_input.SetDataType(ge::DT_FLOAT);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT);

  TENSOR_INPUT(opParas, tensor_input, x);
  TENSOR_OUTPUT(opParas, tensor_output, y);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "100110 4096 1 7 7 33 33 4 7 1 ");
  // test performance start
  // PROFILING_TEST(iter->second, (opParas, op_compile_info, runInfo), 1000, 10);

  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  }
  // test performance end
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_tiling_2) {
  std::string op_name = "ResizeBilinearV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::ResizeBilinearV2("ResizeBilinearV2");

  std::string compileInfo =
      "{\"vars\": {\"max_w_len\": 1305, \"core_num\": 32, \"align_corners\": 0, \"half_pixel_centers\": 0, "
      "\"strides_h\": 1, \"strides_w\": 1, \"padding\": 0}}";

  std::vector<int64_t> input{16, 1, 1000, 1000, 16};
  std::vector<int64_t> output{16, 1, 999, 999, 16};

  TensorDesc tensor_input;
  tensor_input.SetShape(ge::Shape(input));
  tensor_input.SetDataType(ge::DT_FLOAT);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT);

  TENSOR_INPUT(opParas, tensor_input, x);
  TENSOR_OUTPUT(opParas, tensor_output, y);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "100000 16 1 1000 1000 999 999 2 4 4 ");

  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  }
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_tiling_3) {
  std::string op_name = "ResizeBilinearV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::ResizeBilinearV2("ResizeBilinearV2");

  std::string compileInfo =
      "{\"vars\": {\"max_w_len\": 1305, \"core_num\": 32, \"align_corners\": 0, \"half_pixel_centers\": 0, "
      "\"strides_h\": 1, \"strides_w\": 1, \"padding\": 0}}";

  std::vector<int64_t> input{16, 1, 1000, 1000, 16};
  std::vector<int64_t> output{16, 1, 1000, 1000, 16};

  TensorDesc tensor_input;
  tensor_input.SetShape(ge::Shape(input));
  tensor_input.SetDataType(ge::DT_FLOAT);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT);

  TENSOR_INPUT(opParas, tensor_input, x);
  TENSOR_OUTPUT(opParas, tensor_output, y);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "999999 16000000 1 1 1 1 1 32 1 1 ");

  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo);
  }
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_tiling_4) {
  std::string op_name = "ResizeBilinearV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::ResizeBilinearV2("ResizeBilinearV2");

  std::string compileInfo =
      R"({"vars": {"max_w_len": 1305, "core_num": 32, "align_corners": 0, "half_pixel_centers": 0,
          "strides_h": 1, "strides_w": 1, "padding": 0},
          "_tune_param": {"tune_param": {"tiling_key": 999999}}})";

  std::vector<int64_t> input{16, 1, 1000, 1000, 16};
  std::vector<int64_t> output{16, 1, 1000, 1000, 16};

  TensorDesc tensor_input;
  tensor_input.SetShape(ge::Shape(input));
  tensor_input.SetDataType(ge::DT_FLOAT);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT);

  TENSOR_INPUT(opParas, tensor_input, x);
  TENSOR_OUTPUT(opParas, tensor_output, y);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "999999 16000000 1 1 1 1 1 32 1 1 ");
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_tiling_5) {
  std::string op_name = "ResizeBilinearV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::ResizeBilinearV2("ResizeBilinearV2");

  std::string compileInfo =
      R"({"vars": {"max_w_len": 1305, "core_num": 32, "align_corners": 0, "half_pixel_centers": 0,
          "strides_h": 1, "strides_w": 1, "padding": 0},
          "_tune_param": {"tune_param": {"tiling_key": 100110,
                                         "cut_batch_c1_num": 2,
                                         "cut_height_num": 16,
                                         "cut_width_num": 1}}})";

  std::vector<int64_t> input{16, 1, 1000, 1000, 16};
  std::vector<int64_t> output{16, 1, 999, 999, 16};

  TensorDesc tensor_input;
  tensor_input.SetShape(ge::Shape(input));
  tensor_input.SetDataType(ge::DT_FLOAT);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT);

  TENSOR_INPUT(opParas, tensor_input, x);
  TENSOR_OUTPUT(opParas, tensor_output, y);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "100110 16 1 1000 1000 999 999 2 16 1 ");
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_tiling_6) {
  std::string op_name = "ResizeBilinearV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::ResizeBilinearV2("ResizeBilinearV2");

  std::string compileInfo =
      R"({"vars": {"max_w_len": 1305, "core_num": 32, "align_corners": 0, "half_pixel_centers": 0,
          "strides_h": 1, "strides_w": 1, "padding": 0},
          "_tune_param": {"tune_param": {"tiling_key": 100000,
                                         "cut_batch_c1_num": 3,
                                         "cut_height_num": 2,
                                         "cut_width_num": 5}}})";

  std::vector<int64_t> input{16, 256, 7, 7, 16};
  std::vector<int64_t> output{16, 256, 33, 33, 16};

  TensorDesc tensor_input;
  tensor_input.SetShape(ge::Shape(input));
  tensor_input.SetDataType(ge::DT_FLOAT);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT);

  TENSOR_INPUT(opParas, tensor_input, x);
  TENSOR_OUTPUT(opParas, tensor_output, y);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "100000 4096 1 7 7 33 33 3 2 5 ");
}

TEST_F(ResizeBilinearV2Tiling, resize_bilinear_tiling_7) {
  std::string op_name = "ResizeBilinearV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::ResizeBilinearV2("ResizeBilinearV2");

  std::string compileInfo =
      R"({"vars": {"max_w_len": 1305, "core_num": 32, "align_corners": 0, "half_pixel_centers": 0,
          "strides_h": 1, "strides_w": 1, "padding": 0},
          "_tune_param": {"tune_param": {"tiling_key": 888888,
                                         "cut_batch_c1_num": 3,
                                         "cut_height_num": 2,
                                         "cut_width_num": 5}}})";

  std::vector<int64_t> input{16, 256, 7, 7, 16};
  std::vector<int64_t> output{16, 256, 33, 33, 16};

  TensorDesc tensor_input;
  tensor_input.SetShape(ge::Shape(input));
  tensor_input.SetDataType(ge::DT_FLOAT);
  TensorDesc tensor_output;
  tensor_output.SetShape(ge::Shape(output));
  tensor_output.SetDataType(ge::DT_FLOAT);

  TENSOR_INPUT(opParas, tensor_input, x);
  TENSOR_OUTPUT(opParas, tensor_output, y);

  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_v2_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "100110 4096 1 7 7 33 33 4 7 1 ");
}
