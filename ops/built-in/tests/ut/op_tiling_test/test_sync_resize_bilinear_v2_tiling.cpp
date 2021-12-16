#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "common/utils/ut_profiling_reg.h"
#include "image_ops.h"
#include "array_ops.h"
#include "test_common.h"
#include "common/utils/ut_op_util.h"
using namespace ge;
using namespace ut_util;

class SyncResizeBilinearV2Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SyncResizeBilinearV2Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SyncResizeBilinearV2Tiling TearDown" << std::endl;
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


TEST_F(SyncResizeBilinearV2Tiling, resize_bilinear_tiling_0) {
  std::string op_name = "SyncResizeBilinearV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SyncResizeBilinearV2("SyncResizeBilinearV2");

  std::string compileInfo =
      "{\"vars\": {\"max_w_len\": 1305, \"core_num\": 32, \"align_corners\": 0, \"half_pixel_centers\": 0, "
      "\"strides_h\": 1, \"strides_w\": 1, \"padding\": 0}}";

  std::vector<int64_t> input{16, 256, 7, 7, 16};
  std::vector<int64_t> output{16, 256, 33, 33, 16};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "100110 4096 1 7 7 33 33 4 7 1 ");
}

TEST_F(SyncResizeBilinearV2Tiling, resize_bilinear_tiling_2) {
  std::string op_name = "SyncResizeBilinearV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SyncResizeBilinearV2("SyncResizeBilinearV2");

  std::string compileInfo =
      "{\"vars\": {\"max_w_len\": 1305, \"core_num\": 32, \"align_corners\": 0, \"half_pixel_centers\": 0, "
      "\"strides_h\": 1, \"strides_w\": 1, \"padding\": 0}}";

  std::vector<int64_t> input{16, 1, 1000, 1000, 16};
  std::vector<int64_t> output{16, 1, 999, 999, 16};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "100000 16 1 1000 1000 999 999 2 4 4 ");
}

TEST_F(SyncResizeBilinearV2Tiling, resize_bilinear_tiling_3) {
  std::string op_name = "SyncResizeBilinearV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SyncResizeBilinearV2("SyncResizeBilinearV2");

  std::string compileInfo =
      "{\"vars\": {\"max_w_len\": 1305, \"core_num\": 32, \"align_corners\": 0, \"half_pixel_centers\": 0, "
      "\"strides_h\": 1, \"strides_w\": 1, \"padding\": 0}}";

  std::vector<int64_t> input{16, 1, 1000, 1000, 16};
  std::vector<int64_t> output{16, 1, 1000, 1000, 16};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "999999 16000000 1 1 1 1 1 32 1 1 ");
}

TEST_F(SyncResizeBilinearV2Tiling, resize_bilinear_tiling_4) {
  std::string op_name = "SyncResizeBilinearV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SyncResizeBilinearV2("SyncResizeBilinearV2");

  std::string compileInfo =
      R"({"vars": {"max_w_len": 1305, "core_num": 32, "align_corners": 0, "half_pixel_centers": 0,
          "strides_h": 1, "strides_w": 1, "padding": 0},
          "_tune_param": {"tune_param": {"tiling_key": 999999}}})";

  std::vector<int64_t> input{16, 1, 1000, 1000, 16};
  std::vector<int64_t> output{16, 1, 1000, 1000, 16};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "999999 16000000 1 1 1 1 1 32 1 1 ");
}

TEST_F(SyncResizeBilinearV2Tiling, resize_bilinear_tiling_5) {
  std::string op_name = "SyncResizeBilinearV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SyncResizeBilinearV2("SyncResizeBilinearV2");

  std::string compileInfo =
      R"({"vars": {"max_w_len": 1305, "core_num": 32, "align_corners": 0, "half_pixel_centers": 0,
          "strides_h": 1, "strides_w": 1, "padding": 0},
          "_tune_param": {"tune_param": {"tiling_key": 100110,
                                         "cut_batch_c1_num": 2,
                                         "cut_height_num": 16,
                                         "cut_width_num": 1}}})";

  std::vector<int64_t> input{16, 1, 1000, 1000, 16};
  std::vector<int64_t> output{16, 1, 999, 999, 16};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "100110 16 1 1000 1000 999 999 2 16 1 ");
}

TEST_F(SyncResizeBilinearV2Tiling, resize_bilinear_tiling_6) {
  std::string op_name = "SyncResizeBilinearV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SyncResizeBilinearV2("SyncResizeBilinearV2");

  std::string compileInfo =
      R"({"vars": {"max_w_len": 1305, "core_num": 32, "align_corners": 0, "half_pixel_centers": 0,
          "strides_h": 1, "strides_w": 1, "padding": 0},
          "_tune_param": {"tune_param": {"tiling_key": 100000,
                                         "cut_batch_c1_num": 3,
                                         "cut_height_num": 2,
                                         "cut_width_num": 5}}})";

  std::vector<int64_t> input{16, 256, 7, 7, 16};
  std::vector<int64_t> output{16, 256, 33, 33, 16};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "100000 4096 1 7 7 33 33 3 2 5 ");
}

TEST_F(SyncResizeBilinearV2Tiling, resize_bilinear_tiling_7) {
  std::string op_name = "SyncResizeBilinearV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::SyncResizeBilinearV2("SyncResizeBilinearV2");

  std::string compileInfo =
      R"({"vars": {"max_w_len": 1305, "core_num": 32, "align_corners": 0, "half_pixel_centers": 0,
          "strides_h": 1, "strides_w": 1, "padding": 0},
          "_tune_param": {"tune_param": {"tiling_key": 888888,
                                         "cut_batch_c1_num": 3,
                                         "cut_height_num": 2,
                                         "cut_width_num": 5}}})";

  std::vector<int64_t> input{16, 256, 7, 7, 16};
  std::vector<int64_t> output{16, 256, 33, 33, 16};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "100110 4096 1 7 7 33 33 4 7 1 ");
}
