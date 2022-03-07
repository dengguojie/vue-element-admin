#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "image_ops.h"
#include "array_ops.h"
#include "test_common.h"
#include "common/utils/ut_op_util.h"
using namespace ge;
using namespace ut_util;

class ResizeNearestNeighborV2Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ResizeNearestNeighborV2Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ResizeNearestNeighborV2Tiling TearDown" << std::endl;
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

TEST_F(ResizeNearestNeighborV2Tiling, resize_nearest_neighbor_tiling_0) {
  std::string op_name = "ResizeNearestNeighborV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::ResizeNearestNeighborV2("ResizeNearestNeighborV2");

  std::string compileInfo =
      "{\"vars\": {\"max_w_len\": 1305, \"core_num\": 32, \"align_corners\": 0, \"half_pixel_centers\": 0, "
      "\"strides_h\": 1, \"strides_w\": 1, \"padding\": 0}}";

  std::vector<int64_t> input{16, 1, 1000, 1000, 16};
  std::vector<int64_t> output{16, 1, 1000, 1000, 16};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "113000 16 1 1000 1000 1000 1000 16 1 2 ");
}

TEST_F(ResizeNearestNeighborV2Tiling, resize_nearest_neighbor_tiling_2) {
  std::string op_name = "ResizeNearestNeighborV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::ResizeNearestNeighborV2("ResizeNearestNeighborV2");

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

TEST_F(ResizeNearestNeighborV2Tiling, resize_nearest_neighbor_tiling_3) {
  std::string op_name = "ResizeNearestNeighborV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::ResizeNearestNeighborV2("ResizeNearestNeighborV2");

  std::string compileInfo =
      "{\"vars\": {\"max_w_len\": 2944, \"core_num\": 48, \"align_corners\": 0, \"half_pixel_centers\": 0}}";

  std::vector<int64_t> input{32, 4, 22, 40, 16};
  std::vector<int64_t> output{32, 4, 44, 80, 16};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "111000 32 4 22 40 44 80 22 1 2 ");
}

TEST_F(ResizeNearestNeighborV2Tiling, resize_nearest_neighbor_tiling_4) {
  std::string op_name = "ResizeNearestNeighborV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::ResizeNearestNeighborV2("ResizeNearestNeighborV2");

  std::string compileInfo =
      "{\"vars\": {\"max_w_len\": 2944, \"core_num\": 48, \"align_corners\": 0, \"half_pixel_centers\": 0}}";

  std::vector<int64_t> input{2, 16, 48, 80, 16};
  std::vector<int64_t> output{2, 16, 96, 160, 16};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "111000 2 16 48 80 96 160 16 1 3 ");
}

TEST_F(ResizeNearestNeighborV2Tiling, resize_nearest_neighbor_tiling_5) {
  std::string op_name = "ResizeNearestNeighborV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::ResizeNearestNeighborV2("ResizeNearestNeighborV2");

  std::string compileInfo =
      "{\"vars\": {\"max_w_len\": 2944, \"core_num\": 48, \"align_corners\": 0, \"half_pixel_centers\": 0}}";

  std::vector<int64_t> input{1, 1, 2, 2, 16};
  std::vector<int64_t> output{1, 1, 256, 512, 16};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "100000 1 1 2 2 256 512 1 24 2 ");
}

TEST_F(ResizeNearestNeighborV2Tiling, resize_nearest_neighbor_tiling_6) {
  std::string op_name = "ResizeNearestNeighborV2";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::ResizeNearestNeighborV2("ResizeNearestNeighborV2");

  std::string compileInfo =
      "{\"vars\": {\"max_w_len\": 2944, \"core_num\": 48, \"align_corners\": 0, \"half_pixel_centers\": 0}}";

  std::vector<int64_t> input{2, 16, 48, 80, 16};
  std::vector<int64_t> output{2, 16, 96, 80, 16};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT, FORMAT_ND, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, DT_FLOAT, FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "113000 2 16 48 80 96 80 16 1 3 ");
}
