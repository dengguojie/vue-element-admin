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

class ResizeTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ResizeTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ResizeTiling TearDown" << std::endl;
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

TEST_F(ResizeTiling, resize_tiling_0) {
  std::string op_name = "Resize";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Resize("Resize");
  opParas.SetAttr("mode", "linear");
  std::string compileInfo =
      "{\"vars\": {\"max_w_len\": 1305, \"core_num\": 32, \"align_corners\": 0, \"half_pixel_centers\": 0, "
      "\"strides_h\": 1, \"strides_w\": 1, \"padding\": 0, \"mode_name\": 21}}";

  std::vector<int64_t> input{16, 256, 7, 7, 16};
  std::vector<int64_t> output{16, 256, 33, 33, 16};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT, FORMAT_NC1HWC0, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, DT_FLOAT, FORMAT_NC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "100110 4096 1 7 7 33 33 4 7 1 ");
}


TEST_F(ResizeTiling, resize_tiling_1) {
  std::string op_name = "Resize";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Resize("Resize");
  opParas.SetAttr("mode", "nearest");
  std::string compileInfo =
      "{\"vars\": {\"max_w_len\": 1305, \"core_num\": 32, \"align_corners\": 0, \"half_pixel_centers\": 0, "
      "\"strides_h\": 1, \"strides_w\": 1, \"padding\": 0, \"mode_name\": 20}}";

  std::vector<int64_t> input{16, 1, 1000, 1000, 16};
  std::vector<int64_t> output{16, 1, 1000, 1000, 16};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT, FORMAT_NC1HWC0, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, DT_FLOAT, FORMAT_NC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "113000 16 1 1000 1000 1000 1000 16 1 2 ");
}

TEST_F(ResizeTiling, resize_tiling_2) {
  std::string op_name = "Resize";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Resize("Resize");
  opParas.SetAttr("mode", "nearest");
  std::string compileInfo =
      "{\"vars\": {\"core_num\": 32,  \"left_w\": 3764, \"mode_name\": 22}}";

  std::vector<int64_t> input{32, 1, 16, 128, 64, 16};
  std::vector<int64_t> output{32, 16, 16, 128, 32, 16};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT, FORMAT_NDC1HWC0, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, DT_FLOAT, FORMAT_NDC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 32 16 1 128 64 16 128 32 0 0 0 512 16 16 32 1 128 1 32 0 0 0 ");
}

TEST_F(ResizeTiling, resize_tiling_3) {
  std::string op_name = "Resize";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  auto opParas = op::Resize("Resize");
  opParas.SetAttr("mode", "nearest");
  std::string compileInfo =
      "{\"vars\": {\"core_num\": 32,  \"left_w\": 3764, \"mode_name\": 22}}";

  std::vector<int64_t> input{1, 1, 16, 1, 64, 16};
  std::vector<int64_t> output{1, 1, 16, 1, 256, 16};

  TENSOR_INPUT_WITH_SHAPE(opParas, x, input, DT_FLOAT, FORMAT_NDC1HWC0, {});
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, DT_FLOAT, FORMAT_NDC1HWC0, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "3 1 16 1 1 64 1 1 256 0 0 0 1 1 1 1 1 1 1 256 4 4 4 ");
}
