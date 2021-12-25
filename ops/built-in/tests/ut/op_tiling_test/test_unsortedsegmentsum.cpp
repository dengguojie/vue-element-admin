#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "selection_ops.h"
#include "array_ops.h"

using namespace std;

class UnsortedSegmentSumTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "UnsortedSegmentSumTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "UnsortedSegmentSumTiling TearDown" << std::endl;
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
  std::cout << "to_string" << std::endl;
  std::cout << result << std::endl;
  return result;
}

using namespace ge;
#include "common/utils/ut_op_util.h"
using namespace ut_util;

TEST_F(UnsortedSegmentSumTiling, segmentsum_tiling_0) {
  using namespace optiling;
  std::string op_name = "SegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{2,3132864};
  std::vector<int64_t> inputB{2};
  std::vector<int32_t> segment_ids_value{0,0};
  std::vector<int64_t> output{1,3132864};

  auto opParas = op::SegmentSum("SegmentSum");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, segment_ids, inputB, ge::DT_INT32, FORMAT_ND, segment_ids_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "17 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3132864 96 4096 32768 2488 19904 2 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_0) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{2,3132864};
  std::vector<int64_t> inputB{2};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments_value{1,};
  std::vector<int64_t> output{1,3132864};

  auto opParas = op::UnsortedSegmentSum("UnsortedSegmentSum");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, segment_ids, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num_segments, inputC, ge::DT_INT32, FORMAT_ND, num_segments_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "17 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3132864 96 4096 32768 2488 19904 2 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_1) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{1024,80};
  std::vector<int64_t> inputB{1024};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments_value{300,};
  std::vector<int64_t> output{300,80};

  auto opParas = op::UnsortedSegmentSum("UnsortedSegmentSum");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, segment_ids, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num_segments, inputC, ge::DT_INT32, FORMAT_ND, num_segments_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 32 2560 1 320 320 2560 2560 32 32 1 320 320 2560 2560 32 32 2560 1 320 320 2560 2560 32 32 1 320 320 2560 2560 32 32 80 1 10 80 10 80 1024 32 1 4 4 32 32 32 1 4 4 32 32 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_2) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{1024,80};
  std::vector<int64_t> inputB{1024};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments_value{300,};
  std::vector<int64_t> output{300,80};

  auto opParas = op::UnsortedSegmentSum("UnsortedSegmentSum");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputA, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, segment_ids, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num_segments, inputC, ge::DT_INT32, FORMAT_ND, num_segments_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT16, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "9 32 9 21 1024 1 1024 128 1024 128 80 1 5 80 1 5 80 1 0 0 5 5 419 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_3) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{2,3132864};
  std::vector<int64_t> inputB{2};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments_value{1,};
  std::vector<int64_t> output{1,3132864};

  auto opParas = op::UnsortedSegmentSum("UnsortedSegmentSum");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputA, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, segment_ids, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num_segments, inputC, ge::DT_INT32, FORMAT_ND, num_segments_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT16, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "15 32 97888 98336 2 1 2 1 2 1 3132864 3 2048 32768 1 2022 32 253 0 0 0 0 0 0 0 0 0 0 0 2048 2 2 1 1 4 1 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_4) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{46,44};
  std::vector<int64_t> inputB{46};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments_value{100,};
  std::vector<int64_t> output{100,44};

  auto opParas = op::UnsortedSegmentSum("UnsortedSegmentSum");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, segment_ids, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num_segments, inputC, ge::DT_INT32, FORMAT_ND, num_segments_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4 32 44 1 6 6 44 44 1 1 1 6 6 44 44 1 1 660 1 83 83 660 660 15 15 1 83 83 660 660 15 15 44 1 5 40 1 4 46 1 1 1 1 1 1 15 1 2 2 15 15 1 1 1 1 2 1 2 1 40 48 4 0 1 1 1 1 2 1 2 1 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_5) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{0,44};
  std::vector<int64_t> inputB{0};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments_value{100,};
  std::vector<int64_t> output{100,44};

  auto opParas = op::UnsortedSegmentSum("UnsortedSegmentSum");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, segment_ids, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num_segments, inputC, ge::DT_INT32, FORMAT_ND, num_segments_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_6) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{1024,15};
  std::vector<int64_t> inputB{1024};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments_value{600,};
  std::vector<int64_t> output{300,80};

  auto opParas = op::UnsortedSegmentSum("UnsortedSegmentSum");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputA, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, segment_ids, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num_segments, inputC, ge::DT_INT32, FORMAT_ND, num_segments_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT16, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "13 32 16 104 1024 1 1024 128 1024 128 15 1 15 15 1 15 15 1 0 8 1 1 2180 1 1 16 16 104 104 97 97 0 0 0 0 0 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_7) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{2,3132864};
  std::vector<int64_t> inputB{2};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments_value{2,};
  std::vector<int64_t> output{1,3132864};

  auto opParas = op::UnsortedSegmentSum("UnsortedSegmentSum");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputA, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, segment_ids, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num_segments, inputC, ge::DT_INT32, FORMAT_ND, num_segments_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT16, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "11 2 1 1 2 1 2 1 2 1 3132864 196 1000 16000 125 804 12864 101 0 0 1000 804 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_8) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{1024,15};
  std::vector<int64_t> inputB{1024};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments_value{100000,};
  std::vector<int64_t> output{300,80};

  auto opParas = op::UnsortedSegmentSum("UnsortedSegmentSum");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputA, ge::DT_FLOAT16, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, segment_ids, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num_segments, inputC, ge::DT_INT32, FORMAT_ND, num_segments_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT16, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "13 32 3120 3280 1024 1 1024 128 1024 128 15 1 2043 15 1 881 15 1 12 12 1 1 2180 2 2 2180 940 2180 1100 2043 1031 0 0 0 0 0 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_9) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{1024,1};
  std::vector<int64_t> inputB{1024};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments_value{300,};
  std::vector<int64_t> output{300,1};

  auto opParas = op::UnsortedSegmentSum("UnsortedSegmentSum");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, segment_ids, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num_segments, inputC, ge::DT_INT32, FORMAT_ND, num_segments_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "8 16 64 0 8 0 64 0 64 0 0 8 0 64 0 64 0 64 0 8 0 64 0 64 0 0 8 0 64 0 64 0 1 1 1 0 0 0 1024 64 1 8 8 64 64 64 1 8 8 64 64 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_10) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{1024,20000};
  std::vector<int64_t> inputB{1024};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments_value{300,};
  std::vector<int64_t> output{300,80};

  auto opParas = op::UnsortedSegmentSum("UnsortedSegmentSum");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, segment_ids, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num_segments, inputC, ge::DT_INT32, FORMAT_ND, num_segments_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "5 32 640000 32 0 0 16384 3616 0 0 32 0 0 16384 3616 0 0 640000 32 0 0 16384 3616 0 0 32 0 0 16384 3616 0 0 20000 2 2048 16384 452 3616 1024 32 1 4 4 32 32 32 1 4 4 32 32 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ");
}
TEST_F(UnsortedSegmentSumTiling, unsortedsegmentsum_tiling_11) {
  using namespace optiling;
  std::string op_name = "UnsortedSegmentSum";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("UnsortedSegmentSum");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"ub_size\": 131072, \"core_num\": 32, \"dtype\":\"float32\", \"ub_tensor_num\":2}}";

  std::vector<int64_t> inputA{1024,19994};
  std::vector<int64_t> inputB{1024};
  std::vector<int64_t> inputC{1};
  std::vector<int32_t> num_segments_value{300,};
  std::vector<int64_t> output{300,80};

  auto opParas = op::UnsortedSegmentSum("UnsortedSegmentSum");
  TENSOR_INPUT_WITH_SHAPE(opParas, x, inputA, ge::DT_FLOAT, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(opParas, segment_ids, inputB, ge::DT_INT32, ge::FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(opParas, num_segments, inputC, ge::DT_INT32, FORMAT_ND, num_segments_value);
  TENSOR_OUTPUT_WITH_SHAPE(opParas, y, output, ge::DT_FLOAT, ge::FORMAT_ND, {});

  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "6 32 639808 32 0 0 10920 9074 0 0 32 0 0 10920 9074 0 0 639808 32 0 0 10920 9074 0 0 32 0 0 10920 9074 0 0 19994 2 1365 10920 1134 9074 1024 32 1 4 4 32 32 32 1 4 4 32 32 0 0 0 0 0 0 0 0 9072 9080 2 1135 0 0 0 0 0 0 0 0 ");
}