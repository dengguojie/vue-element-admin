#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "graph/utils/op_desc_utils.h"
#include "graph/graph.h"
#include "register/op_tiling_registry.h"
#include "op_tiling/auto_tiling_rt2.h"
#include "op_tiling/auto_tiling_context.h"
#include "op_tiling/vector_tiling_rt2.h"

#include "common_autotiling_util.h"

using namespace optiling;
class AutoTilingContextTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AutoTilingContextTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AutoTilingContextTest TearDown" << std::endl;
  }
};

TEST_F(AutoTilingContextTest, test_parse_no_pattern) {
  std::vector<std::vector<int64_t>> inputs{{10, 100}};
  std::vector<std::vector<int64_t>> outputs{{10, 100}};
  std::string compile_info = R"({"_pattern": "Test"})";
  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::AutoTilingCompileInfo tiling_info;
  EXPECT_EQ(test.TestParse(compile_info, &tiling_info), false);
}

TEST_F(AutoTilingContextTest, test_tiling_no_op_info) {
  std::vector<std::vector<int64_t>> inputs{{10, 100}};
  std::vector<std::vector<int64_t>> outputs{{10, 100}};
  std::string compile_info = R"({"_pattern": "Test"})";
  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  auto context = test.GetContext();
  EXPECT_EQ(DoAutoTiling(context, nullptr), false);
}

TEST_F(AutoTilingContextTest, test_tiling_no_pattern) {
  std::vector<std::vector<int64_t>> inputs{{10, 100}};
  std::vector<std::vector<int64_t>> outputs{{10, 100}};;
  std::string compile_info = R"({"_pattern": "Test"})";
  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::AutoTilingCompileInfo tiling_info;
  tiling_info.pattern = SchPattern::DEFAULT;
  OpInfo op_info(&tiling_info);
  auto context = test.GetContext();
  EXPECT_EQ(DoAutoTiling(context, &op_info), false);
}

TEST_F(AutoTilingContextTest, test_auto_tiling_context_SetNeedAtomic) {
  std::vector<std::vector<int64_t>> inputs{{10, 100}};
  std::vector<std::vector<int64_t>> outputs{{10, 100}};
  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  auto context = test.GetContext();
  auto auto_tiling_context = AutoTilingContext(context);
  auto_tiling_context.SetNeedAtomic(true);
  EXPECT_EQ(test.GetAtomicFlag(), true);
}

TEST_F(AutoTilingContextTest, test_auto_tiling_context_GetAttr_int) {
  std::vector<std::vector<int64_t>> inputs{{10, 100}};
  std::vector<std::vector<int64_t>> outputs{{10, 100}};
  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  std::vector<std::pair<std::string, int64_t>> attr_int = {{"int", -1}};
  test.SetAttrs<int64_t>(attr_int);
  auto context = test.GetContext();
  auto auto_tiling_context = AutoTilingContext(context);
  int64_t attr_int_value;
  EXPECT_EQ(auto_tiling_context.GetAttr("", 0, attr_int_value), true);
  EXPECT_EQ(attr_int_value, -1);
}

TEST_F(AutoTilingContextTest, test_auto_tiling_context_GetAttr_list) {
  std::vector<std::vector<int64_t>> inputs{{10, 100}};
  std::vector<std::vector<int64_t>> outputs{{10, 100}};
  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  std::vector<std::pair<std::string, std::vector<int64_t>>> attr_list = {{"list_int", {-1, -2}}};
  test.SetAttrs<std::vector<int64_t>>(attr_list);
  auto context = test.GetContext();
  auto auto_tiling_context = AutoTilingContext(context);
  std::vector<int64_t> attr_int_value;
  EXPECT_EQ(auto_tiling_context.GetAttr("", 0, attr_int_value), true);
  EXPECT_EQ(attr_int_value.size(), 2);
  EXPECT_EQ(attr_int_value[0], -1);
  EXPECT_EQ(attr_int_value[1], -2);
}

TEST_F(AutoTilingContextTest, test_auto_tiling_context_GetConstInput_int) {
  std::vector<std::vector<int64_t>> inputs{{1}};
  std::vector<std::vector<int64_t>> outputs{{1}};
  ge::DataType dtype = ge::DT_INT32;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  int32_t data[1] = {11};
  test.SetInt32ConstInput(0, data, 1);
  auto context = test.GetContext();
  auto auto_tiling_context = AutoTilingContext(context);
  int64_t const_int_value;
  EXPECT_EQ(auto_tiling_context.GetConstInput("", 0, const_int_value), true);
  EXPECT_EQ(const_int_value,11);
}

TEST_F(AutoTilingContextTest, test_auto_tiling_context_GetConstInput_list) {
  std::vector<std::vector<int64_t>> inputs{{2}};
  std::vector<std::vector<int64_t>> outputs{{2}};
  ge::DataType dtype = ge::DT_INT32;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  int32_t data[2] = {11, 105};
  test.SetInt32ConstInput(0, data, 2);
  auto context = test.GetContext();
  auto auto_tiling_context = AutoTilingContext(context);
  std::vector<int64_t> const_int_value;
  EXPECT_EQ(auto_tiling_context.GetConstInput("", 0, const_int_value), true);
  EXPECT_EQ(const_int_value.size(), 2);
  EXPECT_EQ(const_int_value[0], 11);
  EXPECT_EQ(const_int_value[1], 105);
}

TEST_F(AutoTilingContextTest, test_auto_tiling_context_GetShapeSize) {
  std::vector<std::vector<int64_t>> inputs{{10, 100}};
  std::vector<std::vector<int64_t>> outputs{{10, 100}};
  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  auto context = test.GetContext();
  auto auto_tiling_context = AutoTilingContext(context);
  auto shape = auto_tiling_context.GetInputShape(0);
  EXPECT_EQ(shape.GetShapeSize(), 1000);
}