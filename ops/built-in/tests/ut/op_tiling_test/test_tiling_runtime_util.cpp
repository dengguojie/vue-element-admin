#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "error_util.h"
#include "runtime/runtime2_util.h"
#include "kernel_run_context_facker.h"

using namespace std;

class OpUtilTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "OpUtilTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "OpUtilTest TearDown" << std::endl;
  }
};

TEST_F(OpUtilTest, op_util_test_func_AddReduceMeanCof_fp32) {
  gert::Shape input_shape({2, 5, 4});
  std::vector<int32_t> reduce_axis = {1};
  ge::DataType input_dtype = ge::DT_FLOAT;
  auto param = gert::TilingData::CreateCap(10);
  gert::TilingData *tiling_data = reinterpret_cast<gert::TilingData *>(param.get());
  ASSERT_NE(tiling_data, nullptr);

  EXPECT_EQ(optiling::AddReduceMeanCof(input_shape, input_dtype, reduce_axis, tiling_data), true);
  const float* data = reinterpret_cast<const float*>(tiling_data->GetData());
  EXPECT_EQ(ge::ConcatString(data[0]), "0.2");
}

TEST_F(OpUtilTest, op_util_test_func_AddReduceMeanCof_fp16) {
  gert::Shape input_shape({2, 5, 4});
  std::vector<int32_t> reduce_axis = {1};
  ge::DataType input_dtype = ge::DT_FLOAT16;
  auto param = gert::TilingData::CreateCap(10);
  gert::TilingData *tiling_data = reinterpret_cast<gert::TilingData *>(param.get());
  ASSERT_NE(tiling_data, nullptr);

  EXPECT_EQ(optiling::AddReduceMeanCof(input_shape, input_dtype, reduce_axis, tiling_data), true);
}
