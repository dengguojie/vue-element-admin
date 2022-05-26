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

TEST_F(OpUtilTest, op_util_test_func_AddReducMeanCof_fp32) {
  gert::Shape input_shape({2, 5, 4});
  std::vector<int32_t> reduce_axis = {1};
  ge::DataType input_dtype = ge::DT_FLOAT;
  auto param = gert::TilingData::CreateCap(10);
  auto faker = gert::TilingContextFaker().NodeIoNum(0, 0).TilingData(param.get());
  auto holder = faker.Build();
  gert::TilingContext* context = holder.GetContext<gert::TilingContext>();
  gert::TilingData* tiling_data = context->GetRawTilingData();

  EXPECT_EQ(optiling::AddReducMeanCof(input_shape, input_dtype, reduce_axis, tiling_data), true);
  const float* data = reinterpret_cast<const float*>(tiling_data->GetData());
  EXPECT_EQ(ge::ConcatString(data[0]), "0.2");
}

TEST_F(OpUtilTest, op_util_test_func_AddReducMeanCof_fp16) {
  gert::Shape input_shape({2, 5, 4});
  std::vector<int32_t> reduce_axis = {1};
  ge::DataType input_dtype = ge::DT_FLOAT16;
  auto param = gert::TilingData::CreateCap(10);
  auto faker = gert::TilingContextFaker().NodeIoNum(0, 0).TilingData(param.get());
  auto holder = faker.Build();
  gert::TilingContext* context = holder.GetContext<gert::TilingContext>();
  gert::TilingData* tiling_data = context->GetRawTilingData();

  EXPECT_EQ(optiling::AddReducMeanCof(input_shape, input_dtype, reduce_axis, tiling_data), true);
}