
#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class clip_by_value : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "clip_by_value SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "clip_by_value TearDown" << std::endl;
  }
};

TEST_F(clip_by_value, clip_by_value_infershape_diff_test_1) {
  ge::op::ClipByValue op;
  op.UpdateInputDesc("x", create_desc_shape_range({16000, 1}, ge::DT_FLOAT, ge::FORMAT_NHWC, {16000, 1}, ge::FORMAT_NHWC, {{16000,16000},{1,1}}));

  ge::TensorDesc tensor_x = op.GetInputDesc("x");
  tensor_x.SetRealDimCnt(2);
  op.UpdateInputDesc("x", tensor_x);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}