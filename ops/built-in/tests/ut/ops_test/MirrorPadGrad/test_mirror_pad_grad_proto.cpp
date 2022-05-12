#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "array_ops_shape_fns.h"

using namespace ge;
using namespace op;

class mirror_pad_grad_infer_shape_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "mirror_pad_grad_infershape SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "mirror_pad_grad_infershape TearDown" << std::endl;
  }
};

TEST_F(mirror_pad_grad_infer_shape_test, mirror_pad_grad_infer_shape_05) {
  ge::op::MirrorPadGrad op;
  std::cout<< "mirror_pad_grad test_222!!!"<<std::endl;
  op.UpdateInputDesc("x", create_desc_with_ori({0, 5, 0}, ge::DT_INT32, ge::FORMAT_ND, {0, 5, 0}, ge::FORMAT_ND));
  op.UpdateInputDesc("paddings", create_desc_with_ori({3, 2}, ge::DT_INT32, ge::FORMAT_ND, {3, 2}, ge::FORMAT_ND));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expected_output_shape = {0, 5, 0};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}