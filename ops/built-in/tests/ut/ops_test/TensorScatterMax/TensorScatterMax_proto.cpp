#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "matrix_calculation_ops.h"

class TensorScatterMax : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TensorScatterMax Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TensorScatterMax Proto Test TearDown" << std::endl;
  }
};

TEST_F(TensorScatterMax,
       tensor_scatter_max_infershape_verify_test) {
  ge::op::TensorScatterMax op;
  op.UpdateInputDesc("input", create_desc({2, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("indices", create_desc({8, 4}, ge::DT_INT32));
  op.UpdateInputDesc("updates", create_desc({8, }, ge::DT_FLOAT));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("output");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {2, 48, 16, 32};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}


TEST_F(TensorScatterMax,
       tensor_scatter_max_verify_dtype_test) {
  ge::op::TensorScatterMax op;
  op.UpdateInputDesc("input", create_desc({2, 48, 16, 32}, ge::DT_FLOAT));
  op.UpdateInputDesc("indices", create_desc({8, 4}, ge::DT_INT32));
  op.UpdateInputDesc("updates", create_desc({8, }, ge::DT_DOUBLE));

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}