#include <gtest/gtest.h>
#include <vector>
#include "elewise_calculation_ops.h"
#include "op_proto_test_util.h"

class ErfinvTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "erfinv test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "erfinv test TearDown" << std::endl;
  }
};

TEST_F(ErfinvTest, erfinv_test_case_1) {
  ge::op::Erfinv erfinv_op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({10, 8, 6});
  tensorDesc.SetDataType(ge::DT_FLOAT);
  tensorDesc.SetShape(shape);
  tensorDesc.SetOriginShape(shape);

  erfinv_op.UpdateInputDesc("input_x", tensorDesc);

  auto status = erfinv_op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = erfinv_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = erfinv_op.GetOutputDesc("output_y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {10, 8, 6};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}