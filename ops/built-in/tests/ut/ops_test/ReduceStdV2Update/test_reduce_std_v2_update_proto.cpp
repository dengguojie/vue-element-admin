#include <gtest/gtest.h>
#include <vector>
#include "reduce_ops.h"

class ReduceStdV2UpdateTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "reduce_std_v2_update test SetUp" << std::endl;
}

  static void TearDownTestCase() {
    std::cout << "reduce_std_v2_update test TearDown" << std::endl;
    }
};

TEST_F(ReduceStdV2UpdateTest, reduce_std_v2_update_test_case_1) {
  ge::op::ReduceStdV2Update reduce_std_v2_update_op;
  
  ge::TensorDesc tensorDesc;
  ge::Shape shape({-1, -1});
  ge::Shape ori_shape({3, 4});
  
  tensorDesc.SetDataType(ge::DT_FLOAT);
  tensorDesc.SetShape(shape);
  tensorDesc.SetFormat(ge::FORMAT_ND);
  tensorDesc.SetOriginShape(shape);
  tensorDesc.SetOriginFormat(ge::FORMAT_ND);
  tensorDesc.SetShapeRange({{3, 3}, {4, 4}});
  
  reduce_std_v2_update_op.UpdateInputDesc("x", tensorDesc);
  reduce_std_v2_update_op.UpdateInputDesc("mean", tensorDesc);
  reduce_std_v2_update_op.SetAttr("dim", {1,});
  reduce_std_v2_update_op.SetAttr("if_std", true);
  reduce_std_v2_update_op.SetAttr("unbiased", true);
  reduce_std_v2_update_op.SetAttr("keepdim", true);
  
  auto ret = reduce_std_v2_update_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  
  auto output_desc = reduce_std_v2_update_op.GetOutputDescByName("output_var");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  
  std::vector<int64_t> expected_output_shape = {-1, 1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
}

TEST_F(ReduceStdV2UpdateTest, reduce_std_v2_update_test_case_2) {

  ge::op::ReduceStdV2Update reduce_std_v2_update_op;
  ge::TensorDesc tensorDesc;

  tensorDesc.SetDataType(ge::DT_FLOAT16);
  reduce_std_v2_update_op.UpdateInputDesc("x", tensorDesc);

  tensorDesc.SetDataType(ge::DT_FLOAT);
  reduce_std_v2_update_op.UpdateInputDesc("mean", tensorDesc);

  auto ret = reduce_std_v2_update_op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
