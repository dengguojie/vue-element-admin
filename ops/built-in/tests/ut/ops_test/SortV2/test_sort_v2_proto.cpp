#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"

class SortV2Test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SortV2Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SortV2Test TearDown" << std::endl;
  }
};

TEST_F(SortV2Test, Sort_test_0){
  ge::op::SortV2 op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({10, 10, 20});
  tensorDesc.SetDataType(ge::DT_FLOAT16);
  tensorDesc.SetShape(shape);
  tensorDesc.SetOriginShape(shape);
  
  op.UpdateInputDesc("x",tensorDesc);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
 
  std::vector<int64_t> expected_output_shape = {10, 10, 20};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(SortV2Test, Sort_test_1){
  ge::op::SortV2 op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({10, 200});
  tensorDesc.SetDataType(ge::DT_FLOAT16);
  tensorDesc.SetShape(shape);
  tensorDesc.SetOriginShape(shape);
  
  op.UpdateInputDesc("x",tensorDesc);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
 
  std::vector<int64_t> expected_output_shape = {10, 200};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(SortV2Test, Sort_test_2){
  ge::op::SortV2 op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({2000});
  tensorDesc.SetDataType(ge::DT_FLOAT16);
  tensorDesc.SetShape(shape);
  tensorDesc.SetOriginShape(shape);
  
  op.UpdateInputDesc("x",tensorDesc);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
 
  std::vector<int64_t> expected_output_shape = {2000};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
