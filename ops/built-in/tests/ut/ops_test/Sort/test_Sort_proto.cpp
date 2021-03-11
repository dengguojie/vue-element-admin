#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_detect_ops.h"

class SortTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Sort Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Sort Test TearDown" << std::endl;
  }
};

TEST_F(SortTest, Sort_test_0){
  ge::op::Sort op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({10, 10, 20});
  tensorDesc.SetDataType(ge::DT_FLOAT16);
  tensorDesc.SetShape(shape);
  
  op.UpdateInputDesc("x",tensorDesc);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc_1 = op.GetOutputDesc("y1");
  auto output_desc_2 = op.GetOutputDesc("y2");
  EXPECT_EQ(output_desc_1.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output_desc_2.GetDataType(), ge::DT_INT32);
  
  std::vector<int64_t> expected_output_shape = {10, 10, 20};
  EXPECT_EQ(output_desc_1.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc_2.GetShape().GetDims(), expected_output_shape);
}

TEST_F(SortTest, Sort_test_1){
  ge::op::Sort op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({10, 200});
  tensorDesc.SetDataType(ge::DT_FLOAT16);
  tensorDesc.SetShape(shape);
  
  op.UpdateInputDesc("x",tensorDesc);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc_1 = op.GetOutputDesc("y1");
  auto output_desc_2 = op.GetOutputDesc("y2");
  EXPECT_EQ(output_desc_1.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output_desc_2.GetDataType(), ge::DT_INT32);
  
  std::vector<int64_t> expected_output_shape = {10, 200};
  EXPECT_EQ(output_desc_1.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc_2.GetShape().GetDims(), expected_output_shape);
}

TEST_F(SortTest, Sort_test_2){
  ge::op::Sort op;
  ge::TensorDesc tensorDesc;
  ge::Shape shape({2000});
  tensorDesc.SetDataType(ge::DT_FLOAT16);
  tensorDesc.SetShape(shape);
  
  op.UpdateInputDesc("x",tensorDesc);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc_1 = op.GetOutputDesc("y1");
  auto output_desc_2 = op.GetOutputDesc("y2");
  EXPECT_EQ(output_desc_1.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output_desc_2.GetDataType(), ge::DT_INT32);
  
  std::vector<int64_t> expected_output_shape = {2000};
  EXPECT_EQ(output_desc_1.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(output_desc_2.GetShape().GetDims(), expected_output_shape);
}
