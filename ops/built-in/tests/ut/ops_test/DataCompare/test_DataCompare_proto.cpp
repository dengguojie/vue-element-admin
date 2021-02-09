#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class datacompare : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DataCompare Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DataCompare Proto Test TearDown" << std::endl;
  }
};

TEST_F(datacompare, datacompare_infershape_diff_test){
  ge::op::DataCompare op;
  op.UpdateInputDesc("x1", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("x2", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  
  float value =  0.001;
  op.SetAttr("atol", value);
  op.SetAttr("rtol", value);
  
  auto ret = op.InferShapeAndType();
  ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("num");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(datacompare, datacompare_verify_test){
  ge::op::DataCompare op;
  op.UpdateInputDesc("x1", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("x2", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  
  float value =  -0.001;
  op.SetAttr("atol", value);
  op.SetAttr("rtol", value);
  
  auto ret = op.InferShapeAndType();
  ret = op.VerifyAllAttr(true);

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

}