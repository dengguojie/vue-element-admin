#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "random_ops.h"
#include "array_ops.h"

class dropOutGenMask : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "dropOutGenMask Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "dropOutGenMask Proto Test TearDown" << std::endl;
  }
};

TEST_F(dropOutGenMask, dropOutGenMask_infershape_diff_test){
  ge::op::DropOutGenMask op;
  op.UpdateInputDesc("x", create_desc({3}, ge::DT_INT32));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(dropOutGenMask, dropOutGenMask_infershape_prob_rank_err_1){
  ge::op::DropOutGenMask op;
  auto probDesc = op.GetInputDesc("prob");
  probDesc.SetDataType(ge::DT_FLOAT);
  probDesc.SetShape(ge::Shape({1}));
  probDesc.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("prob", probDesc);
  

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
