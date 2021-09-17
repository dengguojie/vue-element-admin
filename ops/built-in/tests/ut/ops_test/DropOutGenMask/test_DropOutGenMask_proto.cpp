#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "random_ops.h"
#include "array_ops.h"
#include "utils/attr_utils.h"
#include "utils/op_desc_utils.h"

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

TEST_F(dropOutGenMask, dropOutGenMask_infershape_unconstData_test){
  ge::op::DropOutGenMask op;

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto input_sizes_desc = op_desc->MutableInputDesc("shape");
  std::vector<std::pair<int64_t, int64_t>> value_range = {{1, 2}, {32, 32}, {4, 5}, {4, 5}};
  input_sizes_desc->SetValueRange(value_range);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
