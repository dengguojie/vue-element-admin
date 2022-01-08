#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "random_ops.h"
#include "array_ops.h"
#include "utils/attr_utils.h"
#include "utils/op_desc_utils.h"

class statelessDropOutGenMask : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "statelessDropOutGenMask Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "statelessDropOutGenMask Proto Test TearDown" << std::endl;
  }
};

TEST_F(statelessDropOutGenMask, statelessDropOutGenMask_infershape_diff_test){
  ge::op::StatelessDropOutGenMask op;
  op.UpdateInputDesc("x", create_desc({3}, ge::DT_INT32));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(statelessDropOutGenMask, statelessDropOutGenMask_infershape_prob_rank_err_1){
  ge::op::StatelessDropOutGenMask op;
  auto probDesc = op.GetInputDesc("prob");
  probDesc.SetDataType(ge::DT_FLOAT);
  probDesc.SetShape(ge::Shape({1}));
  probDesc.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("prob", probDesc);
  

  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(statelessDropOutGenMask, statelessDropOutGenMask_infershape_seed_rank_err_1){
  ge::op::StatelessDropOutGenMask op;
  auto seedDesc = op.GetInputDesc("seed");
  seedDesc.SetDataType(ge::DT_FLOAT);
  seedDesc.SetShape(ge::Shape({1}));
  seedDesc.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("seed", seedDesc);
  

  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(statelessDropOutGenMask, statelessDropOutGenMask_infershape_seed1_rank_err_1){
  ge::op::StatelessDropOutGenMask op;
  auto seed1Desc = op.GetInputDesc("seed1");
  seed1Desc.SetDataType(ge::DT_FLOAT);
  seed1Desc.SetShape(ge::Shape({1}));
  seed1Desc.SetOriginShape(ge::Shape({1}));
  op.UpdateInputDesc("seed1", seed1Desc);
  

  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(statelessDropOutGenMask, statelessDropOutGenMask_infershape_prob_rank_suc_0){
  ge::op::StatelessDropOutGenMask op;
  auto probDesc = op.GetInputDesc("prob");
  probDesc.SetDataType(ge::DT_FLOAT);
  probDesc.SetShape(ge::Shape());
  probDesc.SetOriginShape(ge::Shape());
  op.UpdateInputDesc("prob", probDesc);
  

  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(statelessDropOutGenMask, statelessDropOutGenMask_infershape_seed_rank_suc_0){
  ge::op::StatelessDropOutGenMask op;
  auto seedDesc = op.GetInputDesc("seed");
  seedDesc.SetDataType(ge::DT_FLOAT);
  seedDesc.SetShape(ge::Shape());
  seedDesc.SetOriginShape(ge::Shape());
  op.UpdateInputDesc("seed", seedDesc);
  

  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(statelessDropOutGenMask, statelessDropOutGenMask_infershape_seed1_rank_suc_0){
  ge::op::StatelessDropOutGenMask op;
  auto seed1Desc = op.GetInputDesc("seed1");
  seed1Desc.SetDataType(ge::DT_FLOAT);
  seed1Desc.SetShape(ge::Shape());
  seed1Desc.SetOriginShape(ge::Shape());
  op.UpdateInputDesc("seed1", seed1Desc);
  

  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(statelessDropOutGenMask, statelessDropOutGenMask_infershape_unconstData_test){
  ge::op::StatelessDropOutGenMask op;

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto input_sizes_desc = op_desc->MutableInputDesc("shape");
  std::vector<std::pair<int64_t, int64_t>> value_range = {{1, 2}, {32, 32}, {4, 5}, {4, 5}};
  input_sizes_desc->SetValueRange(value_range);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
