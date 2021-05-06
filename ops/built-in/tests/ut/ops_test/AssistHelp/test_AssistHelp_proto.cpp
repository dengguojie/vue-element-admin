#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "internal_ops.h"
#include "inference_context.h"

class assistHelp : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "assistHelp Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "assistHelp Proto Test TearDown" << std::endl;
  }
};

TEST_F(assistHelp, assistHelp_infershape_input0_attr_func_name_failed){
  ge::op::AssistHelp op;
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(assistHelp, assistHelp_infershape_input0_func_name_value_failed){
  ge::op::AssistHelp op;
  op.SetAttr("func_name", "invalid");
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

