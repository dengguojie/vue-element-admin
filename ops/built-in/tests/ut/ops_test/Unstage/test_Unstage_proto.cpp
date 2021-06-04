#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"
#include "inference_context.h"

class unstage : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "unstage Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "unstage Proto Test TearDown" << std::endl;
  }
};

TEST_F(unstage, unstage_infershape_success){
  ge::op::Unstage op;
  op.create_dynamic_output_y(1);
  std::vector<ge::DataType> dtypes = {ge::DT_FLOAT16};
  op.SetAttr("dtypes", dtypes);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}