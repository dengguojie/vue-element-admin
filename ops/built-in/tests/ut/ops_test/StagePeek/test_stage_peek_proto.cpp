#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"

class StagePeek : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StagePeek SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StagePeek TearDown" << std::endl;
  }
};

TEST_F(StagePeek, StagePeek_infershape_test01) {
  ge::op::StagePeek op;
  op.UpdateInputDesc("index", create_desc({0}, ge::DT_INT32));
  op.create_dynamic_output_y(2);
  std::vector<ge::DataType> dtypes{ge::DT_INT32, ge::DT_INT32};
  op.SetAttr("dtypes", dtypes);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc0 = op.GetDynamicOutputDesc("y", 0);
  EXPECT_EQ(output_desc0.GetDataType(), ge::DT_INT32);
  auto output_desc1 = op.GetDynamicOutputDesc("y", 1);
  EXPECT_EQ(output_desc1.GetDataType(), ge::DT_INT32);
}