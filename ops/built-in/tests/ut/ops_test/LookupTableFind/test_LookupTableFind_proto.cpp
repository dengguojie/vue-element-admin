#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "lookup_ops.h"

class lookup_table_find : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "lookup_table_find Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "lookup_table_find Proto Test TearDown" << std::endl;
  }
};

TEST_F(lookup_table_find, lookup_table_find_infershape_diff_test){
  ge::op::LookupTableFind op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_FLOAT16));
  op.UpdateInputDesc("keys", create_desc({3, 4, 5}, ge::DT_FLOAT16));
  op.UpdateInputDesc("default_value", create_desc({1}, ge::DT_FLOAT16));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("values");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}