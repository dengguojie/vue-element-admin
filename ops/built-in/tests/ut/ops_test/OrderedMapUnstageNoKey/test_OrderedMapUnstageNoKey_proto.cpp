#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "data_flow_ops.h"
#include "op_proto_test_util.h"
#include "utils/op_desc_utils.h"

class ordered_map_unstage_no_key : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ordered_map_unstage_no_key Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ordered_map_unstage_no_key Proto Test TearDown" << std::endl;
  }
};

TEST_F(ordered_map_unstage_no_key,
       ordered_map_unstage_no_key_infershape_getattr_dtypes_failed) {
  ge::op::OrderedMapUnstageNoKey op;
  op.UpdateInputDesc("indices", create_desc({}, ge::DT_FLOAT16));
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  op_desc->DelAttr("dtypes");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ordered_map_unstage_no_key,
       ordered_map_unstage_no_key_infershape_failed) {
  ge::op::OrderedMapUnstageNoKey op;
  op.UpdateInputDesc("indices", create_desc({}, ge::DT_FLOAT16));

  std::vector<ge::DataType> dtypes = {ge::DT_FLOAT16};
  op.SetAttr("dtypes", dtypes);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ordered_map_unstage_no_key,
       ordered_map_unstage_no_key_infershape_success) {
  ge::op::OrderedMapUnstageNoKey op;
  op.UpdateInputDesc("indices", create_desc({}, ge::DT_FLOAT16));
  op.create_dynamic_output_values(2);
  std::vector<ge::DataType> dtypes = {ge::DT_FLOAT16};
  op.SetAttr("dtypes", dtypes);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetDynamicOutputDesc("values", 0);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);

  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  auto output_desc_key = op.GetOutputDesc("key");
  EXPECT_EQ(output_desc_key.GetDataType(), ge::DT_INT64);
}
