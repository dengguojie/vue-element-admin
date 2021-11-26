#include <gtest/gtest.h>

#include <iostream>
#include <vector>
#define private public
#define protected public
#include "data_flow_ops.h"
#include "op_proto_test_util.h"
#include "utils/op_desc_utils.h"
#undef protected
#undef private

class ordered_map_unstage : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ordered_map_unstage Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ordered_map_unstage Proto Test TearDown" << std::endl;
  }
};

TEST_F(ordered_map_unstage,
       ordered_map_unstage_infershape_getattr_dtypes_failed) {
  ge::op::OrderedMapUnstage op;
  op.UpdateInputDesc("key", create_desc({}, ge::DT_FLOAT16));
  op.UpdateInputDesc("indices", create_desc({}, ge::DT_FLOAT16));
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  op_desc->DelAttr("dtypes");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ordered_map_unstage, ordered_map_unstage_infershape_failed) {
  ge::op::OrderedMapUnstage op;
  op.UpdateInputDesc("key", create_desc({}, ge::DT_FLOAT16));
  op.UpdateInputDesc("indices", create_desc({}, ge::DT_FLOAT16));
  std::vector<ge::DataType> dtypes = {ge::DT_FLOAT16};
  op.SetAttr("dtypes", dtypes);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ordered_map_unstage, ordered_map_unstage_infershape_success) {
  ge::op::OrderedMapUnstage op;
  op.UpdateInputDesc("key", create_desc({}, ge::DT_FLOAT16));
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
}
