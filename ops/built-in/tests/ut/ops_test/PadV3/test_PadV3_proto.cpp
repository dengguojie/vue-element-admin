#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "pad_ops.h"

using namespace ge;
using namespace op;

class pad_v3_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "pad_v3_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "pad_v3_test TearDown" << std::endl;
    }
};


TEST_F(pad_v3_test, pad_v3_infer_shape_01) {
  ge::op::PadV3 op;
  std::cout<< "pad_v3 test_1!!!"<<std::endl;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1,-1},{1,-1},{1,-1},{1,-1}};
  auto tensor_desc = create_desc_shape_range({-1,64,-1,20}, ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {-1,64,-1,20}, ge::FORMAT_ND, shape_range);

  auto paddings_desc = create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND,
                                               {-1}, ge::FORMAT_ND, {{1,-1}});

  auto constant_values_desc = create_desc_shape_range({1}, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                      {1}, ge::FORMAT_ND, {{1,1}});
  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("paddings", paddings_desc);
  op.UpdateInputDesc("constant_values", constant_values_desc);
  op.SetAttr("mode", "constant");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1,-1,-1,-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
}

TEST_F(pad_v3_test, pad_v3_infer_shape_02) {
  ge::op::PadV3 op;
  std::cout<< "pad_v3 test_1!!!"<<std::endl;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1,-1},{1,-1},{1,-1},{1,-1}};
  auto tensor_desc = create_desc_shape_range({-1,20,-1,-1}, ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {-1,20,-1,-1}, ge::FORMAT_ND, shape_range);

  auto paddings_desc = create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND,
                                               {-1}, ge::FORMAT_ND, {{1,-1}});

  auto constant_values_desc = create_desc_shape_range({1}, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                      {1}, ge::FORMAT_ND, {{1,1}});
  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("paddings", paddings_desc);
  op.UpdateInputDesc("constant_values", constant_values_desc);
  op.SetAttr("mode", "constant");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1,-1,-1,-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
}
