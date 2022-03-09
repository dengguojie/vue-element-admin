#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "selection_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"

using namespace ut_util;
using namespace ge;

class onehotd : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "onehot SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "onehot TearDown" << std::endl;
  }
};

TEST_F(onehotd, onehotd_infershape1) {
  ge::op::OneHotD op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 16},{1, 16}};
  std::vector<std::pair<int64_t,int64_t>> shape_range1= {{1, 1},{1, 1}};

  auto tensor_desc = create_desc_shape_range({2, 2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {2, 2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc1 = create_desc_shape_range({1,},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {1,},
                                             ge::FORMAT_ND, shape_range1);
  

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("on_value", tensor_desc1);
  op.UpdateInputDesc("off_value", tensor_desc1);
  int depth =10;
  int axis=-1;

  op.SetAttr("depth", depth);
  op.SetAttr("axis", axis);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

  std::vector<int64_t> expected_output_shape = {2,2,10};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(onehotd, onehotd_infershape2) {
  ge::op::OneHotD op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 16},{1, 16}};
  std::vector<std::pair<int64_t,int64_t>> shape_range1= {{1, 1},{1, 1}};

  auto tensor_desc = create_desc_shape_range({2, 2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {2, 2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc1 = create_desc_shape_range({1,},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {1,},
                                             ge::FORMAT_ND, shape_range1);
  

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("on_value", tensor_desc1);
  op.UpdateInputDesc("off_value", tensor_desc1);
  int depth =10;
  int axis=-2;

  op.SetAttr("depth", depth);
  op.SetAttr("axis", axis);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
TEST_F(onehotd, onehotd_infershape3) {
  ge::op::OneHotD op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 16},{1, 16}};
  std::vector<std::pair<int64_t,int64_t>> shape_range1= {{1, 1},{1, 1}};

  auto tensor_desc = create_desc_shape_range({2, 2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {2, 2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc1 = create_desc_shape_range({1,},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {1,},
                                             ge::FORMAT_ND, shape_range1);
  

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("on_value", tensor_desc1);
  op.UpdateInputDesc("off_value", tensor_desc1);
  int depth =10;
  int axis=4;

  op.SetAttr("depth", depth);
  op.SetAttr("axis", axis);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(onehotd, onehot_infershape1) {
  ge::op::OneHot op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 16},{1, 16}};
  std::vector<std::pair<int64_t,int64_t>> shape_range1= {{1, 1},{1, 1}};

  auto tensor_desc = create_desc_shape_range({2, 2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {2, 2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc1 = create_desc_shape_range({1,},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {1,},
                                             ge::FORMAT_ND, shape_range1);
  

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("on_value", tensor_desc1);
  op.UpdateInputDesc("off_value", tensor_desc1);
  int depth =10;
  int axis=-1;

  vector<int32_t> value = {depth};
  vector<int64_t> depth_shape = {1};
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(op, depth, depth_shape, ge::DT_INT32, FORMAT_ND, value);
  op.SetAttr("axis", axis);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);

  std::vector<int64_t> expected_output_shape = {2,2,10};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(onehotd, onehot_infershape2) {
  ge::op::OneHot op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 16},{1, 16}};
  std::vector<std::pair<int64_t,int64_t>> shape_range1= {{1, 1},{1, 1}};

  auto tensor_desc = create_desc_shape_range({2, 2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {2, 2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc1 = create_desc_shape_range({1,},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {1,},
                                             ge::FORMAT_ND, shape_range1);
  

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("on_value", tensor_desc1);
  op.UpdateInputDesc("off_value", tensor_desc1);
  int depth =10;
  int axis=-2;

  vector<int32_t> value = {depth};
  vector<int64_t> depth_shape = {1};
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(op, depth, depth_shape, ge::DT_INT32, FORMAT_ND, value);
  op.SetAttr("axis", axis);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
TEST_F(onehotd, onehot_infershape3) {
  ge::op::OneHot op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 16},{1, 16}};
  std::vector<std::pair<int64_t,int64_t>> shape_range1= {{1, 1},{1, 1}};

  auto tensor_desc = create_desc_shape_range({2, 2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {2, 2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc1 = create_desc_shape_range({1,},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {1,},
                                             ge::FORMAT_ND, shape_range1);
  

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("on_value", tensor_desc1);
  op.UpdateInputDesc("off_value", tensor_desc1);
  int depth =10;
  int axis=4;

  vector<int32_t> value = {depth};
  vector<int64_t> depth_shape = {1};
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(op, depth, depth_shape, ge::DT_INT32, FORMAT_ND, value);
  op.SetAttr("axis", axis);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
TEST_F(onehotd, onehot_infershape4) {
  ge::op::OneHot op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, 16},{1, 16}};
  std::vector<std::pair<int64_t,int64_t>> shape_range1= {{1, 1},{1, 1}};

  auto tensor_desc = create_desc_shape_range({2, 2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {2, 2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc1 = create_desc_shape_range({1,},
                                             ge::DT_FLOAT, ge::FORMAT_ND,
                                             {1,},
                                             ge::FORMAT_ND, shape_range1);
  

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("on_value", tensor_desc1);
  op.UpdateInputDesc("off_value", tensor_desc1);
  int depth =10;
  int axis=2;

  vector<int32_t> value = {depth};
  vector<int64_t> depth_shape = {1};
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(op, depth, depth_shape, ge::DT_INT32, FORMAT_ND, value);
  op.SetAttr("axis", axis);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
