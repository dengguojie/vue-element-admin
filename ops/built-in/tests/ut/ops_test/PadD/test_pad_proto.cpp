#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "pad_ops.h"

using namespace ge;
using namespace op;

class pad_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "pad_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "pad_test TearDown" << std::endl;
    }
};


TEST_F(pad_test, pad_infer_shape_01) {
  ge::op::Pad op;
  std::cout<< "pad test_1!!!"<<std::endl;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{3,3},{5,5},{10,10},{5,5},{2,2}};
  auto tensor_desc = create_desc_shape_range({3,5,10,5,2},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {3,5,10,5,2},
                                             ge::FORMAT_ND, shape_range);

  auto paddings_desc = create_desc_shape_range({-2},
                                           ge::DT_INT32, ge::FORMAT_ND,
                                           {-2},
                                           ge::FORMAT_ND, {});

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("paddings", paddings_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1,-1,-1,-1,-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{0,-1},{0,-1},{0,-1},{0,-1},{0,-1}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(pad_test, pad_infer_shape_02) {
  ge::op::Pad op;
  std::cout<< "pad test_2!!!"<<std::endl;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {};
  auto tensor_desc = create_desc_shape_range({-2},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {-2},
                                             ge::FORMAT_ND, shape_range);

  auto axes_desc = create_desc_shape_range({-2},
                                           ge::DT_INT32, ge::FORMAT_ND,
                                           {-2},
                                           ge::FORMAT_ND, shape_range);

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("paddings", axes_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(pad_test, pad_infer_shape_03) {
  ge::op::Pad op;
  std::cout<< "pad test_3!!!"<<std::endl;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{3,3},{2,3},{2,3},{1,1}};
  auto tensor_desc = create_desc_shape_range({3,-1,-1, 1},
                                             ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                             {3,-1,-1, 1},
                                             ge::FORMAT_NCHW, shape_range);

  auto paddings_desc = create_desc_shape_range({-1,-1},
                                           ge::DT_INT32, ge::FORMAT_ND,
                                           {-1,-1},
                                           ge::FORMAT_ND, {{1,80},{1,80}});

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("paddings", paddings_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1,-1,-1,-1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{0,-1},{0,-1},{0,-1},{0,-1}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}