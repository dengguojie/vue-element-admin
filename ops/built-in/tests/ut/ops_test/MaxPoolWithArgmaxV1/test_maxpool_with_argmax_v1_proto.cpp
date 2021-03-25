#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_pooling_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class maxpool_with_argmax_v1_infer_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "maxpool_with_argmax_v1_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "maxpool_with_argmax_v1_infer_test TearDown" << std::endl;
  }
};

TEST_F(maxpool_with_argmax_v1_infer_test, maxpool_with_argmax_v1_infer_test_1) {
  auto shape_x1 = vector<int64_t>({1, 64, -1, -1});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{1, 1}, {64, 64}, {1, 672}, {1, 672}};

  // expect result
  std::vector<int64_t> expected_shape0 = {1, 64, -1, -1};
  std::vector<std::pair<int64_t,int64_t>> expected_range0 = {{1, 1}, {64, 64},{1, 671}, {1, 671}};
  std::vector<int64_t> expected_shape1 = {1, 64, -1, -1};
  std::vector<std::pair<int64_t,int64_t>> expected_range1 = {{1, 1}, {64, 64},{1, 4}, {1, 28142}};
  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                shape_x1, ge::FORMAT_NCHW, range_x1);
  
  bool ceil_mode = false;
  // new op and do infershape
  ge::op::MaxPoolWithArgmaxV1 op;
  op.UpdateInputDesc("x", tensor_desc_x1);
  op.SetAttr("ksize", {1,2,2,1});
  op.SetAttr("strides", {1,1,1,1});
  op.SetAttr("pads", {1,0,0,1});
  op.SetAttr("dilation", {1,1,1,1});
  op.SetAttr("ceil_mode", ceil_mode);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc0 = op.GetOutputDesc("y");
  auto output_desc1 = op.GetOutputDesc("argmax");
  EXPECT_EQ(output_desc0.GetShape().GetDims(), expected_shape0);
  std::vector<std::pair<int64_t,int64_t>> output_range0;
  EXPECT_EQ(output_desc0.GetShapeRange(output_range0), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range0, expected_range0);
  
  EXPECT_EQ(output_desc1.GetShape().GetDims(), expected_shape1);
  std::vector<std::pair<int64_t,int64_t>> output_range1;
  EXPECT_EQ(output_desc1.GetShapeRange(output_range1), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range1, expected_range1);
}

TEST_F(maxpool_with_argmax_v1_infer_test, maxpool_with_argmax_v1_infer_test_2) {
  auto shape_x1 = vector<int64_t>({1, 64, 672, 672});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{1, 1}, {64, 64}, {672, 672}, {672, 672}};

  // expect result
  std::vector<int64_t> expected_shape0 = {1, 64, 671, 671};
  std::vector<std::pair<int64_t,int64_t>> expected_range0 = {{1, 1}, {64, 64},{671, 671}, {671, 671}};
  std::vector<int64_t> expected_shape1 = {1, 64, 4, 28142};
  std::vector<std::pair<int64_t,int64_t>> expected_range1 = {{1, 1}, {64, 64},{4, 4}, {28142, 28142}};
  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                shape_x1, ge::FORMAT_NCHW, range_x1);
  
  bool ceil_mode = false;
  // new op and do infershape
  ge::op::MaxPoolWithArgmaxV1 op;
  op.UpdateInputDesc("x", tensor_desc_x1);
  op.SetAttr("ksize", {1,2,2,1});
  op.SetAttr("strides", {1,1,1,1});
  op.SetAttr("pads", {1,0,0,1});
  op.SetAttr("dilation", {1,1,1,1});
  op.SetAttr("ceil_mode", ceil_mode);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc0 = op.GetOutputDesc("y");
  auto output_desc1 = op.GetOutputDesc("argmax");
  EXPECT_EQ(output_desc0.GetShape().GetDims(), expected_shape0);
  std::vector<std::pair<int64_t,int64_t>> output_range0;
  EXPECT_EQ(output_desc0.GetShapeRange(output_range0), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range0, expected_range0);
  
  EXPECT_EQ(output_desc1.GetShape().GetDims(), expected_shape1);
  std::vector<std::pair<int64_t,int64_t>> output_range1;
  EXPECT_EQ(output_desc1.GetShapeRange(output_range1), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range1, expected_range1);
}