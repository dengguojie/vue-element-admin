#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "image_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class sync_resize_bilinear_v2_grad_infer_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "sync_resize_bilinear_v2_grad_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "sync_resize_bilinear_v2_grad_infer_test TearDown" << std::endl;
  }
};

TEST_F(sync_resize_bilinear_v2_grad_infer_test, sync_resize_bilinear_v2_grad_infer_test_1) {
  // input grads shape {-1, 5}
  // input grads range {{1, 60}, {5, 5}}
  // input original_image shape {6, -1}
  // input original_image range {{6, 6}, {5, 90}}
  // attr  NA
  // output shape  {5, 6}
  // output shape {{5, 5}, {6, 6}}
  // set input info
  auto shape_x1 = vector<int64_t>({3, -1, -1, 5});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{3, 3}, {2, 3}, {2, 3}, {5, 5}};
  auto shape_x2 = vector<int64_t>({3, 9, 9, 5});
  std::vector<std::pair<int64_t,int64_t>> range_x2 = {};

  // expect result
  std::vector<int64_t> expected_shape = {3, 9, 9, 5};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{3, 3}, {9, 9},{9, 9}, {5, 5}};
  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT, ge::FORMAT_ND,
                                                shape_x1, ge::FORMAT_NHWC, range_x1);
  auto tensor_desc_x2 = create_desc_shape_range(shape_x2, ge::DT_FLOAT, ge::FORMAT_ND,
                                                shape_x2, ge::FORMAT_NHWC, range_x2);
  // new op and do infershape
  ge::op::SyncResizeBilinearV2Grad op;
  op.UpdateInputDesc("grads", tensor_desc_x1);
  op.UpdateInputDesc("original_image", tensor_desc_x2);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

TEST_F(sync_resize_bilinear_v2_grad_infer_test, sync_resize_bilinear_v2_grad_infer_test_3) {
  // input grads shape {-1, 5}
  // input grads range {{1, 60}, {5, 5}}
  // input original_image shape {6, -1}
  // input original_image range {{6, 6}, {5, 90}}
  // attr  NA
  // output shape  {5, 6}
  // output shape {{5, 5}, {6, 6}}
  // set input info
  auto shape_x1 = vector<int64_t>({3, 5, -1, -1});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{3, 3}, {5, 5}, {2, 3}, {2, 3}};
  auto shape_x2 = vector<int64_t>({-1, -1, -1, -1});
  std::vector<std::pair<int64_t,int64_t>> range_x2 = {{3, 8}, {3, 83}, {1, -1}, {2, 20}};

  // expect result
  std::vector<int64_t> expected_shape = {3, 5, -1, -1};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{3, 3},{5, 5}, {1, -1},{2, 20}};
  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT, ge::FORMAT_ND,
                                                shape_x1, ge::FORMAT_NCHW, range_x1);
  auto tensor_desc_x2 = create_desc_shape_range(shape_x2, ge::DT_FLOAT, ge::FORMAT_ND,
                                                shape_x2, ge::FORMAT_NCHW, range_x2);
  // new op and do infershape
  ge::op::SyncResizeBilinearV2Grad op;
  op.UpdateInputDesc("grads", tensor_desc_x1);
  op.UpdateInputDesc("original_image", tensor_desc_x2);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}
