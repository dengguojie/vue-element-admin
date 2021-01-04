#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_pooling_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class MaxPoolProto : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "MaxPoolProto SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MaxPoolProto TearDown" << std::endl;
  }
};

TEST_F(MaxPoolProto, maxpool_proto_0) {
  // set input info
  auto shape_x1 = vector<int64_t>({3, -1, -1, 5});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{3, 3}, {12, 51}, {17, 60}, {5, 5}};
  
  auto ksize_list = vector<int64_t>({1, 3, 2, 1});
  auto stride_list = vector<int64_t>({1, 2, 3, 1});
  auto test_format = ge::FORMAT_NHWC;
  auto pad_mode = "SAME";
  auto data_format = "NHWC";

  // expect result
  std::vector<int64_t> expected_shape = {3, -1, -1, 5};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{3, 3}, {6, 26}, {6, 20}, {5, 5}};

  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, test_format,
                                                shape_x1, test_format, range_x1);

  // new op and do infershape
  ge::op::MaxPool op;
  op.UpdateInputDesc("x", tensor_desc_x1);
  op.set_attr_ksize(ksize_list);
  op.set_attr_strides(stride_list);
  op.set_attr_padding(pad_mode);
  op.set_attr_data_format(data_format);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

TEST_F(MaxPoolProto, maxpool_proto_1) {
  // set input info
  auto shape_x1 = vector<int64_t>({3, 5, 10, 10});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{3, 3}, {5, 5}, {10, 10}, {10, 10}};

  auto ksize_list = vector<int64_t>({1, 1, 2, 2});
  auto stride_list = vector<int64_t>({1, 1, 2, 2});
  auto test_format = ge::FORMAT_NCHW;
  auto pad_mode = "VALID";
  auto data_format = "NCHW";

  // expect result
  std::vector<int64_t> expected_shape = {3, 5, 5, 5};

  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, test_format,
  shape_x1, test_format, range_x1);

  // new op and do infershape
  ge::op::MaxPool op;
  op.UpdateInputDesc("x", tensor_desc_x1);
  op.set_attr_ksize(ksize_list);
  op.set_attr_strides(stride_list);
  op.set_attr_padding(pad_mode);
  op.set_attr_data_format(data_format);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}
