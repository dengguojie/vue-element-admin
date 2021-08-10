#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_pooling_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/attr_utils.h"
#include "graph/common_error_codes.h"

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
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, test_format, shape_x1, test_format, range_x1);

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
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, test_format, shape_x1, test_format, range_x1);

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

TEST_F(MaxPoolProto, maxpool_proto_2) {
  // set input info
  auto shape_x1 = vector<int64_t>({3, 5, 10, 10});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{3, 3}, {5, 5}, {10, 10}, {10, 10}};

  auto ksize_list = vector<int64_t>({1, 1, 2, 2});
  auto stride_list = vector<int64_t>({1, 1, 2, 2});
  auto test_format = ge::FORMAT_NCHW;
  auto pad_mode = "VALID";
  auto data_format = "NCHW";

  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, test_format, shape_x1, test_format, range_x1);

  // new op and do infershape
  ge::op::MaxPool op;
  op.UpdateInputDesc("x", tensor_desc_x1);
  op.set_attr_strides(stride_list);
  op.set_attr_padding(pad_mode);
  op.set_attr_data_format(data_format);
  auto ret = op.InferShapeAndType();
  ret = op.VerifyAllAttr(true);

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolProto, maxpool_proto_3) {
  // set input info
  auto shape_x1 = vector<int64_t>({3, 5, 10, 10});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{3, 3}, {5, 5}, {10, 10}, {10, 10}};

  auto ksize_list = vector<int64_t>({1, 1, 2, 2, 1});
  auto stride_list = vector<int64_t>({1, 1, 2, 2});
  auto test_format = ge::FORMAT_NCHW;
  auto pad_mode = "VALID";
  auto data_format = "NCHW";

  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, test_format, shape_x1, test_format, range_x1);

  // new op and do infershape
  ge::op::MaxPool op;
  op.UpdateInputDesc("x", tensor_desc_x1);
  op.set_attr_ksize(ksize_list);
  op.set_attr_strides(stride_list);
  op.set_attr_padding(pad_mode);
  op.set_attr_data_format(data_format);
  auto ret = op.InferShapeAndType();
  ret = op.VerifyAllAttr(true);

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolProto, maxpool_proto_4) {
  // set input info
  auto shape_x1 = vector<int64_t>({3, 5, 10, 10});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{3, 3}, {5, 5}, {10, 10}, {10, 10}};

  auto ksize_list = vector<int64_t>({1, 1, 2, 2});
  auto stride_list = vector<int64_t>({1, 1, 2, 2});
  auto test_format = ge::FORMAT_NCHW;
  auto pad_mode = "VALID";
  auto data_format = "NCHW";

  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, test_format, shape_x1, test_format, range_x1);

  // new op and do infershape
  ge::op::MaxPool op;
  op.UpdateInputDesc("x", tensor_desc_x1);
  op.set_attr_ksize(ksize_list);
  op.set_attr_padding(pad_mode);
  op.set_attr_data_format(data_format);
  auto ret = op.InferShapeAndType();
  ret = op.VerifyAllAttr(true);

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolProto, maxpool_proto_5) {
  // set input info
  auto shape_x1 = vector<int64_t>({3, 5, 10, 10});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{3, 3}, {5, 5}, {10, 10}, {10, 10}};

  auto ksize_list = vector<int64_t>({1, 1, 2, 2});
  auto stride_list = vector<int64_t>({1, 1, 2, 2, 1});
  auto test_format = ge::FORMAT_NCHW;
  auto pad_mode = "VALID";
  auto data_format = "NCHW";

  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, test_format, shape_x1, test_format, range_x1);

  // new op and do infershape
  ge::op::MaxPool op;
  op.UpdateInputDesc("x", tensor_desc_x1);
  op.set_attr_ksize(ksize_list);
  op.set_attr_strides(stride_list);
  op.set_attr_padding(pad_mode);
  op.set_attr_data_format(data_format);
  auto ret = op.InferShapeAndType();
  ret = op.VerifyAllAttr(true);

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolProto, maxpool_proto_6) {
  // set input info
  auto shape_x1 = vector<int64_t>({3, 5, 10, 10});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{3, 3}, {5, 5}, {10, 10}, {10, 10}};

  auto ksize_list = vector<int64_t>({1, 1, 2, 2});
  auto stride_list = vector<int64_t>({1, 1, 2, 2});
  auto test_format = ge::FORMAT_NCHW;
  auto pad_mode = "VALID";
  auto data_format = "NCHW";

  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, test_format, shape_x1, test_format, range_x1);

  // new op and do infershape
  ge::op::MaxPool op;
  op.UpdateInputDesc("x", tensor_desc_x1);
  op.set_attr_ksize(ksize_list);
  op.set_attr_strides(stride_list);
  op.set_attr_data_format(data_format);
  auto ret = op.InferShapeAndType();
  ret = op.VerifyAllAttr(true);

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolProto, maxpool_proto_7) {
  // set input info
  auto shape_x1 = vector<int64_t>({3, 5, 10, 10});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{3, 3}, {5, 5}, {10, 10}, {10, 10}};

  auto ksize_list = vector<int64_t>({1, 1, 2, 2});
  auto stride_list = vector<int64_t>({1, 1, 2, 2});
  auto test_format = ge::FORMAT_NCHW;
  auto pad_mode = "VALID";
  auto data_format = "NCHW";

  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, test_format, shape_x1, test_format, range_x1);

  // new op and do infershape
  ge::op::MaxPool op;
  op.UpdateInputDesc("x", tensor_desc_x1);
  op.set_attr_ksize(ksize_list);
  op.set_attr_strides(stride_list);
  op.set_attr_padding(pad_mode);
  auto ret = op.InferShapeAndType();
  ret = op.VerifyAllAttr(true);

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolProto, maxpool_proto_8) {
  // set input info
  auto shape_x1 = vector<int64_t>({3, 5, 10, 10});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{3, 3}, {5, 5}, {10, 10}, {10, 10}};

  auto ksize_list = vector<int64_t>({1, 1, 2, 2});
  auto stride_list = vector<int64_t>({1, 1, 2, 2});
  auto test_format = ge::FORMAT_NCHW;
  auto pad_mode = "ND";
  auto data_format = "NCHW";

  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, test_format, shape_x1, test_format, range_x1);

  // new op and do infershape
  ge::op::MaxPool op;
  op.UpdateInputDesc("x", tensor_desc_x1);
  op.set_attr_ksize(ksize_list);
  op.set_attr_strides(stride_list);
  op.set_attr_padding(pad_mode);
  op.set_attr_data_format(data_format);
  auto ret = op.InferShapeAndType();
  ret = op.VerifyAllAttr(true);

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolProto, maxpool_proto_9) {
  // set input info
  auto shape_x1 = vector<int64_t>({3, 5, 10, 10});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{3, 3}, {5, 5}, {10, 10}, {10, 10}};

  auto ksize_list = vector<int64_t>({1, 1, 2, 2});
  auto stride_list = vector<int64_t>({1, 1, 2, 2});
  auto test_format = ge::FORMAT_NCHW;
  auto pad_mode = "VALID";
  auto data_format = "ND";

  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, test_format, shape_x1, test_format, range_x1);

  // new op and do infershape
  ge::op::MaxPool op;
  op.UpdateInputDesc("x", tensor_desc_x1);
  op.set_attr_ksize(ksize_list);
  op.set_attr_strides(stride_list);
  op.set_attr_padding(pad_mode);
  op.set_attr_data_format(data_format);
  auto ret = op.InferShapeAndType();
  ret = op.VerifyAllAttr(true);

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolProto, maxpool_proto_10) {
  // set input info
  auto shape_x1 = vector<int64_t>({3, 5, 10, 10});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{3, 3}, {5, 5}, {10, 10}, {10, 10}};

  auto ksize_list = vector<int64_t>({2, 2, 2, 2});
  auto stride_list = vector<int64_t>({2, 2, 2, 2});
  auto test_format = ge::FORMAT_NCHW;
  auto pad_mode = "VALID";
  auto data_format = "NCHW";

  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, test_format, shape_x1, test_format, range_x1);

  // new op and do infershape
  ge::op::MaxPool op;
  op.UpdateInputDesc("x", tensor_desc_x1);
  op.set_attr_ksize(ksize_list);
  op.set_attr_strides(stride_list);
  op.set_attr_padding(pad_mode);
  op.set_attr_data_format(data_format);
  auto ret = op.InferShapeAndType();
  ret = op.VerifyAllAttr(true);

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
TEST_F(MaxPoolProto, maxpool_proto_11) {
  // set input info
  auto shape_x1 = vector<int64_t>({3, 5, 10, 10});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{3, 3}, {5, 5}, {10, 10}, {10, 10}};

  auto ksize_list = vector<int64_t>({2, 2, 2, 2});
  auto stride_list = vector<int64_t>({2, 2, 2, 2});
  auto test_format = ge::FORMAT_NCHW;
  auto pad_mode = "VALID";
  auto data_format = "NHWC";

  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, test_format, shape_x1, test_format, range_x1);

  // new op and do infershape
  ge::op::MaxPool op;
  op.UpdateInputDesc("x", tensor_desc_x1);
  op.set_attr_ksize(ksize_list);
  op.set_attr_strides(stride_list);
  op.set_attr_padding(pad_mode);
  op.set_attr_data_format(data_format);
  auto ret = op.InferShapeAndType();
  ret = op.VerifyAllAttr(true);

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolProto, maxpool_proto_12) {
  // set input info
  auto shape_x1 = vector<int64_t>({-2});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{1, -1}, {1, -1}, {1, -1}, {1, -1}};

  auto ksize_list = vector<int64_t>({1, 3, 2, 1});
  auto stride_list = vector<int64_t>({1, 2, 3, 1});
  auto test_format = ge::FORMAT_NHWC;
  auto pad_mode = "SAME";
  auto data_format = "NHWC";

  // expect result
  std::vector<int64_t> expected_shape = {-1, -1, -1, -1};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{1, -1}, {1, -1}, {1, -1}, {1, -1}};

  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, test_format, shape_x1, test_format, range_x1);

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

TEST_F(MaxPoolProto, maxpool_proto_13) {
  // set input info
  auto shape_x1 = vector<int64_t>({3, 5, 10, 10});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{3, 3}, {5, 5}, {10, 10}, {10, 10}};

  auto ksize_list = vector<int64_t>({1, 1, -1, -1});
  auto stride_list = vector<int64_t>({1, 1, 1, 1});
  auto test_format = ge::FORMAT_NCHW;
  auto pad_mode = "VALID";
  auto data_format = "NCHW";

  // expect result
  std::vector<int64_t> expected_shape = {3, 5, 1, 1};

  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, test_format, shape_x1, test_format, range_x1);

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
  std::vector<int32_t> process_ksize;
  EXPECT_EQ(op.GetAttr("ksize", process_ksize), ge::GRAPH_SUCCESS);
  EXPECT_EQ(process_ksize[2], 10);
  EXPECT_EQ(process_ksize[3], 10);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}

TEST_F(MaxPoolProto, MaxPool_data_slice_infer1) {
  ge::op::MaxPool op;

  // set MaxPool attr
  op.SetAttr("data_format", "NHWC");
  op.SetAttr("padding", "VALID");
  std::vector<int64_t> window = {0, 3, 3, 0};
  op.SetAttr("ksize", window);
  std::vector<int64_t> stride = {0, 2, 2, 0};
  op.SetAttr("strides", stride);

  auto tensor_desc = create_desc_with_ori({1,1,224,224,16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1,3,224,224}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);

  auto output_tensor_desc_temp = create_desc_with_ori({1,1,115,115,16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1,3,115,115}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("y", output_tensor_desc_temp);

  std::vector<std::vector<int64_t>> output_data_slice ={{}, {}, {10, 20}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(output_tensor_desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x = op_desc->MutableInputDesc("x");
  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);

  std::vector<std::vector<int64_t>> expected_x_data_slice = {{}, {}, {20, 42}, {}, {}};
  EXPECT_EQ(expected_x_data_slice, x_data_slice);
}

TEST_F(MaxPoolProto, MaxPool_data_slice_infer2) {
  ge::op::MaxPool op;

  // set MaxPool attr
  op.SetAttr("data_format", "NCHW");
  op.SetAttr("padding", "VALID");
  std::vector<int64_t> window = {0, 0, 3, 3};
  op.SetAttr("ksize", window);
  std::vector<int64_t> stride = {0, 0, 2, 2};
  op.SetAttr("strides", stride);

  auto tensor_desc = create_desc_with_ori({1,3,224,224}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1,3,224,224}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);

  auto output_tensor_desc_temp = create_desc_with_ori({1,3,115,115}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1,3,115,115}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("y", output_tensor_desc_temp);

  std::vector<std::vector<int64_t>> output_data_slice ={{}, {}, {10, 20}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(output_tensor_desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolProto, MaxPool_data_slice_infer3) {
  ge::op::MaxPool op;

  // set MaxPool attr
  op.SetAttr("data_format", "NCHW");
  op.SetAttr("padding", "VALID");
  std::vector<int64_t> window = {0, 0, 3, 3};
  op.SetAttr("ksize", window);
  std::vector<int64_t> stride = {0, 0, 2, 2};
  op.SetAttr("strides", stride);

  auto tensor_desc = create_desc_with_ori({1,1,224,224,16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1,3,224,224}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);

  auto output_tensor_desc_temp = create_desc_with_ori({1,1,115,115,16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1,3,115,115}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("y", output_tensor_desc_temp);

  std::vector<std::vector<int64_t>> output_data_slice ={{}, {}, {10, 20}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(output_tensor_desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x = op_desc->MutableInputDesc("x");
  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);

  std::vector<std::vector<int64_t>> expected_x_data_slice = {{}, {}, {20, 42}, {}, {}};
  EXPECT_EQ(expected_x_data_slice, x_data_slice);
}

TEST_F(MaxPoolProto, MaxPool_data_slice_infer4) {
  ge::op::MaxPool op;

  // set MaxPool attr
  op.SetAttr("data_format", "NZ");
  op.SetAttr("padding", "VALID");
  std::vector<int64_t> window = {0, 0, 3, 3};
  op.SetAttr("ksize", window);
  std::vector<int64_t> stride = {0, 0, 2, 2};
  op.SetAttr("strides", stride);

  auto tensor_desc = create_desc_with_ori({1,1,224,224,16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1,3,224,224}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);

  auto output_tensor_desc_temp = create_desc_with_ori({1,1,115,115,16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1,3,115,115}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("y", output_tensor_desc_temp);

  std::vector<std::vector<int64_t>> output_data_slice ={{}, {}, {10, 20}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(output_tensor_desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolProto, MaxPool_data_slice_infer5) {
  ge::op::MaxPool op;

  // set MaxPool attr
  op.SetAttr("data_format", "NCHW");
  op.SetAttr("padding", "SAME");
  std::vector<int64_t> window = {0, 0, 3, 3};
  op.SetAttr("ksize", window);
  std::vector<int64_t> stride = {0, 0, 2, 2};
  op.SetAttr("strides", stride);

  auto tensor_desc = create_desc_with_ori({1,1,224,224,16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1,3,224,224}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("x", tensor_desc);

  auto output_tensor_desc_temp = create_desc_with_ori({1,1,115,115,16}, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0, {1,3,115,115}, ge::FORMAT_NCHW);
  op.UpdateInputDesc("y", output_tensor_desc_temp);

  std::vector<std::vector<int64_t>> output_data_slice ={{}, {}, {10, 20}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(output_tensor_desc, ge::ATTR_NAME_DATA_SLICE, output_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::NO_OVERLAP_DIM);
}