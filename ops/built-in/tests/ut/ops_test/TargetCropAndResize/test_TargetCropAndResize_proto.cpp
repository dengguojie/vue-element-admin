#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "target_crop_and_resize.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class target_crop_and_resize_infer_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "target_crop_and_resize_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "target_crop_and_resize_infer_test TearDown" << std::endl;
  }
};

//REG_OP(TargetCropAndResize)
//    .INPUT(x, TensorType({DT_UINT8}))
//    .INPUT(boxes, TensorType({DT_INT32}))
//    .INPUT(box_index, TensorType({DT_INT32}))
//    .OUTPUT(y, TensorType({DT_UINT8}))
//    .ATTR(output_h, Int, 224)
//    .ATTR(output_w, Int, 224)
//    .ATTR(input_format, String, "YUV420SP_U8")
//    .OP_END_FACTORY_REG(TargetCropAndResize)
//}

TEST_F(target_crop_and_resize_infer_test, target_crop_and_resize_infer_test_1) {
  // expect result
  std::vector<int64_t> expected_shape = {5,4,100,120};

  // new op and do infershape
  ge::op::TargetCropAndResize op;
  op.UpdateInputDesc("x", create_desc_with_ori({1,3,224,224}, ge::DT_UINT8, ge::FORMAT_NCHW, {1,3,224,224}, ge::FORMAT_NCHW));
  op.UpdateInputDesc("boxes", create_desc_with_ori({5,4}, ge::DT_INT32, ge::FORMAT_ND, {5,4}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori({5}, ge::DT_INT32, ge::FORMAT_ND, {5}, ge::FORMAT_ND));
  op.set_attr_output_h(100);
  op.set_attr_output_w(120);
  op.set_attr_input_format("YUV420SP_U8");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(target_crop_and_resize_infer_test, target_crop_and_resize_infer_test_2) {
  // new op and do infershape
  ge::op::TargetCropAndResize op;
  op.UpdateInputDesc("x", create_desc_with_ori({1,3,224}, ge::DT_UINT8, ge::FORMAT_NCHW, {1,3,224}, ge::FORMAT_NCHW));
  op.UpdateInputDesc("boxes", create_desc_with_ori({5,4}, ge::DT_INT32, ge::FORMAT_ND, {5,4}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori({5}, ge::DT_INT32, ge::FORMAT_ND, {5}, ge::FORMAT_ND));
  op.set_attr_output_h(100);
  op.set_attr_output_w(120);
  op.set_attr_input_format("YUV420SP_U8");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(target_crop_and_resize_infer_test, target_crop_and_resize_infer_test_3) {
  // new op and do infershape
  ge::op::TargetCropAndResize op;
  op.UpdateInputDesc("x", create_desc_with_ori({1,3,224,224}, ge::DT_UINT8, ge::FORMAT_NCHW, {1,3,224,224}, ge::FORMAT_NCHW));
  op.UpdateInputDesc("boxes", create_desc_with_ori({5}, ge::DT_INT32, ge::FORMAT_ND, {5}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori({5}, ge::DT_INT32, ge::FORMAT_ND, {5}, ge::FORMAT_ND));
  op.set_attr_output_h(100);
  op.set_attr_output_w(120);
  op.set_attr_input_format("YUV420SP_U8");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(target_crop_and_resize_infer_test, InfershapeTargetCropAndResize_001) {
  ge::op::TargetCropAndResize op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 3, 224, 224}, ge::DT_UINT8, ge::FORMAT_NHWC, {1, 3, 224, 224}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("boxes", create_desc_with_ori({5, 4}, ge::DT_INT32, ge::FORMAT_ND, {5, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori({5}, ge::DT_INT32, ge::FORMAT_ND, {5}, ge::FORMAT_ND));
  op.set_attr_output_h(100);
  op.set_attr_output_w(120);
  op.set_attr_input_format("YUV420SP_U8");

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  std::vector<int64_t> expect_output_shape = {5, 224, 100, 120};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expect_output_shape);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_UINT8);
}

TEST_F(target_crop_and_resize_infer_test, InfershapeTargetCropAndResize_002) {
  ge::op::TargetCropAndResize op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 3, 224, 224}, ge::DT_UINT8, ge::FORMAT_NHWC, {1, 3, 224, 224}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("boxes", create_desc_with_ori({5, 4}, ge::DT_INT32, ge::FORMAT_ND, {5, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori({5}, ge::DT_INT32, ge::FORMAT_ND, {5}, ge::FORMAT_ND));
  op.SetAttr("output_h", true);
  op.set_attr_input_format("YUV420SP_U8");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(target_crop_and_resize_infer_test, InfershapeTargetCropAndResize_003) {
  ge::op::TargetCropAndResize op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({1, 3, 224, 224}, ge::DT_UINT8, ge::FORMAT_NHWC, {1, 3, 224, 224}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("boxes", create_desc_with_ori({5, 4}, ge::DT_INT32, ge::FORMAT_ND, {5, 4}, ge::FORMAT_ND));
  op.UpdateInputDesc("box_index", create_desc_with_ori({5}, ge::DT_INT32, ge::FORMAT_ND, {5}, ge::FORMAT_ND));
  op.SetAttr("output_h", 220);
  op.SetAttr("output_w", true);
  op.set_attr_input_format("YUV420SP_U8");

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}