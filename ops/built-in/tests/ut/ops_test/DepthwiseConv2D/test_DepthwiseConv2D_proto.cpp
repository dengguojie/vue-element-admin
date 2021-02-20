#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_calculation_ops.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "op_desc.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

// ---------------DepthwiseConv2D-------------------
class DepthwiseConv2DProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DepthwiseConv2D Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DepthwiseConv2D Proto Test TearDown" << std::endl;
  }
};

// REG_OP(DepthwiseConv2D)
//     .INPUT(x, TensorType({DT_FLOAT16, DT_INT8}))
//     .INPUT(filter, TensorType({DT_FLOAT16, DT_INT8}))
//     .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_INT32}))
//     .OPTIONAL_INPUT(offset_w, TensorType({DT_FLOAT16, DT_INT8}))
//     .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT32}))
//     .REQUIRED_ATTR(strides, ListInt)
//     .ATTR(dilations, ListInt, {1, 1, 1, 1})
//     .REQUIRED_ATTR(pads, ListInt)
//     .ATTR(data_format, String, "NHWC")
//     .ATTR(offset_x, Int, 0)
//     .OP_END_FACTORY_REG(DepthwiseConv2D)

TEST_F(DepthwiseConv2DProtoTest, conv2dSplicDataTest) {
    ge::op::DepthwiseConv2D depthwiseconv2d;
    depthwiseconv2d.UpdateInputDesc("x", create_desc_with_ori({4, 64, 64, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4, 64, 64, 16},ge::FORMAT_NHWC));
    depthwiseconv2d.UpdateInputDesc("filter", create_desc_with_ori({3, 3, 16, 1}, ge::DT_FLOAT16, ge::FORMAT_HWCN,{3, 3, 16, 1},ge::FORMAT_HWCN));
    depthwiseconv2d.UpdateOutputDesc("y", create_desc_with_ori({4,64,64,1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,{4,64,64,1},ge::FORMAT_NHWC));
    depthwiseconv2d.SetAttr("strides", {1, 1, 1, 1});
    depthwiseconv2d.SetAttr("pads", {1, 1, 1, 1});
    depthwiseconv2d.SetAttr("dilations", {2, 2, 2, 2});
    std::vector<std::vector<int64_t>> y_data_slice ={{0,3}, {4,6}, {0,62}, {3,10}, {5,7}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(depthwiseconv2d);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    std::vector<std::vector<int64_t>> tt;
    ge::AttrUtils::GetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, tt);

    auto status = op_desc->InferDataSlice();
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::vector<int64_t>> x_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);

    std::vector<std::vector<int64_t>> expect_x_data_slice = {{0,3}, {4,6}, {0,63}, {2,13}, {5,7}};
    EXPECT_EQ(expect_x_data_slice, x_data_slice);

    std::vector<int> pads;
    depthwiseconv2d.GetAttr("pads",pads);
    std::vector<int> expect_pads = {1, 1, 0, 0};
    EXPECT_EQ(expect_pads, pads);
}