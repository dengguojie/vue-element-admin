#include <iostream>

#include "common/util/error_manager/error_manager.h"
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_calculation_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"
#include "op_log.h"
#include "op_desc.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

// ---------------Conv2DBackpropFilter-------------------
class Conv2DBackpropFilterProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv2DBackpropFilter Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv2DBackpropFilter Proto Test TearDown" << std::endl;
  }
};

// base ut
TEST_F(Conv2DBackpropFilterProtoTest, Conv2DBackpropFilterVerifyBaseTest) {
    ge::op::Conv2DBackpropFilter op;
    op.UpdateInputDesc("x", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format","NCHW");
    std::string padding = "SAME";
    op.SetAttr("padding", padding);

    std::vector<int64_t> dims_filter_size{512, 256, 1, 1};
    ge::Tensor constTensor;
    ge::TensorDesc tensor_desc_filter_size(ge::Shape(),
      ge::FORMAT_NCHW, ge::DT_INT32);
    int element_size = dims_filter_size.size();
    tensor_desc_filter_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_filter_size);

    int *conv_filter_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_filter_size_tensor_value + i) = dims_filter_size[i];
    }
    constTensor.SetData((uint8_t *) conv_filter_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("filter_size").set_attr_value(constTensor);
    op.set_input_filter_size(const0);
    delete[] conv_filter_size_tensor_value;
    op.UpdateInputDesc("filter_size", tensor_desc_filter_size);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// check fm type diff x type
TEST_F(Conv2DBackpropFilterProtoTest, Conv2DBackpropFilterVerifyXDtypeTest) {
    ge::op::Conv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_INT32, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// check x out of size 4
TEST_F(Conv2DBackpropFilterProtoTest, Conv2DBackpropFilterVerifyXDimTest1) {
    ge::op::Conv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x", create_desc({512, 256, 1, 1, 1}, ge::DT_FLOAT16));
    op.UpdateOutputDesc("y", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.SetAttr("filter_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// check x less then size 4
TEST_F(Conv2DBackpropFilterProtoTest, Conv2DBackpropFilterVerifyXDimTest2) {
    ge::op::Conv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop", create_desc({128, 512, 7, 7}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x", create_desc({512, 256, 1}, ge::DT_FLOAT16));
    op.UpdateOutputDesc("y", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.SetAttr("filter_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// check out_backprop out of size 4
TEST_F(Conv2DBackpropFilterProtoTest, Conv2DBackpropFilterVerifyOutBackpropDimTest) {
    ge::op::Conv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop", create_desc({128, 512, 7, 7, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x", create_desc({512, 256, 1, 1}, ge::DT_FLOAT16));
    op.SetAttr("filter_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// check filter_size out of size 4
TEST_F(Conv2DBackpropFilterProtoTest, Conv2DBackpropFilterVerifyFilterSizeDimTest) {
    ge::op::Conv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("x", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({128, 256, 14, 14, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14, 1}, ge::FORMAT_NCHW));

    std::vector<int64_t> dims_filter_size{128, 256, 14, 14, 1};
    ge::Tensor constTensor;
    ge::TensorDesc tensor_desc_filter_size(ge::Shape(),
      ge::FORMAT_NCHW, ge::DT_INT32);
    int element_size = dims_filter_size.size();
    tensor_desc_filter_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_filter_size);

    int *conv_filter_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_filter_size_tensor_value + i) = dims_filter_size[i];
    }
    constTensor.SetData((uint8_t *) conv_filter_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("filter_size").set_attr_value(constTensor);
    op.set_input_filter_size(const0);
    delete[] conv_filter_size_tensor_value;
    op.UpdateInputDesc("filter_size", tensor_desc_filter_size);

    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format","NCHW");

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// no stride
TEST_F(Conv2DBackpropFilterProtoTest, Conv2DBackpropFilterVerifyStrideTest1) {
    ge::op::Conv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("x", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.SetAttr("filter_size", {128, 256, 14, 14});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// stride out of size 4
TEST_F(Conv2DBackpropFilterProtoTest, Conv2DBackpropFilterVerifyStrideTest2) {
    ge::op::Conv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("x", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.SetAttr("filter_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2, 1});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// dilations out of size 4
TEST_F(Conv2DBackpropFilterProtoTest, Conv2DBackpropFilterVerifyDilationsTest) {
    ge::op::Conv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("x", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.SetAttr("filter_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// no pad
TEST_F(Conv2DBackpropFilterProtoTest, Conv2DBackpropFilterVerifyPadsTest1) {
    ge::op::Conv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("x", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.SetAttr("filter_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// pad size 3
TEST_F(Conv2DBackpropFilterProtoTest, Conv2DBackpropFilterVerifyPadsTest2) {
    ge::op::Conv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("x", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.SetAttr("filter_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// pad out of -1
TEST_F(Conv2DBackpropFilterProtoTest, Conv2DBackpropFilterVerifyPadsTest3) {
    ge::op::Conv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("x", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.SetAttr("filter_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, -1});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", true);
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// dynamic c
TEST_F(Conv2DBackpropFilterProtoTest, Conv2DBackpropFilterDynamicCTest) {
    ge::op::Conv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({5,-1,60,50},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {5,-1,60,50},
                                               ge::FORMAT_NCHW,
                                               {{5, 5}, {10, 10}, {60, 60}, {50, 60}}));
    op.UpdateInputDesc("x",
                       create_desc_shape_range({5,-1,120,100},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {5,-1,120,100},
                                               ge::FORMAT_NCHW,
                                               {{5, 5}, {60, 64}, {120, 120}, {100, 100}}));
    op.UpdateOutputDesc("y", create_desc_with_ori({10, 64, 7, 6},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {10, 64, 7, 6}, ge::FORMAT_NCHW));
    auto filter_ori_shape_data = ge::op::Data("filter_size");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    filter_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    filter_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_filter_size(filter_ori_shape_data);
    op.UpdateInputDesc("filter_size", ori_tensorDesc);

    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {2, 3, 2, 2});
    op.SetAttr("padding", "SAME");
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dynamic nwc
TEST_F(Conv2DBackpropFilterProtoTest, Conv2DBackpropFilterDynamicNWCTest) {
    ge::op::Conv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop",
                       create_desc_shape_range({-1,-1,60,-1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1,-1,60,-1},
                                               ge::FORMAT_NCHW,
                                               {{5, 15}, {10, 20}, {60, 60}, {50, 60}}));
    op.UpdateInputDesc("x",
                       create_desc_shape_range({-1,-1,120,-1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1,-1,120,-1},
                                               ge::FORMAT_NCHW,
                                               {{5, 15}, {60, 64}, {100, 120}, {100, 120}}));
    op.UpdateOutputDesc("y", create_desc_with_ori({10, 64, 7, 6},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {10, 64, 7, 6}, ge::FORMAT_NCHW));
    std::vector<int64_t> dims_filter_size{10, 64, 7, 6};
    ge::Tensor constTensor;
    ge::TensorDesc tensor_desc_filter_size(ge::Shape(),
      ge::FORMAT_NCHW, ge::DT_INT32);
    int element_size = dims_filter_size.size();
    tensor_desc_filter_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_filter_size);

    int *conv_filter_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_filter_size_tensor_value + i) = dims_filter_size[i];
    }
    constTensor.SetData((uint8_t *) conv_filter_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("filter_size").set_attr_value(constTensor);
    op.set_input_filter_size(const0);
    delete[] conv_filter_size_tensor_value;
    op.UpdateInputDesc("filter_size", tensor_desc_filter_size);

    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {2, 3, 2, 2});
    op.SetAttr("padding", "SAME");
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// -2
TEST_F(Conv2DBackpropFilterProtoTest, Conv2DBackpropFilterUnKnownRankTest) {
    ge::op::Conv2DBackpropFilter op;
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({-2},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {-2}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("x", create_desc_with_ori({-2},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {-2}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({10, 64, 7, 6},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {10, 64, 7, 6}, ge::FORMAT_NCHW));
    std::vector<int64_t> dims_filter_size{10, 64, 7, 6};
    ge::Tensor constTensor;
    ge::TensorDesc tensor_desc_filter_size(ge::Shape(),
      ge::FORMAT_NCHW, ge::DT_INT32);
    int element_size = dims_filter_size.size();
    tensor_desc_filter_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_filter_size);

    int *conv_filter_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_filter_size_tensor_value + i) = dims_filter_size[i];
    }
    constTensor.SetData((uint8_t *) conv_filter_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("filter_size").set_attr_value(constTensor);
    op.set_input_filter_size(const0);
    delete[] conv_filter_size_tensor_value;
    op.UpdateInputDesc("filter_size", tensor_desc_filter_size);

    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("padding", "VALID");
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("groups", 1);
    op.SetAttr("data_format","NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DBackpropFilterProtoTest, Conv2DBackpropFilterSplit) {
    ge::op::Conv2DBackpropFilter op;
    op.UpdateInputDesc("x", create_desc_with_ori({1, 32, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{1, 32, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({1, 32, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{1, 32, 3, 3}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({32, 32, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW,{32, 32, 1, 1}, ge::FORMAT_NCHW));
    op.SetAttr("filter_size", {32, 32, 1, 1});

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {0, 1}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
  
    ge::GeTensorDescPtr tensor_desc_dedy = op_desc->MutableInputDesc("out_backprop");
    std::vector<std::vector<int64_t>> dedy_data_slice;
    ge::AttrUtils::GetListListInt(tensor_desc_dedy, ge::ATTR_NAME_DATA_SLICE, dedy_data_slice);
    
    std::vector<std::vector<int64_t>> expect_dedy_data_slice = {{}, {0, 1}, {}, {}, {}};
    EXPECT_EQ(expect_dedy_data_slice, dedy_data_slice);

}

// get filter_size list failed
TEST_F(Conv2DBackpropFilterProtoTest, Conv2DBackpropFilterVerifyDataSliceTest1) {
    ge::op::Conv2DBackpropFilter op;
    op.UpdateInputDesc("x", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format","NCHW");
    std::string padding = "SAME";
    op.SetAttr("padding", padding);

    std::vector<int64_t> dims_filter_size{};
    ge::Tensor constTensor;
    ge::TensorDesc tensor_desc_filter_size(ge::Shape(),
      ge::FORMAT_NCHW, ge::DT_INT32);
    int element_size = dims_filter_size.size();
    tensor_desc_filter_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_filter_size);

    int *conv_filter_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_filter_size_tensor_value + i) = dims_filter_size[i];
    }
    constTensor.SetData((uint8_t *) conv_filter_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("filter_size").set_attr_value(constTensor);
    op.set_input_filter_size(const0);
    delete[] conv_filter_size_tensor_value;
    op.UpdateInputDesc("filter_size", tensor_desc_filter_size);

    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// not need infer input
TEST_F(Conv2DBackpropFilterProtoTest, Conv2DBackpropFilterVerifyDataSliceTest2) {
    ge::op::Conv2DBackpropFilter op;
    op.UpdateInputDesc("x", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.SetAttr("filter_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format","NCHW");
    std::string padding = "SAME";
    op.SetAttr("padding", padding);

    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// can not supported split in Cin, H and W
TEST_F(Conv2DBackpropFilterProtoTest, Conv2DBackpropFilterVerifyDataSliceTest3) {
    ge::op::Conv2DBackpropFilter op;
    op.UpdateInputDesc("x", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.SetAttr("filter_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format","NCHW");
    std::string padding = "SAME";
    op.SetAttr("padding", padding);

    std::vector<std::vector<int64_t>> y_data_slice ={{6, 20}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// not need infer input
TEST_F(Conv2DBackpropFilterProtoTest, Conv2DBackpropFilterVerifyDataSliceTest4) {
    ge::op::Conv2DBackpropFilter op;
    op.UpdateInputDesc("x", create_desc_with_ori({128, 256, 14, 14},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 256, 14, 14}, ge::FORMAT_NCHW));
    op.UpdateInputDesc("out_backprop", create_desc_with_ori({128, 512, 7, 7},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {128, 512, 7, 7}, ge::FORMAT_NCHW));
    op.UpdateOutputDesc("y", create_desc_with_ori({512, 256, 1, 1},
        ge::DT_FLOAT16, ge::FORMAT_NCHW, {512, 256, 1, 1}, ge::FORMAT_NCHW));
    op.SetAttr("filter_size", {128, 256, 14, 14});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("pads", {0, 0, 0, 0});
    op.SetAttr("dilations", {1, 1, 1, 1});
    op.SetAttr("data_format","NCHW");
    std::string padding = "SAME";
    op.SetAttr("padding", padding);

    std::vector<std::vector<int64_t>> y_data_slice ={{}, {}, {}, {}, {}};
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
    auto status = op_desc->InferDataSlice();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}