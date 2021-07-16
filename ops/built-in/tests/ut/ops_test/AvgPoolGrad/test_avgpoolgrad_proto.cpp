#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "nn_pooling_ops.h"
#include "utils/attr_utils.h"
#include "utils/op_desc_utils.h"


class AvgPoolGradProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AvgPoolGrad Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AvgPoolGrad Proto Test TearDown" << std::endl;
  }
};

// dynamic nhw + Const + range -1
TEST_F(AvgPoolGradProtoTest, Base_Pass_Case){
    ge::op::AvgPoolGrad op;
    op.UpdateInputDesc("input_grad",
                       create_desc_shape_range({-1, 32, -1, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, 32, -1, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 2}, {32, 32}, {1, -1}, {1, 2}}));
    op.UpdateOutputDesc("out_grad",
                       create_desc_shape_range({-1, 32, -1, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, 32, -1, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 2}, {32, 32}, {1, -1}, {1, 8}}));
    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{1 ,32 ,4 , 4};
    ge::TensorDesc tensor_desc_input_size(ge::Shape(),
      ge::FORMAT_NCHW, ge::DT_INT32);
    int element_size = dims_input_size.size();
    tensor_desc_input_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_input_size);

    int *conv_input_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_input_size_tensor_value + i) = dims_input_size[i];
    }
    constTensor.SetData((uint8_t *) conv_input_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("input_size").set_attr_value(constTensor);
    op.set_input_orig_input_shape(const0);
    delete[] conv_input_size_tensor_value;
    op.UpdateInputDesc("orig_input_shape", tensor_desc_input_size);

    op.SetAttr("ksize", {1, 1, 3, 3});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("padding", "VALID");
    op.SetAttr("data_format", "NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("out_grad");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

// dynamic nw + var 
TEST_F(AvgPoolGradProtoTest, No_Input_Size){
    ge::op::AvgPoolGrad op;
    op.UpdateInputDesc("input_grad",
                       create_desc_shape_range({-1, 32, 2, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, 32, 2, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 2}, {32, 32}, {1, 2}, {1, 2}}));
          
    auto avg_pool_grad_input_ori_shape_data = ge::op::Data("orig_input_shape");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    avg_pool_grad_input_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    avg_pool_grad_input_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_orig_input_shape(avg_pool_grad_input_ori_shape_data);
    op.UpdateInputDesc("orig_input_shape", ori_tensorDesc);

    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    auto input_sizes_desc = op_desc->MutableInputDesc("orig_input_shape");
    std::vector<std::pair<int64_t, int64_t>> value_range = {{1, 2}, {32, 32}, {4, 5}, {4, 5}};
    input_sizes_desc->SetValueRange(value_range);

    op.UpdateOutputDesc("out_grad",
                    create_desc_shape_range({-1, 32, -1, -1},
                                            ge::DT_FLOAT16,
                                            ge::FORMAT_NCHW,
                                            {-1, 32, -1, -1},
                                            ge::FORMAT_NCHW,
                                            {{1, 2}, {32, 32}, {4, 5}, {4, 5}}));

    op.SetAttr("ksize", {1, 1, 3, 3});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("padding", "SAME");
    op.SetAttr("data_format", "NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("out_grad");
    std::vector<std::pair<int64_t, int64_t>> output_range;
    tensor_desc_y->GetShapeRange(output_range);
    std::vector<std::pair<int64_t, int64_t>> expect_y_range = {{1, 2}, {32, 32}, {4, 5}, {4, 5}};
    EXPECT_EQ((output_range == expect_y_range), true);
}

// dynamic c + var 
TEST_F(AvgPoolGradProtoTest, avg_pool_grad_dynamic_c){
    ge::op::AvgPoolGrad op;
    op.UpdateInputDesc("input_grad",
                       create_desc_shape_range({1, -1, 2, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {1, -1, 2, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 2}, {16, 32}, {1, 2}, {1, 2}}));
          
    auto avg_pool_grad_input_ori_shape_data = ge::op::Data("orig_input_shape");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    avg_pool_grad_input_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    avg_pool_grad_input_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_orig_input_shape(avg_pool_grad_input_ori_shape_data);
    op.UpdateInputDesc("orig_input_shape", ori_tensorDesc);

    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    auto input_sizes_desc = op_desc->MutableInputDesc("orig_input_shape");
    std::vector<std::pair<int64_t, int64_t>> value_range = {{1, 2}, {32, 32}, {4, 5}, {4, 5}};
    input_sizes_desc->SetValueRange(value_range);

    op.UpdateOutputDesc("out_grad",
                    create_desc_shape_range({-1, 32, -1, -1},
                                            ge::DT_FLOAT16,
                                            ge::FORMAT_NCHW,
                                            {-1, 32, -1, -1},
                                            ge::FORMAT_NCHW,
                                            {{1, 2}, {32, 32}, {4, 5}, {4, 5}}));

    op.SetAttr("ksize", {1, 1, 3, 3});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("padding", "SAME");
    op.SetAttr("data_format", "NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dynamic opti ut input_grad shape [-1, c, -1, -1] with no range
TEST_F(AvgPoolGradProtoTest, avg_pool_grad_dynamic_nhw_no_range){
    ge::op::AvgPoolGrad op;
    op.UpdateInputDesc("input_grad",
                       create_desc_shape_range({1, -1, 2, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {1, -1, 2, -1},
                                               ge::FORMAT_NCHW,
                                               {}));
          
    auto avg_pool_grad_input_ori_shape_data = ge::op::Data("orig_input_shape");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    avg_pool_grad_input_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    avg_pool_grad_input_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_orig_input_shape(avg_pool_grad_input_ori_shape_data);
    op.UpdateInputDesc("orig_input_shape", ori_tensorDesc);

    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    auto input_sizes_desc = op_desc->MutableInputDesc("orig_input_shape");
    std::vector<std::pair<int64_t, int64_t>> value_range = {{1, 2}, {32, 32}, {4, 5}, {4, 5}};
    input_sizes_desc->SetValueRange(value_range);

    op.UpdateOutputDesc("out_grad",
                    create_desc_shape_range({-1, 32, -1, -1},
                                            ge::DT_FLOAT16,
                                            ge::FORMAT_NCHW,
                                            {-1, 32, -1, -1},
                                            ge::FORMAT_NCHW,
                                            {{1, 2}, {32, 32}, {4, 5}, {4, 5}}));

    op.SetAttr("ksize", {1, 1, 3, 3});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("padding", "SAME");
    op.SetAttr("data_format", "NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

//dynamic opti ut input_grad shape [-1, c, -1, -1] with range < 4
TEST_F(AvgPoolGradProtoTest, avg_pool_grad_dynamic_nhw_part_range){
    ge::op::AvgPoolGrad op;
    op.UpdateInputDesc("input_grad",
                       create_desc_shape_range({1, -1, 2, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {1, -1, 2, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 2}}));
          
    auto avg_pool_grad_input_ori_shape_data = ge::op::Data("orig_input_shape");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    avg_pool_grad_input_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    avg_pool_grad_input_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_orig_input_shape(avg_pool_grad_input_ori_shape_data);
    op.UpdateInputDesc("orig_input_shape", ori_tensorDesc);

    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    auto input_sizes_desc = op_desc->MutableInputDesc("orig_input_shape");
    std::vector<std::pair<int64_t, int64_t>> value_range = {{1, 2}, {32, 32}, {4, 5}, {4, 5}};
    input_sizes_desc->SetValueRange(value_range);

    op.UpdateOutputDesc("out_grad",
                    create_desc_shape_range({-1, 32, -1, -1},
                                            ge::DT_FLOAT16,
                                            ge::FORMAT_NCHW,
                                            {-1, 32, -1, -1},
                                            ge::FORMAT_NCHW,
                                            {{1, 2}, {32, 32}, {4, 5}, {4, 5}}));

    op.SetAttr("ksize", {1, 1, 3, 3});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("padding", "SAME");
    op.SetAttr("data_format", "NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// dynamic -2 + Const 
TEST_F(AvgPoolGradProtoTest, avg_pool_grad_dynamic_rank){
    ge::op::AvgPoolGrad op;
    op.UpdateInputDesc("input_grad", 
                       create_desc_with_ori({-2}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {-2}, ge::FORMAT_NHWC));
    op.UpdateOutputDesc("out_grad",
                       create_desc_shape_range({-1, 32, -1, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, 32, -1, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 2}, {32, 32}, {1, -1}, {1, 8}}));

    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{1, 4, 4, 32};
    ge::TensorDesc tensor_desc_input_size(ge::Shape(),
      ge::FORMAT_NHWC, ge::DT_INT32);
    int element_size = dims_input_size.size();
    tensor_desc_input_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_input_size);

    int *conv_input_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_input_size_tensor_value + i) = dims_input_size[i];
    }
    constTensor.SetData((uint8_t *) conv_input_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("input_size").set_attr_value(constTensor);
    op.set_input_orig_input_shape(const0);
    delete[] conv_input_size_tensor_value;
    op.UpdateInputDesc("orig_input_shape", tensor_desc_input_size);

    op.SetAttr("ksize", {1, 3, 3, 1});
    op.SetAttr("strides", {1, 2, 2, 1});
    op.SetAttr("padding", "VALID");
    op.SetAttr("data_format", "NHWC");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// fix NHWC
TEST_F(AvgPoolGradProtoTest, avg_pool_grad_fix){
    ge::op::AvgPoolGrad op;
    op.UpdateInputDesc("input_grad", 
                       create_desc_with_ori({1, 1, 1, 32}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 1, 1, 32}, ge::FORMAT_NHWC));
    op.UpdateOutputDesc("out_grad",
                       create_desc_with_ori({1, 4, 4, 32}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 4, 4, 32}, ge::FORMAT_NHWC));
    ge::Tensor constTensor;
    std::vector<int64_t> dims_input_size{1, 4, 4, 32};
    ge::TensorDesc tensor_desc_input_size(ge::Shape(),
      ge::FORMAT_NHWC, ge::DT_INT32);
    int element_size = dims_input_size.size();
    tensor_desc_input_size.SetSize(element_size * sizeof(int32_t));
    constTensor.SetTensorDesc(tensor_desc_input_size);

    int *conv_input_size_tensor_value = new int[element_size];
    for (int i = 0; i < element_size; i++) {
        *(conv_input_size_tensor_value + i) = dims_input_size[i];
    }
    constTensor.SetData((uint8_t *) conv_input_size_tensor_value,
      element_size * sizeof(int32_t));
    auto const0 = ge::op::Constant("orig_input_shape").set_attr_value(constTensor);
    op.set_input_orig_input_shape(const0);
    delete[] conv_input_size_tensor_value;
    op.UpdateInputDesc("orig_input_shape", tensor_desc_input_size);

    op.SetAttr("ksize", {1, 3, 3, 1});
    op.SetAttr("strides", {1, 2, 2, 1});
    op.SetAttr("padding", "VALID");
    op.SetAttr("data_format", "NHWC");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(AvgPoolGradProtoTest, avg_pool_grad_verify_test_dataformat) {
    ge::op::AvgPoolGrad op;
    op.UpdateInputDesc("input_grad",
                       create_desc_shape_range({-1, 32, -1, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, 32, -1, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 2}, {32, 32}, {1, 2}, {1, 2}}));

    op.SetAttr("ksize", {1, 1, 3, 3});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("padding", "SAME");
    op.SetAttr("data_format", "HWCN");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolGradProtoTest, avg_pool_grad_verify_test_padding) {
    ge::op::AvgPoolGrad op;
    op.UpdateInputDesc("input_grad",
                       create_desc_shape_range({-1, 32, -1, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, 32, -1, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 2}, {32, 32}, {1, 2}, {1, 2}}));

    op.SetAttr("ksize", {1, 1, 3, 3});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("padding", "LIST");
    op.SetAttr("data_format", "NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(AvgPoolGradProtoTest, avg_pool_grad_verify_test_strides) {
    ge::op::AvgPoolGrad op;
    op.UpdateInputDesc("input_grad",
                       create_desc_shape_range({-1, 32, -1, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, 32, -1, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 2}, {32, 32}, {1, 2}, {1, 2}}));

    op.SetAttr("ksize", {1, 1, 3, 3});
    op.SetAttr("strides", {1, 1, 2});
    op.SetAttr("padding", "SAME");
    op.SetAttr("data_format", "NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// no_output shape and no value_range
TEST_F(AvgPoolGradProtoTest, avg_pool_grad_no_outputshape){
    ge::op::AvgPoolGrad op;
    op.UpdateInputDesc("input_grad",
                       create_desc_shape_range({-1, 32, 2, -1},
                                               ge::DT_FLOAT16,
                                               ge::FORMAT_NCHW,
                                               {-1, 32, 2, -1},
                                               ge::FORMAT_NCHW,
                                               {{1, 2}, {32, 32}, {1, 2}, {1, 2}}));
          
    auto avg_pool_grad_input_ori_shape_data = ge::op::Data("orig_input_shape");
    std::vector<int64_t> ori_dims{4};
    ge::Shape ori_shape(ori_dims);
    ge::TensorDesc ori_tensorDesc(ori_shape, ge::FORMAT_NCHW, ge::DT_INT32);
    avg_pool_grad_input_ori_shape_data.update_input_desc_x(ori_tensorDesc);
    avg_pool_grad_input_ori_shape_data.update_output_desc_y(ori_tensorDesc);
    op.set_input_orig_input_shape(avg_pool_grad_input_ori_shape_data);
    op.UpdateInputDesc("orig_input_shape", ori_tensorDesc);

    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    auto input_sizes_desc = op_desc->MutableInputDesc("orig_input_shape");

    op.SetAttr("ksize", {1, 1, 3, 3});
    op.SetAttr("strides", {1, 1, 2, 2});
    op.SetAttr("padding", "SAME");
    op.SetAttr("data_format", "NCHW");

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}