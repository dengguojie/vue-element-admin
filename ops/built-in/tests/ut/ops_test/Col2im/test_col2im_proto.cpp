#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "transformation_ops.h"
#include "array_ops.h"

using namespace std;

const static char* x_c = "x";
const static char* output_size_c = "output_size";
const static char* kernel_size_c = "kernel_size";
const static char* dilation_c = "dilation";
const static char* padding_c = "padding";
const static char* stride_c = "stride";
const static char* y_c = "y";
class Col2imTest : public testing::Test {
  protected:
    static void SetUpTestCase() {
      std::cout << "Col2im Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "Col2im Proto Test TearDown" << std::endl;
    }
};

TEST_F(Col2imTest, col2im_test_case_0){
  ge::op::Col2im op;


  ge::TensorDesc x_tensor_desc = create_desc_with_ori(
    {2,48,9,16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {2,48,9,16}, ge::FORMAT_NCHW
  );
  op.UpdateInputDesc(x_c, x_tensor_desc);

  ge::Tensor output_size_tensor;
  ge::TensorDesc output_size_tensor_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  output_size_tensor.SetTensorDesc(output_size_tensor_desc);
  int32_t output_size_data[2] = {6, 6};
  output_size_tensor.SetData((uint8_t*)output_size_data, 2 * sizeof(int32_t));
  auto output_size_const = ge::op::Constant().set_attr_value(output_size_tensor);

  op.set_input_output_size(output_size_const);
  op.UpdateInputDesc(output_size_c, output_size_tensor_desc);

  vector<int32_t> kernel_size({3, 3});
  op.SetAttr(kernel_size_c, kernel_size);
  vector<int32_t> dilation({1, 1});
  op.SetAttr(dilation_c, dilation);
  vector<int32_t> padding({0, 0});
  op.SetAttr(padding_c, padding);
  vector<int32_t> stride({1, 1});
  op.SetAttr(stride_c, stride);

  auto verify_ret = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_ret, ge::GRAPH_SUCCESS);

  auto infer_ret = op.InferShapeAndType();
  EXPECT_EQ(infer_ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName(y_c);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
}

TEST_F(Col2imTest, col2im_test_case_1){
  ge::op::Col2im op;

  ge::TensorDesc x_tensor_desc = create_desc_with_ori(
    {2,48,9,16}, ge::DT_FLOAT, ge::FORMAT_NCHW, {2,48,9,16}, ge::FORMAT_NCHW
  );
  op.UpdateInputDesc(x_c, x_tensor_desc);

  ge::Tensor output_size_tensor;
  ge::TensorDesc output_size_tensor_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  output_size_tensor.SetTensorDesc(output_size_tensor_desc);
  int32_t output_size_data[2] = {6, 6};
  output_size_tensor.SetData((uint8_t*)output_size_data, 2 * sizeof(int32_t));
  auto output_size_const = ge::op::Constant().set_attr_value(output_size_tensor);

  op.set_input_output_size(output_size_const);
  op.UpdateInputDesc(output_size_c, output_size_tensor_desc);

  vector<int32_t> kernel_size({3, 3});
  op.SetAttr(kernel_size_c, kernel_size);
  vector<int32_t> dilation({1, 1});
  op.SetAttr(dilation_c, dilation);
  vector<int32_t> padding({0, 0});
  op.SetAttr(padding_c, padding);
  vector<int32_t> stride({1, 1});
  op.SetAttr(stride_c, stride);

  auto verify_ret = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_ret, ge::GRAPH_SUCCESS);

  auto infer_ret = op.InferShapeAndType();
  EXPECT_EQ(infer_ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName(y_c);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
}

TEST_F(Col2imTest, col2im_test_case_2){
  ge::op::Col2im op;

  ge::TensorDesc x_tensor_desc = create_desc_with_ori(
    {9,16,16}, ge::DT_FLOAT, ge::FORMAT_NCHW, {9,16,16}, ge::FORMAT_NCHW
  );
  op.UpdateInputDesc(x_c, x_tensor_desc);

  ge::Tensor output_size_tensor;
  ge::TensorDesc output_size_tensor_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  output_size_tensor.SetTensorDesc(output_size_tensor_desc);
  int32_t output_size_data[2] = {6, 6};
  output_size_tensor.SetData((uint8_t*)output_size_data, 2 * sizeof(int32_t));
  auto output_size_const = ge::op::Constant().set_attr_value(output_size_tensor);

  op.set_input_output_size(output_size_const);
  op.UpdateInputDesc(output_size_c, output_size_tensor_desc);

  vector<int32_t> kernel_size({3, 3});
  op.SetAttr(kernel_size_c, kernel_size);
  vector<int32_t> dilation({1, 1});
  op.SetAttr(dilation_c, dilation);
  vector<int32_t> padding({0, 0});
  op.SetAttr(padding_c, padding);
  vector<int32_t> stride({1, 1});
  op.SetAttr(stride_c, stride);

  auto verify_ret = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_ret, ge::GRAPH_SUCCESS);

  auto infer_ret = op.InferShapeAndType();
  EXPECT_EQ(infer_ret, ge::GRAPH_FAILED);

  auto output_desc = op.GetOutputDescByName(y_c);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
}

TEST_F(Col2imTest, col2im_test_case_3){
  ge::op::Col2im op;

  ge::TensorDesc x_tensor_desc = create_desc_with_ori(
    {1,16,9,16}, ge::DT_FLOAT, ge::FORMAT_NCHW, {1,16,9,16}, ge::FORMAT_NCHW
  );
  op.UpdateInputDesc(x_c, x_tensor_desc);

  ge::Tensor output_size_tensor;
  ge::TensorDesc output_size_tensor_desc(ge::Shape({3}), ge::FORMAT_ND, ge::DT_INT32);
  output_size_tensor.SetTensorDesc(output_size_tensor_desc);
  int32_t output_size_data[3] = {6, 6, 6};
  output_size_tensor.SetData((uint8_t*)output_size_data, 3 * sizeof(int32_t));
  auto output_size_const = ge::op::Constant().set_attr_value(output_size_tensor);

  op.set_input_output_size(output_size_const);
  op.UpdateInputDesc(output_size_c, output_size_tensor_desc);

  vector<int32_t> kernel_size({3, 3});
  op.SetAttr(kernel_size_c, kernel_size);
  vector<int32_t> dilation({1, 1});
  op.SetAttr(dilation_c, dilation);
  vector<int32_t> padding({0, 0});
  op.SetAttr(padding_c, padding);
  vector<int32_t> stride({1, 1});
  op.SetAttr(stride_c, stride);

  auto verify_ret = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_ret, ge::GRAPH_SUCCESS);

  auto infer_ret = op.InferShapeAndType();
  EXPECT_EQ(infer_ret, ge::GRAPH_FAILED);

  auto output_desc = op.GetOutputDescByName(y_c);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
}

TEST_F(Col2imTest, col2im_test_case_4){
  ge::op::Col2im op;

  ge::TensorDesc x_tensor_desc = create_desc_with_ori(
    {1,16,9,16}, ge::DT_FLOAT, ge::FORMAT_NCHW, {1,16,9,16}, ge::FORMAT_NCHW
  );
  op.UpdateInputDesc(x_c, x_tensor_desc);

  ge::Tensor output_size_tensor;
  ge::TensorDesc output_size_tensor_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  output_size_tensor.SetTensorDesc(output_size_tensor_desc);
  int32_t output_size_data[2] = {6, 6};
  output_size_tensor.SetData((uint8_t*)output_size_data, 2 * sizeof(int32_t));
  auto output_size_const = ge::op::Constant().set_attr_value(output_size_tensor);

  op.set_input_output_size(output_size_const);
  op.UpdateInputDesc(output_size_c, output_size_tensor_desc);

  vector<int32_t> kernel_size({3, 3, 2});
  op.SetAttr(kernel_size_c, kernel_size);
  vector<int32_t> dilation({1, 1});
  op.SetAttr(dilation_c, dilation);
  vector<int32_t> padding({0, 0});
  op.SetAttr(padding_c, padding);
  vector<int32_t> stride({1, 1});
  op.SetAttr(stride_c, stride);

  auto verify_ret = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_ret, ge::GRAPH_FAILED);

  auto infer_ret = op.InferShapeAndType();
  EXPECT_EQ(infer_ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName(y_c);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
}

TEST_F(Col2imTest, col2im_test_case_5){
  ge::op::Col2im op;

  ge::TensorDesc x_tensor_desc = create_desc_with_ori(
    {1,16,9,16}, ge::DT_FLOAT, ge::FORMAT_NCHW, {1,16,9,16}, ge::FORMAT_NCHW
  );
  op.UpdateInputDesc(x_c, x_tensor_desc);

  ge::Tensor output_size_tensor;
  ge::TensorDesc output_size_tensor_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  output_size_tensor.SetTensorDesc(output_size_tensor_desc);
  int32_t output_size_data[2] = {6, 6};
  output_size_tensor.SetData((uint8_t*)output_size_data, 2 * sizeof(int32_t));
  auto output_size_const = ge::op::Constant().set_attr_value(output_size_tensor);

  op.set_input_output_size(output_size_const);
  op.UpdateInputDesc(output_size_c, output_size_tensor_desc);

  vector<int32_t> kernel_size({3, 3});
  op.SetAttr(kernel_size_c, kernel_size);
  vector<int32_t> dilation({1, 1, 1});
  op.SetAttr(dilation_c, dilation);
  vector<int32_t> padding({0, 0});
  op.SetAttr(padding_c, padding);
  vector<int32_t> stride({1, 1});
  op.SetAttr(stride_c, stride);

  auto verify_ret = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_ret, ge::GRAPH_FAILED);

  auto infer_ret = op.InferShapeAndType();
  EXPECT_EQ(infer_ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName(y_c);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
}

TEST_F(Col2imTest, col2im_test_case_6){
  ge::op::Col2im op;

  ge::TensorDesc x_tensor_desc = create_desc_with_ori(
    {1,16,9,16}, ge::DT_FLOAT, ge::FORMAT_NCHW, {1,16,9,16}, ge::FORMAT_NCHW
  );
  op.UpdateInputDesc(x_c, x_tensor_desc);

  ge::Tensor output_size_tensor;
  ge::TensorDesc output_size_tensor_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  output_size_tensor.SetTensorDesc(output_size_tensor_desc);
  int32_t output_size_data[2] = {6, 6};
  output_size_tensor.SetData((uint8_t*)output_size_data, 2 * sizeof(int32_t));
  auto output_size_const = ge::op::Constant().set_attr_value(output_size_tensor);

  op.set_input_output_size(output_size_const);
  op.UpdateInputDesc(output_size_c, output_size_tensor_desc);

  vector<int32_t> kernel_size({3, 3});
  op.SetAttr(kernel_size_c, kernel_size);
  vector<int32_t> dilation({1, 1});
  op.SetAttr(dilation_c, dilation);
  vector<int32_t> padding({0, 0, 0});
  op.SetAttr(padding_c, padding);
  vector<int32_t> stride({1, 1});
  op.SetAttr(stride_c, stride);

  auto verify_ret = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_ret, ge::GRAPH_FAILED);

  auto infer_ret = op.InferShapeAndType();
  EXPECT_EQ(infer_ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName(y_c);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
}

TEST_F(Col2imTest, col2im_test_case_7){
  ge::op::Col2im op;

  ge::TensorDesc x_tensor_desc = create_desc_with_ori(
    {1,16,9,16}, ge::DT_FLOAT, ge::FORMAT_NCHW, {1,16,9,16}, ge::FORMAT_NCHW
  );
  op.UpdateInputDesc(x_c, x_tensor_desc);

  ge::Tensor output_size_tensor;
  ge::TensorDesc output_size_tensor_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  output_size_tensor.SetTensorDesc(output_size_tensor_desc);
  int32_t output_size_data[2] = {6, 6};
  output_size_tensor.SetData((uint8_t*)output_size_data, 2 * sizeof(int32_t));
  auto output_size_const = ge::op::Constant().set_attr_value(output_size_tensor);

  op.set_input_output_size(output_size_const);
  op.UpdateInputDesc(output_size_c, output_size_tensor_desc);

  vector<int32_t> kernel_size({3, 3});
  op.SetAttr(kernel_size_c, kernel_size);
  vector<int32_t> dilation({1, 1});
  op.SetAttr(dilation_c, dilation);
  vector<int32_t> padding({0, 0});
  op.SetAttr(padding_c, padding);
  vector<int32_t> stride({1, 1, 1});
  op.SetAttr(stride_c, stride);

  auto verify_ret = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_ret, ge::GRAPH_FAILED);

  auto infer_ret = op.InferShapeAndType();
  EXPECT_EQ(infer_ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDescByName(y_c);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
}

TEST_F(Col2imTest, col2im_test_case_8){
  ge::op::Col2im op;

  ge::TensorDesc x_tensor_desc = create_desc_with_ori(
    {1,16,9,2044}, ge::DT_FLOAT, ge::FORMAT_NCHW, {1,16,9,2044}, ge::FORMAT_NCHW
  );
  op.UpdateInputDesc(x_c, x_tensor_desc);

  ge::Tensor output_size_tensor;
  ge::TensorDesc output_size_tensor_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  output_size_tensor.SetTensorDesc(output_size_tensor_desc);
  int32_t output_size_data[2] = {4, 1024};
  output_size_tensor.SetData((uint8_t*)output_size_data, 2 * sizeof(int32_t));
  auto output_size_const = ge::op::Constant().set_attr_value(output_size_tensor);

  op.set_input_output_size(output_size_const);
  op.UpdateInputDesc(output_size_c, output_size_tensor_desc);

  vector<int32_t> kernel_size({3, 3});
  op.SetAttr(kernel_size_c, kernel_size);
  vector<int32_t> dilation({1, 1});
  op.SetAttr(dilation_c, dilation);
  vector<int32_t> padding({0, 0});
  op.SetAttr(padding_c, padding);
  vector<int32_t> stride({1, 1});
  op.SetAttr(stride_c, stride);

  auto verify_ret = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_ret, ge::GRAPH_SUCCESS);

  auto infer_ret = op.InferShapeAndType();
  EXPECT_EQ(infer_ret, ge::GRAPH_FAILED);

  auto output_desc = op.GetOutputDescByName(y_c);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
}