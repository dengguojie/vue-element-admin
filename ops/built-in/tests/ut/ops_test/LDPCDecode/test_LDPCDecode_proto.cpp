#include <gtest/gtest.h>

#include <iostream>

#include "encoding_ops.h"
#include "op_proto_test_util.h"
#include "utils/op_desc_utils.h"


class LDPCDecodeTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "LDPCDecode Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "LDPCDecode Proto Test TearDown" << std::endl;
  }
};

TEST_F(LDPCDecodeTest, LDPC_decode_infershape_test1){
  ge::op::LDPCDecode op;

  ge::DataType dtype = ge::DT_INT32;
  ge::Format format = ge::FORMAT_ND;
  auto valid_num_desc = create_desc_with_ori({12}, dtype, format, {12}, format);
  auto matrix_info_desc = create_desc_with_ori({72, 3}, dtype, format, {72, 3}, format);
  op.UpdateInputDesc("valid_num", valid_num_desc);
  op.UpdateInputDesc("matrix_info", matrix_info_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("indices");
  EXPECT_EQ(output_desc.GetDataType(), dtype);

  std::vector<int64_t> expected_output_shape = {6144, 6};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(LDPCDecodeTest, LDPC_decode_infershape_test2){
  ge::op::LDPCDecode op;

  ge::DataType dtype = ge::DT_INT32;
  ge::DataType err_dtype = ge::DT_INT64;
  ge::Format format = ge::FORMAT_ND;
  auto valid_num_desc = create_desc_with_ori({12}, dtype, format, {12}, format);
  auto matrix_info_desc = create_desc_with_ori({72, 3}, err_dtype, format, {72, 3}, format);
  op.UpdateInputDesc("valid_num", valid_num_desc);
  op.UpdateInputDesc("matrix_info", matrix_info_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(LDPCDecodeTest, LDPC_decode_infershape_test3){
  ge::op::LDPCDecode op;

  ge::DataType dtype = ge::DT_INT32;
  ge::Format format = ge::FORMAT_ND;
  auto valid_num_desc = create_desc_with_ori({12, 3}, dtype, format, {12, 3}, format);
  auto matrix_info_desc = create_desc_with_ori({72, 3}, dtype, format, {72, 3}, format);
  op.UpdateInputDesc("valid_num", valid_num_desc);
  op.UpdateInputDesc("matrix_info", matrix_info_desc);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
