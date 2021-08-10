#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "op_desc.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/common_error_codes.h"


class avg_pool : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "avg_pool SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "avg_pool TearDown" << std::endl;
    }
};

TEST_F(avg_pool, avg_pool_infershape_test_fp16_1) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", create_desc_with_ori({3, 16, 16, 64}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {3, 16, 16, 64}, ge::FORMAT_NHWC));

    std::string padding = "VALID";
    std::string data_format = "NHWC";
    op.SetAttr("ksize", {1,2,2,1});
    op.SetAttr("strides", {1,1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {3, 15, 15, 64};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(avg_pool, avg_pool_infershape_test_fp16_2) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", create_desc_with_ori({1, 128, 32, 32}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 128, 32, 32}, ge::FORMAT_NCHW));

    std::string padding = "SAME";
    std::string data_format = "NCHW";
    op.SetAttr("ksize", {1,1,2,2});
    op.SetAttr("strides",{1,1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {1, 128, 32, 32};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

// dynamic hw SAME NHWC
TEST_F(avg_pool, avg_pool_infershape_test_fp16_dynamic_1) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", 
                       create_desc_shape_range({1, -1, -1, 128},
                                                ge::DT_FLOAT16,
                                                ge::FORMAT_NHWC,
                                                {1, -1, -1, 128},
                                                ge::FORMAT_NHWC,
                                                {{1, 1}, {20, 100}, {20, 100}, {128, 128}}));
    std::string padding = "SAME";
    std::string data_format = "NHWC";
    op.SetAttr("ksize", {1,2,2,1});
    op.SetAttr("strides",{1,1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {1, -1, -1, 128};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

// dynamic nhw VALID NCHW range -1
TEST_F(avg_pool, avg_pool_infershape_test_fp16_dynamic_2) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", 
                       create_desc_shape_range({-1, 128, -1, -1},
                                                ge::DT_FLOAT16,
                                                ge::FORMAT_NCHW,
                                                {-1, 128, -1, -1},
                                                ge::FORMAT_NCHW,
                                                {{1, -1}, {128, 128}, {20, -1}, {20, 100}}));
    std::string padding = "VALID";
    std::string data_format = "NCHW";
    op.SetAttr("ksize", {1,1,2,2});
    op.SetAttr("strides",{1,1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {-1, 128, -1, -1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

// dynamic -2
TEST_F(avg_pool, avg_pool_infershape_test_fp16_dynamic_3) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", create_desc_with_ori({-2}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {-2}, ge::FORMAT_NCHW));
    std::string padding = "VALID";
    std::string data_format = "NCHW";
    op.SetAttr("ksize", {1,1,2,2});
    op.SetAttr("strides",{1,1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// dynamic c SAME NHWC
TEST_F(avg_pool, avg_pool_infershape_test_fp16_dynamic_4) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", 
                       create_desc_shape_range({1, 24, 24, -1},
                                                ge::DT_FLOAT16,
                                                ge::FORMAT_NHWC,
                                                {1, 24, 24, -1},
                                                ge::FORMAT_NHWC,
                                                {{1, 1}, {20, 100}, {20, 100}, {128, 128}}));
    std::string padding = "SAME";
    std::string data_format = "NHWC";
    op.SetAttr("ksize", {1,2,2,1});
    op.SetAttr("strides",{1,1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(avg_pool, avg_pool_verify_test_dataformat) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", create_desc_with_ori({1, 128, 32, 32}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 128, 32, 32}, ge::FORMAT_NCHW));

    std::string padding = "SAME";
    std::string data_format = "HWCN";
    op.SetAttr("ksize", {1,1,2,2});
    op.SetAttr("strides",{1,1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_pool, avg_pool_verify_test_strides) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", create_desc_with_ori({1, 128, 32, 32}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 128, 32, 32}, ge::FORMAT_NCHW));

    std::string padding = "SAME";
    std::string data_format = "NCHW";
    op.SetAttr("ksize", {1,1,2,2});
    op.SetAttr("strides",{1,1,1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_pool, avg_pool_verify_test_ksize1) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", create_desc_with_ori({1, 128, 32, 32}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 128, 32, 32}, ge::FORMAT_NCHW));

    std::string padding = "SAME";
    std::string data_format = "NCHW";
    op.SetAttr("ksize", {-1,-1,2,2});
    op.SetAttr("strides",{1,1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_pool, avg_pool_verify_test_ksize2) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", create_desc_with_ori({3, 16, 16, 64}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {3, 16, 16, 64}, ge::FORMAT_NHWC));

    std::string padding = "VALID";
    std::string data_format = "NHWC";
    op.SetAttr("ksize", {-1,2,2,1});
    op.SetAttr("strides", {1,1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_pool, avg_pool_verify_test_strides1) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", create_desc_with_ori({1, 128, 32, 32}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 128, 32, 32}, ge::FORMAT_NCHW));

    std::string padding = "SAME";
    std::string data_format = "NCHW";
    op.SetAttr("ksize", {1,1,2,2});
    op.SetAttr("strides",{-1,-1,1,1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(avg_pool, avg_pool_verify_test_strides2) {
    ge::op::AvgPool op;
    op.UpdateInputDesc("x", create_desc_with_ori({3, 16, 16, 64}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {3, 16, 16, 64}, ge::FORMAT_NHWC));

    std::string padding = "VALID";
    std::string data_format = "NHWC";
    op.SetAttr("ksize", {1,2,2,1});
    op.SetAttr("strides", {1,1,1,-1});
    op.SetAttr("padding", padding);
    op.SetAttr("data_format", data_format);

    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// fuzz build all static shape
TEST_F(avg_pool, avg_pool_fuzz_build_all_static_shape) {
    ge::op::AvgPool op;
    op.SetAttr("_fuzz_build", true);
    op.UpdateInputDesc("x", create_desc_with_ori(
        {16, 3, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {16, 3, 16, 16}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("padding", "VALID");
    op.SetAttr("data_format", "NCHW");
    op.SetAttr("ksize", {1,1,3,5});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = op.InferShapeAndType();
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::pair<int64_t, int64_t>> input_range;
    tensor_desc_x->GetShapeRange(input_range);
    std::vector<std::pair<int64_t, int64_t>> expect_x_range = {{16, 31}, {3,3}, {16, 31}, {16, 31}};
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ((input_range == expect_x_range), true);
}

// fuzz build all static shape
TEST_F(avg_pool, avg_pool_fuzz_build_all_static_shape_1) {
    ge::op::AvgPool op;
    op.SetAttr("_fuzz_build", true);
    op.UpdateInputDesc("x", create_desc_with_ori(
        {16, 3, 4096, 16}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {16, 3, 16, 16}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 1, 1});
    op.SetAttr("padding", "VALID");
    op.SetAttr("data_format", "NCHW");
    op.SetAttr("ksize", {1,1,3,5});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = op.InferShapeAndType();
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
    std::vector<std::pair<int64_t, int64_t>> input_range;
    tensor_desc_x->GetShapeRange(input_range);
    std::vector<std::pair<int64_t, int64_t>> expect_x_range = {{16, 31}, {3,3}, {1024, 4096}, {16, 31}};
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ((input_range == expect_x_range), true);
}

// fuzz build correct left range
TEST_F(avg_pool, avg_pool_fuzz_build_correct_left_range) {
    ge::op::AvgPool op;
    op.SetAttr("_fuzz_build", true);
    op.UpdateInputDesc("x", create_desc_with_ori(
        {1, 93, 47, 452}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 93, 47, 452}, ge::FORMAT_NCHW));
    op.SetAttr("strides", {1, 1, 4, 1});
    op.SetAttr("padding", "VALID");
    op.SetAttr("data_format", "NCHW");
    op.SetAttr("ksize", {1,1,31,97});
    auto status = op.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = op.InferShapeAndType();
    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(avg_pool, AvgPool_data_slice_infer1) {
  ge::op::AvgPool op;

  // set AvgPool attr
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

TEST_F(avg_pool, AvgPool_data_slice_infer2) {
  ge::op::AvgPool op;

  // set AvgPool attr
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

TEST_F(avg_pool, AvgPool_data_slice_infer3) {
  ge::op::AvgPool op;

  // set AvgPool attr
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

TEST_F(avg_pool, AvgPool_data_slice_infer4) {
  ge::op::AvgPool op;

  // set AvgPool attr
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

TEST_F(avg_pool, AvgPool_data_slice_infer5) {
  ge::op::AvgPool op;

  // set AvgPool attr
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