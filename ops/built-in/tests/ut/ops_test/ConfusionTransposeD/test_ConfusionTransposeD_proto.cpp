#include "gtest/gtest.h"
#include "graph/common_error_codes.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "array_ops.h"
#include "op_proto_test_util.h"
#include "transformation_ops.h"

using namespace ge;
using namespace op;

class confusionTransposeD_test : public testing::Test {
  protected:
  static void SetUpTestCase() {
    std::cout << "confusionTransposeD_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "confusionTransposeD_test TearDown" << std::endl;
  }
};
TEST_F(confusionTransposeD_test, confusionTransposeD_test_1) {
  ge::op::ConfusionTransposeD op;  
  // set input info
  auto input_x_shape = vector<int64_t>({10, 11, 20, 16});
  std::vector<std::pair<int64_t,int64_t>> input_x_range = {{10, 10}, {11, 11}, {20, 20}, {16, 16}};
  auto test_format = ge::FORMAT_NHWC;
  auto input_x_desc = create_desc_shape_range(input_x_shape, ge::DT_FLOAT16, test_format,
  input_x_shape, test_format, input_x_range);
  std::vector<int64_t> expected_shape = {2, 2, 2, 2};
  op.UpdateInputDesc("x", input_x_desc);
  op.set_attr_shape({2,2,2,2});
  op.set_attr_perm({3,3,3,3});
  op.set_attr_transpose_first(true);
  auto ret = op.InferShapeAndType();
  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}
TEST_F(confusionTransposeD_test, confusionTransposeD_test_2) {
  ge::op::ConfusionTransposeD op;  
  // set input info
  auto input_x_shape = vector<int64_t>({10, 11, 20, 16});
  std::vector<std::pair<int64_t,int64_t>> input_x_range = {{10, 10}, {11, 11}, {20, 20}, {16, 16}};
  auto test_format = ge::FORMAT_NHWC;
  auto input_x_desc = create_desc_shape_range(input_x_shape, ge::DT_FLOAT16, test_format,
  input_x_shape, test_format, input_x_range);
  op.UpdateInputDesc("x", input_x_desc);
  op.set_attr_shape({2,2,2,2});
  op.set_attr_perm({3,3,3,3});
  auto ret = op.InferShapeAndType();
  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
TEST_F(confusionTransposeD_test, confusionTransposeD_test_3) {
  ge::op::ConfusionTransposeD op;  
  // set input info
  auto input_x_shape = vector<int64_t>({10, 11, 20, 16});
  std::vector<std::pair<int64_t,int64_t>> input_x_range = {{10, 10}, {11, 11}, {20, 20}, {16, 16}};
  auto input_shape_shape = vector<int64_t>({2, 2, 2, 2});
  std::vector<std::pair<int64_t,int64_t>> input_shape_range = {{2, 2}, {2, 2}, {2, 2}, {2, 2}};
  auto test_format = ge::FORMAT_NHWC;
  auto input_x_desc = create_desc_shape_range(input_x_shape, ge::DT_FLOAT16, test_format,
  input_x_shape, test_format, input_x_range);
  auto input_shape_desc = create_desc_shape_range(input_shape_shape, ge::DT_INT32, test_format,
  input_shape_shape, test_format, input_shape_range);
  op.UpdateInputDesc("x", input_x_desc);
  op.set_attr_shape({2,2,2,2});
  op.set_attr_transpose_first(true);
  auto ret = op.InferShapeAndType();
  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
TEST_F(confusionTransposeD_test, confusionTransposeD_test_4) {
  ge::op::ConfusionTransposeD op;  
  // set input info
  auto input_x_shape = vector<int64_t>({10, 11, 20, 16});
  std::vector<std::pair<int64_t,int64_t>> input_x_range = {{10, 10}, {11, 11}, {20, 20}, {16, 16}};
  auto test_format = ge::FORMAT_NHWC;
  auto input_x_desc = create_desc_shape_range(input_x_shape, ge::DT_FLOAT16, test_format,
  input_x_shape, test_format, input_x_range);
  op.UpdateInputDesc("x", input_x_desc);
  op.set_attr_perm({3,3,3,3});
  op.set_attr_transpose_first(true);
  auto ret = op.InferShapeAndType();
  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
TEST_F(confusionTransposeD_test, confusionTransposeD_test_5) {
  ge::op::ConfusionTransposeD op;  
  // set input info
  auto input_x_shape = vector<int64_t>({10, 11, 20, 16});
  std::vector<std::pair<int64_t,int64_t>> input_x_range = {{10, 10}, {11, 11}, {20, 20}, {16, 16}};
  auto test_format = ge::FORMAT_NHWC;
  auto input_x_desc = create_desc_shape_range(input_x_shape, ge::DT_FLOAT16, test_format,
  input_x_shape, test_format, input_x_range);
  std::vector<int64_t> expected_shape = {2, 2, 2, 2};
  op.UpdateInputDesc("x", input_x_desc);
  op.set_attr_shape({2,2,2,2});
  op.set_attr_perm({3,3,3,3});
  op.set_attr_transpose_first(false);
  auto ret = op.InferShapeAndType();
  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}

TEST_F(confusionTransposeD_test, data_slice_infer_nd_reshape_transpose_1) {
  ge::op::ConfusionTransposeD op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{4, 4}, {8, 8}, {12, 12},{24, 24}, {14, 14}};
  auto tensor_desc = create_desc_shape_range({4, 8, 12, 24, 14},
                                             ge::DT_FLOAT16,
                                             ge::FORMAT_NCHW,
                                             {4, 8, 12, 24, 14},
                                             ge::FORMAT_NCHW,
                                             shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("shape", {2, 2, 1, 9, 64, 56});
  op.SetAttr("perm", {2, 5, 0, 3, 1, 4});
  op.SetAttr("transpose_first", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("y");

  std::vector<std::vector<int64_t>> y_data_slice = {{0, 0}, {0, 55}, {1, 1}, {0, 8}, {0, 1}, {0, 63}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x = op_desc->MutableInputDesc("x");
  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
  std::vector<std::vector<int64_t>> excepted_x_data_slice = {{2, 3}, {0, 7}, {0, 11}, {0, 23}, {0, 13}};
  EXPECT_EQ(excepted_x_data_slice, x_data_slice);
}

TEST_F(confusionTransposeD_test, data_slice_infer_nd_transpose_reshape_2) {
  ge::op::ConfusionTransposeD op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{4, 4}, {8, 8}, {12, 12},{24, 24}, {14, 14}};
  auto tensor_desc = create_desc_shape_range({4, 8, 12, 24, 14},
                                             ge::DT_FLOAT16,
                                             ge::FORMAT_NCHW,
                                             {4, 8, 12, 24, 14},
                                             ge::FORMAT_NCHW,
                                             shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("shape", {8, 9, 4, 1, 2, 2, 7, 16});
  op.SetAttr("perm", {2, 3, 0, 1, 4});
  op.SetAttr("transpose_first", true);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("y");

  std::vector<std::vector<int64_t>> y_data_slice = {{}, {}, {}, {}, {1, 1}, {}, {}, {}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x = op_desc->MutableInputDesc("x");
  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
  std::vector<std::vector<int64_t>> excepted_x_data_slice = {{2, 3}, {0, 7}, {0, 11}, {0, 23}, {0, 13}};
  EXPECT_EQ(excepted_x_data_slice, x_data_slice);
}

TEST_F(confusionTransposeD_test, data_slice_infer_nz_3_4_reshape_transpose_3) {
  ge::op::ConfusionTransposeD op;
  std::vector<std::pair<int64_t, int64_t>> shape_x_range = {{32, 32}, {4, 4}, {48, 48}, {16, 16}, {16, 16}};
  std::vector<std::pair<int64_t, int64_t>> shape_y_range = {{16, 16}, {32, 32}, {4, 4}, {3, 3}, {16, 16}, {16, 16}};
  auto x_desc = create_desc_shape_range({32, 4, 48, 16, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {32, 768, 64},
                                         ge::FORMAT_NCHW,
                                         shape_x_range);
  auto y_desc = create_desc_shape_range({16, 32, 4, 3, 16, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {16, 32, 48, 64},
                                         ge::FORMAT_NCHW,
                                         shape_y_range);
  op.UpdateInputDesc("x", x_desc);
  op.SetAttr("shape", {32, 48, 16, 64});
  op.SetAttr("perm", {2, 0, 1, 3});
  op.SetAttr("transpose_first", false);
  op.UpdateOutputDesc("y", y_desc);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("y");

  std::vector<std::vector<int64_t>> y_data_slice = {{}, {1, 17}, {}, {}, {}, {}, {}, {}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto tensor_desc_x = op_desc->MutableInputDesc("x");
  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
  std::vector<std::vector<int64_t>> excepted_x_data_slice = {{1, 17}, {0, 3}, {0, 47}, {0, 15}, {0, 15}};
  EXPECT_EQ(excepted_x_data_slice, x_data_slice);
}

TEST_F(confusionTransposeD_test, data_slice_infer_nz_3_4_reshape_transpose_4) {
  ge::op::ConfusionTransposeD op;
  std::vector<std::pair<int64_t, int64_t>> shape_x_range = {{32, 32}, {288, 288}, {4, 4}, {16, 16}, {16, 16}};
  std::vector<std::pair<int64_t, int64_t>> shape_y_range = {{64, 64}, {96, 96}, {3, 3}, {2, 2}, {16, 16}, {16, 16}};
  auto x_desc = create_desc_shape_range({32, 288, 4, 16, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {32, 64, 4608},
                                         ge::FORMAT_NCHW,
                                         shape_x_range);
  auto y_desc = create_desc_shape_range({64, 96, 3, 2, 16, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {64, 96, 32, 48},
                                         ge::FORMAT_NCHW,
                                         shape_y_range);
  op.UpdateInputDesc("x", x_desc);
  op.SetAttr("shape", {32, 64, 96, 48});
  op.SetAttr("perm", {1, 2, 0, 3});
  op.SetAttr("transpose_first", false);
  op.UpdateOutputDesc("y", y_desc);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("y");

  std::vector<std::vector<int64_t>> y_data_slice = {{}, {}, {}, {1, 1}, {}, {}, {}, {}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x = op_desc->MutableInputDesc("x");
  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
  std::vector<std::vector<int64_t>> excepted_x_data_slice = {{16, 31}, {0, 287}, {0, 3}, {0, 15}, {0, 15}};
  EXPECT_EQ(excepted_x_data_slice, x_data_slice);
}

TEST_F(confusionTransposeD_test, data_slice_infer_nz_2_4_reshape_transpose_5) {
  ge::op::ConfusionTransposeD op;
  std::vector<std::pair<int64_t, int64_t>> shape_x_range = {{288, 288}, {96, 96}, {16, 16}, {16, 16}};
  std::vector<std::pair<int64_t, int64_t>> shape_y_range = {{48, 48}, {16, 16}, {18, 18}, {2, 2}, {16, 16}, {16, 16}};
  auto x_desc = create_desc_shape_range({288, 96, 16, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {1536, 4608},
                                         ge::FORMAT_NCHW,
                                         shape_x_range);
  auto y_desc = create_desc_shape_range({48, 16, 18, 2, 16, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {48, 16, 32, 288},
                                         ge::FORMAT_NCHW,
                                         shape_y_range);
  op.UpdateInputDesc("x", x_desc);
  op.SetAttr("shape", {32, 48, 16, 288});
  op.SetAttr("perm", {1, 2, 0, 3});
  op.SetAttr("transpose_first", false);
  op.UpdateOutputDesc("y", y_desc);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("y");

  std::vector<std::vector<int64_t>> y_data_slice = {{}, {}, {}, {1, 1}, {}, {}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x = op_desc->MutableInputDesc("x");
  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
  std::vector<std::vector<int64_t>> excepted_x_data_slice = {{0, 287}, {48, 95}, {0, 15}, {0, 15}};
  EXPECT_EQ(excepted_x_data_slice, x_data_slice);
}

TEST_F(confusionTransposeD_test, data_slice_infer_nz_4_2_transpose_reshape_6) {
  ge::op::ConfusionTransposeD op;
  std::vector<std::pair<int64_t, int64_t>> shape_x_range = {{48, 48}, {16, 16}, {18, 18}, {2, 2}, {16, 16}, {16, 16}};
  std::vector<std::pair<int64_t, int64_t>> shape_y_range = {{864, 864}, {32, 32}, {16, 16}, {16, 16}};
  auto x_desc = create_desc_shape_range({48, 16, 18, 2, 16, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {48, 16, 32, 288},
                                         ge::FORMAT_NCHW,
                                         shape_x_range);
  auto y_desc = create_desc_shape_range({864, 32, 16, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {512, 13824},
                                         ge::FORMAT_NCHW,
                                         shape_y_range);
  op.UpdateInputDesc("x", x_desc);
  op.SetAttr("shape", {512, 13824});
  op.SetAttr("perm", {1, 2, 0, 3});
  op.SetAttr("transpose_first", true);
  op.UpdateOutputDesc("y", y_desc);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("y");

  std::vector<std::vector<int64_t>> y_data_slice = {{288, 575}, {0, 31}, {0, 15}, {0,15}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x = op_desc->MutableInputDesc("x");
  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
  std::vector<std::vector<int64_t>> excepted_x_data_slice = {{16, 31}, {0, 15}, {0, 17}, {0, 1}, {0, 15}, {0, 15}};
  EXPECT_EQ(excepted_x_data_slice, x_data_slice);
}

TEST_F(confusionTransposeD_test, data_slice_infer_nz_4_3_transpose_reshape_7) {
  ge::op::ConfusionTransposeD op;
  std::vector<std::pair<int64_t, int64_t>> shape_x_range = {{64, 64}, {32, 32}, {4, 4}, {3, 3}, {16, 16}, {16, 16}};
  std::vector<std::pair<int64_t, int64_t>> shape_y_range = {{32, 32}, {4, 4}, {192, 192}, {16, 16}, {16, 16}};
  auto x_desc = create_desc_shape_range({64, 32, 4, 3, 16, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {64, 32, 48, 64},
                                         ge::FORMAT_NCHW,
                                         shape_x_range);
  auto y_desc = create_desc_shape_range({32, 4, 192, 16, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {32, 3072, 64},
                                         ge::FORMAT_NCHW,
                                         shape_y_range);
  op.UpdateInputDesc("x", x_desc);
  op.SetAttr("shape", {32, 3072, 64});
  op.SetAttr("perm", {1, 0, 2, 3});
  op.SetAttr("transpose_first", true);
  op.UpdateOutputDesc("y", y_desc);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("y");

  std::vector<std::vector<int64_t>> y_data_slice = {{}, {}, {45, 89}, {}, {}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x = op_desc->MutableInputDesc("x");
  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
  std::vector<std::vector<int64_t>> excepted_x_data_slice = {{15, 29}, {0, 31}, {0, 3}, {0, 2}, {0, 15}, {0, 15}};
  EXPECT_EQ(excepted_x_data_slice, x_data_slice);
}

TEST_F(confusionTransposeD_test, data_slice_infer_nz_4_3_transpose_reshape_8) {
  ge::op::ConfusionTransposeD op;
  std::vector<std::pair<int64_t, int64_t>> shape_x_range = {{64, 64}, {96, 96}, {3, 3}, {2, 2}, {16, 16}, {16, 16}};
  std::vector<std::pair<int64_t, int64_t>> shape_y_range = {{32, 32}, {288, 288}, {4, 4}, {16, 16}, {16, 16}};
  auto x_desc = create_desc_shape_range({64, 96, 3, 2, 16, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {64, 96, 32, 48},
                                         ge::FORMAT_NCHW,
                                         shape_x_range);
  auto y_desc = create_desc_shape_range({32, 288, 4, 16, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {32, 64, 4608},
                                         ge::FORMAT_NCHW,
                                         shape_y_range);
  op.UpdateInputDesc("x", x_desc);
  op.SetAttr("shape", {32, 64, 4608});
  op.SetAttr("perm", {2, 0, 1, 3});
  op.SetAttr("transpose_first", true);
  op.UpdateOutputDesc("y", y_desc);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("y");

  std::vector<std::vector<int64_t>> y_data_slice = {{}, {}, {2, 3}, {}, {}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x = op_desc->MutableInputDesc("x");
  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
  std::vector<std::vector<int64_t>> excepted_x_data_slice = {{32, 63}, {0, 95}, {0, 2}, {0, 1}, {0, 15}, {0, 15}};
  EXPECT_EQ(excepted_x_data_slice, x_data_slice);
}

TEST_F(confusionTransposeD_test, data_slice_infer_nd_reshape_transpose_9) {
  ge::op::ConfusionTransposeD op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{4, 4}, {8, 8}, {12, 12},{24, 24}, {14, 14}};
  auto tensor_desc = create_desc_shape_range({4, 8, 12, 24, 14},
                                             ge::DT_FLOAT16,
                                             ge::FORMAT_NCHW,
                                             {4, 8, 12, 24, 14},
                                             ge::FORMAT_NCHW,
                                             shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("shape", {1, 2, 2, 1, 9, 64, 56});
  op.SetAttr("perm", {6, 2, 5, 0, 3, 1, 4});
  op.SetAttr("transpose_first", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("y");

  std::vector<std::vector<int64_t>> y_data_slice = {{}, {}, {}, {0, 0}, {}, {}, {}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, NOT_SUPPORT_SLICE);
}

TEST_F(confusionTransposeD_test, data_slice_infer_nd_reshape_transpose_10) {
  ge::op::ConfusionTransposeD op;
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{9, 9}, {8, 8}, {84, 84}, {24, 24}, {14, 14}};
  auto tensor_desc = create_desc_shape_range({9, 8, 84, 24, 14},
                                             ge::DT_FLOAT16,
                                             ge::FORMAT_NCHW,
                                             {9, 8, 84, 24, 14},
                                             ge::FORMAT_NCHW,
                                             shape_range);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("shape", {6, 3, 7, 4, 64, 63});
  op.SetAttr("perm", {2, 5, 0, 3, 1, 4});
  op.SetAttr("transpose_first", false);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("y");

  std::vector<std::vector<int64_t>> y_data_slice = {{0, 6}, {0, 62}, {1, 2}, {0, 3}, {0, 2}, {0, 63}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, NOT_SUPPORT_SLICE);
}

TEST_F(confusionTransposeD_test, data_slice_infer_nz_4_4_transpose_reshape_11) {
  ge::op::ConfusionTransposeD op;
  std::vector<std::pair<int64_t, int64_t>> shape_x_range = {{64, 64}, {96, 96}, {3, 3}, {2, 2}, {16, 16}, {16, 16}};
  std::vector<std::pair<int64_t, int64_t>> shape_y_range = {{32, 32}, {288, 288}, {4, 4}, {16, 16}, {16, 16}};
  auto x_desc = create_desc_shape_range({64, 96, 3, 2, 16, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {64, 96, 32, 48},
                                         ge::FORMAT_NCHW,
                                         shape_x_range);
  auto y_desc = create_desc_shape_range({96, 384, 16, 16},
                                         ge::DT_FLOAT16,
                                         ge::FORMAT_FRACTAL_NZ,
                                         {6144, 1536},
                                         ge::FORMAT_NCHW,
                                         shape_y_range);
  op.UpdateInputDesc("x", x_desc);
  op.SetAttr("shape", {6144, 1536});
  op.SetAttr("perm", {0, 1, 2, 3});
  op.SetAttr("transpose_first", true);
  op.UpdateOutputDesc("y", y_desc);

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc("y");

  std::vector<std::vector<int64_t>> y_data_slice = {{}, {6, 11}, {}, {}};
  ge::AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto tensor_desc_x = op_desc->MutableInputDesc("x");
  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
  std::vector<std::vector<int64_t>> excepted_x_data_slice = {{1, 1}, {0, 95}, {0, 2}, {0, 1}, {0, 15}, {0, 15}};
  EXPECT_EQ(excepted_x_data_slice, x_data_slice);
}
