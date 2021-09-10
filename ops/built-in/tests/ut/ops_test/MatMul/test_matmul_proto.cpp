#include <tuple>

#include "common/util/error_manager/error_manager.h"
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "matrix_calculation_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"
#include "op_log.h"
#include "op_desc.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

#define OP_TUPLE tuple<vector<int64_t>, DataType, ge::Format, vector<pair<int64_t,int64_t>>>

using namespace std;
using namespace ge;
using namespace op;

class matmul_infer_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "matmul_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "matmul_infer_test TearDown" << std::endl;
  }
};

TEST_F(matmul_infer_test, matmul_infer_test_1) {
  // input x1 shape {-1, -1}
  // input x1 range {1, -1} {1, -1}
  // input x2 shape {-1, -1}
  // input x2 range {1, -1} {1, -1}
  // output shape {-1, -1}
  // output shape {1, -1} {1, -1}
  ge::Graph graph("matmul_infer_test_1");
  auto shape_x1 = vector<int64_t>({-1, -1});
  TensorDesc desc_data_x1(ge::Shape(shape_x1), FORMAT_NCHW, DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> x1_range;
  x1_range.push_back(std::pair<int64_t, int64_t>{1, -1});
  x1_range.push_back(std::pair<int64_t, int64_t>{1, -1});
  desc_data_x1.SetShapeRange(x1_range);

  auto data_x1 = op::Data("x1");
  data_x1.update_input_desc_x(desc_data_x1);
  data_x1.update_output_desc_y(desc_data_x1);

  auto shape_x2 = vector<int64_t>({-1, -1});
  TensorDesc desc_data_x2(ge::Shape(shape_x2), FORMAT_NCHW, DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> x2_range;
  x2_range.push_back(std::pair<int64_t, int64_t>{1, -1});
  x2_range.push_back(std::pair<int64_t, int64_t>{1, -1});
  desc_data_x2.SetShapeRange(x2_range);

  auto data_x2 = op::Data("x2");
  data_x2.update_input_desc_x(desc_data_x2);
  data_x2.update_output_desc_y(desc_data_x2);

  // new attr value
  bool is_transpose_x1 = false;
  bool is_transpose_x2 = false;
  // test op
  auto matmulOp = op::MatMul("MatMul")
                   .set_input_x1(data_x1)
                   .set_input_x2(data_x2)
                   .set_attr_transpose_x1(is_transpose_x1)
                   .set_attr_transpose_x2(is_transpose_x2);

  std::vector<Operator> inputs{data_x1, data_x2};
  std::vector<Operator> outputs{matmulOp};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  matmulOp.InferShapeAndType();

  bool findOp = false;
  bool shapeMatch = false;
  bool shapeRangeMatch = false;

  // set expect_shape
  auto expect_shape = vector<int64_t>({-1, -1});
  // set expect_range
  std::vector<std::pair<int64_t, int64_t>> expect_range;
  expect_range.push_back(std::pair<int64_t, int64_t>{1, -1});
  expect_range.push_back(std::pair<int64_t, int64_t>{1, -1});

  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "MatMul") {
      findOp = true;
      auto outputDesc = node->GetOpDesc()->GetOutputDesc(0);
      std::vector<int64_t> output_shape = outputDesc.GetShape().GetDims();
      std::vector<std::pair<int64_t, int64_t>> output_range;
      outputDesc.GetShapeRange(output_range);
      if (output_shape == expect_shape) {
          shapeMatch = true;
      }
      if (output_range == expect_range) {
          shapeRangeMatch = true;
      }
    }
  }
  EXPECT_EQ(findOp, true);
  EXPECT_EQ(shapeMatch, true);
  EXPECT_EQ(shapeRangeMatch, true);
}

TEST_F(matmul_infer_test, matmul_infer_test_2) {
  // input x1 shape {-1, 5}
  // input x1 range {{1, 60}, {5, 5}}
  // input x2 shape {6, -1}
  // input x2 range {{6, 6}, {1, 60}}
  // attr  true  true
  // output shape  {5, 6}
  // output shape {{5, 5}, {6, 6}}
  // set input info
  auto shape_x1 = vector<int64_t>({-1, 5});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{1, 60}, {5, 5}};
  auto shape_x2 = vector<int64_t>({6, -1});
  std::vector<std::pair<int64_t,int64_t>> range_x2 = {{6, 6}, {1, 60}};
  bool transpose_x1 = true;
  bool transpose_x2 = true;

  // expect result
  std::vector<int64_t> expected_shape = {5, 6};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {};
  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                shape_x1, ge::FORMAT_ND, range_x1);
  auto tensor_desc_x2 = create_desc_shape_range(shape_x2, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                shape_x2, ge::FORMAT_ND, range_x2);
  // new op and do infershape
  ge::op::MatMul op;
  op.UpdateInputDesc("x1", tensor_desc_x1);
  op.UpdateInputDesc("x2", tensor_desc_x2);
  op.SetAttr("transpose_x1", transpose_x1);
  op.SetAttr("transpose_x2", transpose_x2);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

ge::op::MatMul CreateMatMulOp(OP_TUPLE a, OP_TUPLE b,
                              bool trans_a, bool trans_b) {
  auto tensor_a = create_desc_shape_range(get<0>(a), get<1>(a), get<2>(a), get<0>(a), get<2>(a), get<3>(a));
  auto tensor_b = create_desc_shape_range(get<0>(b), get<1>(b), get<2>(b), get<0>(b), get<2>(b), get<3>(b));
  ge::op::MatMul op;
  op.UpdateInputDesc("x1", tensor_a);
  op.UpdateInputDesc("x2", tensor_b);
  op.SetAttr("transpose_x1", trans_a);
  op.SetAttr("transpose_x2", trans_b);
  return op;
}

void Operate(ge::op::MatMul &op) {
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

void Check(ge::op::MatMul &op, vector<int64_t> expected_shape, vector<pair<int64_t,int64_t>> expected_range) {
  auto output_desc = op.GetOutputDesc("y");

  auto shape = output_desc.GetShape().GetDims();
  vector<pair<int64_t,int64_t>> range;
  EXPECT_EQ(output_desc.GetShapeRange(range), ge::GRAPH_SUCCESS);

  EXPECT_EQ(shape, expected_shape);
  EXPECT_EQ(range, expected_range);
}

TEST_F(matmul_infer_test, static_normal) {
  auto op = CreateMatMulOp(OP_TUPLE{{2, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {}},
                           OP_TUPLE{{4, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {}},
                           false, false);

  Operate(op);

  Check(op, {2, 5}, {});
}

TEST_F(matmul_infer_test, static_normal_2) {
  auto op = CreateMatMulOp(OP_TUPLE{{2, 4}, ge::DT_FLOAT16, ge::FORMAT_ND, {}},
                           OP_TUPLE{{4}, ge::DT_FLOAT16, ge::FORMAT_ND, {}},
                           false, false);

  Operate(op);

  Check(op, {2, 1}, {});
}

TEST_F(matmul_infer_test, static_normal_3) {
  auto op = CreateMatMulOp(OP_TUPLE{{4}, ge::DT_FLOAT16, ge::FORMAT_ND, {}},
                           OP_TUPLE{{4, 5}, ge::DT_FLOAT16, ge::FORMAT_ND, {}},
                           false, false);

  Operate(op);

  Check(op, {1, 5}, {});
}

TEST_F(matmul_infer_test, dynamic_normal) {
  auto op = CreateMatMulOp(OP_TUPLE{{-1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {{1, 3}, {2, 5}}},
                           OP_TUPLE{{-1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND, {{1, 7}, {2, 9}}},
                           false, false);

  Operate(op);

  Check(op, {-1, -1}, {{1, 3}, {2, 9}});
}

TEST_F(matmul_infer_test, dynamic_normal_2) {
  auto op = CreateMatMulOp(OP_TUPLE{{-2}, ge::DT_FLOAT16, ge::FORMAT_ND, {}},
                           OP_TUPLE{{-2}, ge::DT_FLOAT16, ge::FORMAT_ND, {}},
                           false, false);

  Operate(op);

  Check(op, {-2}, {});
}

// cut n in NZ
TEST_F(matmul_infer_test, split_test0) {
  ge::op::MatMul op;
  op.UpdateInputDesc("x1", create_desc_with_ori({4, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{32, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("x2", create_desc_with_ori({4, 4, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{64, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("y", create_desc_with_ori({4, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{32, 64}, ge::FORMAT_ND));
  op.SetAttr("transpose_x1", false);
  op.SetAttr("transpose_x2", false);

  std::vector<std::vector<int64_t>> y_data_slice ={{0, 1}, {}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();

  ge::GeTensorDescPtr tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  std::vector<std::vector<int64_t>> x2_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice);

  std::vector<std::vector<int64_t>> expect_x2_data_slice = {{0, 1}, {}, {}, {}};
  EXPECT_EQ(expect_x2_data_slice, x2_data_slice);
}

// cut m in NZ
TEST_F(matmul_infer_test, split_test1) {
  ge::op::MatMul op;
  op.UpdateInputDesc("x1", create_desc_with_ori({4, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{32, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("x2", create_desc_with_ori({4, 4, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{64, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("y", create_desc_with_ori({4, 2, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,{32, 64}, ge::FORMAT_ND));
  op.SetAttr("transpose_x1", false);
  op.SetAttr("transpose_x2", false);

  std::vector<std::vector<int64_t>> y_data_slice ={{}, {0, 1}, {}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();

  ge::GeTensorDescPtr tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  std::vector<std::vector<int64_t>> x1_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice);

  std::vector<std::vector<int64_t>> expect_x1_data_slice = {{}, {0, 1}, {}, {}};
  EXPECT_EQ(expect_x1_data_slice, x1_data_slice);
}

// cut n in ND
TEST_F(matmul_infer_test, split_test2) {
  ge::op::MatMul op;
  op.UpdateInputDesc("x1", create_desc_with_ori({32, 16}, ge::DT_FLOAT16, ge::FORMAT_ND,{32, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("x2", create_desc_with_ori({64, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,{64, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("y", create_desc_with_ori({32, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,{32, 64}, ge::FORMAT_ND));
  op.SetAttr("transpose_x1", false);
  op.SetAttr("transpose_x2", false);

  std::vector<std::vector<int64_t>> y_data_slice ={{}, {0, 15}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();

  ge::GeTensorDescPtr tensor_desc_x2 = op_desc->MutableInputDesc("x2");
  std::vector<std::vector<int64_t>> x2_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x2, ge::ATTR_NAME_DATA_SLICE, x2_data_slice);

  std::vector<std::vector<int64_t>> expect_x2_data_slice = {{}, {0, 15}};
  EXPECT_EQ(expect_x2_data_slice, x2_data_slice);
}

// cut m in ND
TEST_F(matmul_infer_test, split_test3) {
  ge::op::MatMul op;
  op.UpdateInputDesc("x1", create_desc_with_ori({32, 16}, ge::DT_FLOAT16, ge::FORMAT_ND,{32, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("x2", create_desc_with_ori({64, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,{64, 64}, ge::FORMAT_ND));
  op.UpdateInputDesc("y", create_desc_with_ori({32, 64}, ge::DT_FLOAT16, ge::FORMAT_ND,{32, 64}, ge::FORMAT_ND));
  op.SetAttr("transpose_x1", false);
  op.SetAttr("transpose_x2", false);

  std::vector<std::vector<int64_t>> y_data_slice ={{16, 31}, {}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();

  ge::GeTensorDescPtr tensor_desc_x1 = op_desc->MutableInputDesc("x1");
  std::vector<std::vector<int64_t>> x1_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x1, ge::ATTR_NAME_DATA_SLICE, x1_data_slice);

  std::vector<std::vector<int64_t>> expect_x1_data_slice = {{16, 31}, {}};
  EXPECT_EQ(expect_x1_data_slice, x1_data_slice);
}

TEST_F(matmul_infer_test, supportcheckerror1) {
  auto shape_x1 = vector<int64_t>({32, -1});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {};
  auto shape_x2 = vector<int64_t>({-1, 64});
  std::vector<std::pair<int64_t,int64_t>> range_x2 = {};
  bool transpose_x1 = false;
  bool transpose_x2 = false;

  // expect result
  std::vector<int64_t> expected_shape = {32, 64};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {};

  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                shape_x1, ge::FORMAT_ND, range_x1);
  auto tensor_desc_x2 = create_desc_shape_range(shape_x2, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                shape_x2, ge::FORMAT_ND, range_x2);
  // new op and do infershape
  ge::op::MatMul op;
  op.UpdateInputDesc("x1", tensor_desc_x1);
  op.UpdateInputDesc("x2", tensor_desc_x2);
  op.SetAttr("transpose_x1", transpose_x1);
  op.SetAttr("transpose_x2", transpose_x2);
  op.SetAttr("_fuzz_build", true);
  auto verify_ret = op.VerifyAllAttr(true);
  auto infer_ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(verify_ret, GRAPH_SUCCESS);
  EXPECT_EQ(infer_ret, GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

TEST_F(matmul_infer_test, supportcheckerror2) {
  auto shape_x1 = vector<int64_t>({32, 32});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {};
  auto shape_x2 = vector<int64_t>({64, 64});
  std::vector<std::pair<int64_t,int64_t>> range_x2 = {};
  bool transpose_x1 = false;
  bool transpose_x2 = false;
  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                shape_x1, ge::FORMAT_ND, range_x1);
  auto tensor_desc_x2 = create_desc_shape_range(shape_x2, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                shape_x2, ge::FORMAT_ND, range_x2);
  // new op and do infershape
  ge::op::MatMul op;
  op.UpdateInputDesc("x1", tensor_desc_x1);
  op.UpdateInputDesc("x2", tensor_desc_x2);
  op.SetAttr("transpose_x1", transpose_x1);
  op.SetAttr("transpose_x2", transpose_x2);
  op.SetAttr("_fuzz_build", true);
  auto verify_ret = op.VerifyAllAttr(true);
  auto infer_ret = op.InferShapeAndType();
  auto ret = (verify_ret == GRAPH_FAILED || infer_ret == GRAPH_FAILED) ? GRAPH_FAILED : GRAPH_SUCCESS;

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(matmul_infer_test, supportcheckerror3) {
  auto shape_x1 = vector<int64_t>({32, -1});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{2, 3}};
  auto shape_x2 = vector<int64_t>({-1, 64});
  std::vector<std::pair<int64_t,int64_t>> range_x2 = {{4, 5}};
  bool transpose_x1 = false;
  bool transpose_x2 = false;
  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                shape_x1, ge::FORMAT_ND, range_x1);
  auto tensor_desc_x2 = create_desc_shape_range(shape_x2, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                shape_x2, ge::FORMAT_ND, range_x2);
  // new op and do infershape
  ge::op::MatMul op;
  op.UpdateInputDesc("x1", tensor_desc_x1);
  op.UpdateInputDesc("x2", tensor_desc_x2);
  op.SetAttr("transpose_x1", transpose_x1);
  op.SetAttr("transpose_x2", transpose_x2);
  op.SetAttr("_fuzz_build", true);
  auto verify_ret = op.VerifyAllAttr(true);
  auto infer_ret = op.InferShapeAndType();
  auto ret = (verify_ret == GRAPH_FAILED || infer_ret == GRAPH_FAILED) ? GRAPH_FAILED : GRAPH_SUCCESS;

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(matmul_infer_test, supportcheckerror4) {
  auto shape_x1 = vector<int64_t>({32, 32});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {};
  auto shape_x2 = vector<int64_t>({-1, 64});
  std::vector<std::pair<int64_t,int64_t>> range_x2 = {{4, 5}};
  bool transpose_x1 = false;
  bool transpose_x2 = false;
  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                shape_x1, ge::FORMAT_ND, range_x1);
  auto tensor_desc_x2 = create_desc_shape_range(shape_x2, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                shape_x2, ge::FORMAT_ND, range_x2);
  // new op and do infershape
  ge::op::MatMul op;
  op.UpdateInputDesc("x1", tensor_desc_x1);
  op.UpdateInputDesc("x2", tensor_desc_x2);
  op.SetAttr("transpose_x1", transpose_x1);
  op.SetAttr("transpose_x2", transpose_x2);
  op.SetAttr("_fuzz_build", true);
  auto verify_ret = op.VerifyAllAttr(true);
  auto infer_ret = op.InferShapeAndType();
  auto ret = (verify_ret == GRAPH_FAILED || infer_ret == GRAPH_FAILED) ? GRAPH_FAILED : GRAPH_SUCCESS;

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(matmul_infer_test, supportcheckerror5) {
  auto shape_x1 = vector<int64_t>({32, -1});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{2, 3}};
  auto shape_x2 = vector<int64_t>({64, 64});
  std::vector<std::pair<int64_t,int64_t>> range_x2 = {};
  bool transpose_x1 = false;
  bool transpose_x2 = false;
  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                shape_x1, ge::FORMAT_ND, range_x1);
  auto tensor_desc_x2 = create_desc_shape_range(shape_x2, ge::DT_FLOAT16, ge::FORMAT_ND,
                                                shape_x2, ge::FORMAT_ND, range_x2);
  // new op and do infershape
  ge::op::MatMul op;
  op.UpdateInputDesc("x1", tensor_desc_x1);
  op.UpdateInputDesc("x2", tensor_desc_x2);
  op.SetAttr("transpose_x1", transpose_x1);
  op.SetAttr("transpose_x2", transpose_x2);
  op.SetAttr("_fuzz_build", true);
  auto verify_ret = op.VerifyAllAttr(true);
  auto infer_ret = op.InferShapeAndType();
  auto ret = (verify_ret == GRAPH_FAILED || infer_ret == GRAPH_FAILED) ? GRAPH_FAILED : GRAPH_SUCCESS;

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}