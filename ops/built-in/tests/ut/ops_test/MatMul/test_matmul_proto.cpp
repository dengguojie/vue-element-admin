#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "matrix_calculation_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

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
  // input x2 range {{6, 6}, {5, 90}}
  // attr  true  true
  // output shape  {5, 6}
  // output shape {{5, 5}, {6, 6}}
  // set input info
  auto shape_x1 = vector<int64_t>({-1, 5});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{1, 60}, {5, 5}};
  auto shape_x2 = vector<int64_t>({6, -1});
  std::vector<std::pair<int64_t,int64_t>> range_x2 = {{6, 6}, {5, 90}};
  bool transpose_x1 = true;
  bool transpose_x2 = true;

  // expect result
  std::vector<int64_t> expected_shape = {5, 6};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{5, 5}, {6, 6}};
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
