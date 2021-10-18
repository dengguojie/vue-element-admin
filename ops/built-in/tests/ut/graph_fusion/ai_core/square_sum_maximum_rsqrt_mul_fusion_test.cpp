#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "reduce_ops.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class square_sum_maximum_rsqrt_mul_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "square_sum_maximum_rsqrt_mul_fusion SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "square_sum_maximum_rsqrt_mul_fusion TearDown" << std::endl;
    }
};

TEST_F(square_sum_maximum_rsqrt_mul_fusion_test, square_sum_maximum_rsqrt_mul_fusion_test_1) {
    ge::Graph graph("square_sum_maximum_rsqrt_mul_fusion_test_1");

    auto input_x = op::Data("input_x");

    std::vector<int64_t> dims_x{32, 32};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, ge::FORMAT_NHWC, ge::DT_FLOAT16);
    input_x.update_input_desc_x(tensorDescX);
    input_x.update_output_desc_y(tensorDescX);

    auto sum_const = op::Constant("sum_const");
    Tensor sum_consttensor;
    int32_t * dataValue1 = new int32_t[1];
    * dataValue1 = 1;
    sum_consttensor.SetTensorDesc(TensorDesc(ge::Shape({1}), ge::FORMAT_NHWC, ge::DT_INT32));
    sum_consttensor.SetData((uint8_t*)dataValue1, 1 * sizeof(int32_t));
    sum_const.set_attr_value(sum_consttensor);
    delete []dataValue1;

    auto maximum_const = op::Constant("maximum_const");
    Tensor maximum_consttensor;
    float * dataValue2 = new float[1];
    * dataValue2 = 0.1;
    maximum_consttensor.SetTensorDesc(TensorDesc(ge::Shape({1}), ge::FORMAT_NHWC, ge::DT_FLOAT16));
    maximum_consttensor.SetData((uint8_t*)dataValue2, 4);
    maximum_const.set_attr_value(maximum_consttensor);
    delete []dataValue2;

    auto square_op = op::Square("square_0");
    square_op.set_input_x(input_x);

    auto sum_op = op::ReduceSum("sum_0");
    sum_op.set_input_x(square_op)
          .set_input_axes(sum_const);

    auto maximum_op = op::Maximum("maximum_0");
    maximum_op.set_input_x1(sum_op)
              .set_input_x2(maximum_const);

    auto rsqrt_op = op::Rsqrt("rsqrt_0");
    rsqrt_op.set_input_x(maximum_op);

    auto mul_op = op::Mul("mul_0");
    mul_op.set_input_x1(input_x)
          .set_input_x2(rsqrt_op);

    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(mul_op);

    std::vector<Operator> inputs{input_x, sum_const, maximum_const};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "square_sum_maximum_rsqrt_mul_fusion_test_1_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("AASquareSumMaximumRsqrtMulFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "square_sum_maximum_rsqrt_mul_fusion_test_1_after");
    bool findL2Normalize = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Mul") {
            findL2Normalize = true;
        }
    }
    EXPECT_EQ(findL2Normalize, true);
}

TEST_F(square_sum_maximum_rsqrt_mul_fusion_test, square_sum_maximum_rsqrt_mul_fusion_test_2) {
    ge::Graph graph("square_sum_maximum_rsqrt_mul_fusion_test_2");

    auto input_x = op::Data("input_x");

    std::vector<int64_t> dims_x{16, 16};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, ge::FORMAT_NHWC, ge::DT_FLOAT);
    input_x.update_input_desc_x(tensorDescX);
    input_x.update_output_desc_y(tensorDescX);

    auto sum_const = op::Constant("sum_const");
    Tensor sum_consttensor;
    int64_t * dataValue1 = new int64_t[1];
    * dataValue1 = 1;
    sum_consttensor.SetTensorDesc(TensorDesc(ge::Shape({1}), ge::FORMAT_NHWC, ge::DT_INT64));
    sum_consttensor.SetData((uint8_t*)dataValue1, 1 * sizeof(int64_t));
    sum_const.set_attr_value(sum_consttensor);
    delete []dataValue1;

    auto maximum_const = op::Constant("maximum_const");
    Tensor maximum_consttensor;
    float * dataValue2 = new float[1];
    * dataValue2 = 0.1;
    maximum_consttensor.SetTensorDesc(TensorDesc(ge::Shape({1}), ge::FORMAT_NHWC, ge::DT_FLOAT));
    maximum_consttensor.SetData((uint8_t*)dataValue2, 4);
    maximum_const.set_attr_value(maximum_consttensor);
    delete []dataValue2;

    auto square_op = op::Square("square_0");
    square_op.set_input_x(input_x);

    auto sum_op = op::ReduceSum("sum_0");
    sum_op.set_input_x(square_op)
          .set_input_axes(sum_const);

    auto maximum_op = op::Maximum("maximum_0");
    maximum_op.set_input_x1(sum_op)
              .set_input_x2(maximum_const);

    auto rsqrt_op = op::Rsqrt("rsqrt_0");
    rsqrt_op.set_input_x(maximum_op);

    auto mul_op = op::Mul("mul_0");
    mul_op.set_input_x1(input_x)
          .set_input_x2(rsqrt_op);

    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(mul_op);

    std::vector<Operator> inputs{input_x, sum_const, maximum_const};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "square_sum_maximum_rsqrt_mul_fusion_test_2_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("AASquareSumMaximumRsqrtMulFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "square_sum_maximum_rsqrt_mul_fusion_test_2_after");
    bool findL2Normalize = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Mul") {
            findL2Normalize = true;
        }
    }
    EXPECT_EQ(findL2Normalize, true);
}

TEST_F(square_sum_maximum_rsqrt_mul_fusion_test, square_sum_maximum_rsqrt_mul_fusion_test_3) {
    ge::Graph graph("square_sum_maximum_rsqrt_mul_fusion_test_3");

    auto input_x = op::Data("input_x");

    std::vector<int64_t> dims_x{16, 16};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, ge::FORMAT_NHWC, ge::DT_FLOAT);
    input_x.update_input_desc_x(tensorDescX);
    input_x.update_output_desc_y(tensorDescX);

    auto sum_const = op::Constant("sum_const");
    Tensor sum_consttensor;
    int64_t * dataValue1 = new int64_t[1];
    * dataValue1 = 1;
    sum_consttensor.SetTensorDesc(TensorDesc(ge::Shape({1}), ge::FORMAT_NHWC, ge::DT_INT64));
    sum_consttensor.SetData((uint8_t*)dataValue1, 1 * sizeof(int64_t));
    sum_const.set_attr_value(sum_consttensor);
    delete []dataValue1;

    auto maximum_const = op::Const("maximum_const");
    Tensor maximum_consttensor;
    float * dataValue2 = new float[1];
    * dataValue2 = 0.1;
    maximum_consttensor.SetTensorDesc(TensorDesc(ge::Shape({1}), ge::FORMAT_NHWC, ge::DT_FLOAT));
    maximum_consttensor.SetData((uint8_t*)dataValue2, 4);
    maximum_const.set_attr_value(maximum_consttensor);
    delete []dataValue2;

    auto square_op = op::Square("square_0");
    square_op.set_input_x(input_x);

    auto sum_op = op::ReduceSum("sum_0");
    sum_op.set_input_x(square_op)
          .set_input_axes(sum_const);

    auto maximum_op = op::Maximum("maximum_0");
    maximum_op.set_input_x1(maximum_const)
              .set_input_x2(sum_op);

    auto rsqrt_op = op::Rsqrt("rsqrt_0");
    rsqrt_op.set_input_x(maximum_op);

    auto mul_op = op::Mul("mul_0");
    mul_op.set_input_x1(input_x)
          .set_input_x2(rsqrt_op);

    auto end_op = op::Square("end_op_0");
    end_op.set_input_x(mul_op);

    std::vector<Operator> inputs{input_x, sum_const, maximum_const};
    std::vector<Operator> outputs{end_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "square_sum_maximum_rsqrt_mul_fusion_test_2_before");
    fe::FusionPassTestUtils::RunGraphFusionPass("AASquareSumMaximumRsqrtMulFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
    //    GE_DUMP(compute_graph_ptr, "square_sum_maximum_rsqrt_mul_fusion_test_2_after");
    bool findL2Normalize = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Mul") {
            findL2Normalize = true;
        }
    }
    EXPECT_EQ(findL2Normalize, true);
}
TEST_F(square_sum_maximum_rsqrt_mul_fusion_test, square_sum_maximum_rsqrt_mul_fusion_test_4) {
  ge::Graph graph("square_sum_maximum_rsqrt_mul_fusion_test_4");

  auto input_x = op::Data("input_x");

  std::vector<int64_t> dims_x{16, 16};
  ge::Shape shape_x(dims_x);
  ge::TensorDesc tensorDescX(shape_x, ge::FORMAT_NHWC, ge::DT_FLOAT);
  input_x.update_input_desc_x(tensorDescX);
  input_x.update_output_desc_y(tensorDescX);

  auto sum_const = op::Constant("sum_const");
  Tensor sum_consttensor;
  int64_t* dataValue1 = new int64_t[1];
  *dataValue1 = 1;
  sum_consttensor.SetTensorDesc(TensorDesc(ge::Shape({1}), ge::FORMAT_NHWC, ge::DT_INT64));
  sum_consttensor.SetData((uint8_t*)dataValue1, 1 * sizeof(int64_t));
  sum_const.set_attr_value(sum_consttensor);
  delete[] dataValue1;

  auto maximum_const = op::Const("maximum_const");
  Tensor maximum_consttensor;
  float* dataValue2 = new float[1];
  *dataValue2 = 0.1;
  maximum_consttensor.SetTensorDesc(TensorDesc(ge::Shape(), ge::FORMAT_NHWC, ge::DT_FLOAT));
  maximum_consttensor.SetData((uint8_t*)dataValue2, 4);
  maximum_const.set_attr_value(maximum_consttensor);
  delete[] dataValue2;

  auto square_op = op::Square("square_0");
  square_op.set_input_x(input_x);

  auto sum_op = op::ReduceSum("sum_0");
  sum_op.set_input_x(square_op).set_input_axes(sum_const);

  auto maximum_op = op::Maximum("maximum_0");
  maximum_op.set_input_x1(maximum_const).set_input_x2(sum_op);

  auto rsqrt_op = op::Rsqrt("rsqrt_0");
  rsqrt_op.set_input_x(maximum_op);

  auto mul_op = op::Mul("mul_0");
  mul_op.set_input_x1(input_x).set_input_x2(rsqrt_op);

  auto end_op = op::Square("end_op_0");
  end_op.set_input_x(mul_op);

  std::vector<Operator> inputs{input_x, sum_const, maximum_const};
  std::vector<Operator> outputs{end_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  //    GE_DUMP(compute_graph_ptr, "square_sum_maximum_rsqrt_mul_fusion_test_2_before");
  fe::FusionPassTestUtils::RunGraphFusionPass("AASquareSumMaximumRsqrtMulFusionPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);
  //    GE_DUMP(compute_graph_ptr, "square_sum_maximum_rsqrt_mul_fusion_test_2_after");
  bool findL2Normalize = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Rsqrt") {
      findL2Normalize = true;
    }
    if (node->GetType() == "Maximum") {
      findL2Normalize = true;
    }
    if (node->GetType() == "ReduceSum") {
      findL2Normalize = true;
    }
  }
  EXPECT_EQ(findL2Normalize, false);
}