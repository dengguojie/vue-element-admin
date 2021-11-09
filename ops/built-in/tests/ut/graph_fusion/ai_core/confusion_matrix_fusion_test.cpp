#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "fusion_pass_test_utils.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "split_combination_ops.h"
#include "transformation_ops.h"
#include "sparse_ops.h"


using namespace ge;
using namespace op;


class confusion_matrix_fusion_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "confusion_matrix_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "confusion_matrix_fusion_test TearDown" << std::endl;
  }
  template <typename T>
  static Operator GetConstNode(const std::vector<int64_t>& const_shape,
                              const std::vector<T>& const_value,
                              const std::string& const_name,
                              const ge::Format& const_format) {
    auto const_size = const_value.size();
    constexpr ge::DataType const_dtype = std::is_same<T, float>::value ? ge::DT_FLOAT : ge::DT_INT32;
    TensorDesc const_desc(ge::Shape(const_shape), const_format, const_dtype);
    Tensor const_tensor(const_desc);
    const_tensor.SetData(reinterpret_cast<const uint8_t*>(const_value.data()), const_size * sizeof(T));
    auto const_op = op::Const(const_name.c_str()).set_attr_value(const_tensor);
    return const_op;
  }
  template <typename Y>
  void BuildGraph(ComputeGraphPtr &compute_graph_ptr,const std::vector<Y> input_values,const std::vector<Y> input_indices,const std::vector<Y> input_shape) {
    ge::Graph graph("test_confusion_matrix");

    auto input0 = op::Data("input0");
    ge::Shape shape1({100,50});
    ge::TensorDesc tensorDescX(shape1, FORMAT_NHWC, DT_FLOAT16);
    input0.update_input_desc_x(tensorDescX);
    input0.update_output_desc_y(tensorDescX);

    auto input1 = op::Data("input1");
    ge::Shape shape2({100,50});
    ge::TensorDesc tensorDescX1(shape2, FORMAT_NHWC, DT_FLOAT16);
    input1.update_input_desc_x(tensorDescX1);
    input1.update_output_desc_y(tensorDescX1);

    auto cast_layer1 = op::Cast("cast1");
    cast_layer1.set_input_x(input0)
              .set_attr_dst_type(2);

    auto cast_layer2 = op::Cast("cast2");
    cast_layer2.set_input_x(input1)
              .set_attr_dst_type(2);

    auto pack_layer = op::Pack("pack");
    pack_layer.create_dynamic_input_x(2)
              .set_dynamic_input_x(0, cast_layer1)
              .set_dynamic_input_x(1, cast_layer2)
              .set_attr_axis(0)
              .set_attr_N(2);

    auto transpose_input1 = op::Data("transpose_input1");
    ge::Shape shape_x({3,3,3});
    ge::TensorDesc tensorDescX3(shape_x, FORMAT_NHWC, DT_FLOAT16);
    transpose_input1.update_input_desc_x(tensorDescX3);
    transpose_input1.update_output_desc_y(tensorDescX3);   

    auto transpose_layer = op::Transpose("transpose");
    transpose_layer.set_input_x(pack_layer)
                  .set_input_perm(transpose_input1);

    // get a const op
    std::vector<int64_t> const_shape_dims{};
    auto const_format = FORMAT_ND;

    auto input_values_op = GetConstNode(const_shape_dims, input_values, "input_values", const_format);
    auto input_indices_op = GetConstNode(const_shape_dims, input_indices, "input_indices", const_format);
    auto input_shape_op = GetConstNode(const_shape_dims, input_shape, "input_shape", const_format);

    auto sparseTensorDenseAdd_layer = op::SparseTensorDenseAdd("sparseTensorDenseAdd");
    sparseTensorDenseAdd_layer.set_input_x1_values(input_values_op)
                              .set_input_x1_indices(input_indices_op)
                              .set_input_x1_shape(input_shape_op)
                              .set_input_x2(transpose_layer);

    std::vector<Operator> inputs{input0, input1,input_values_op};
    std::vector<Operator> outputs{sparseTensorDenseAdd_layer};
    graph.SetInputs(inputs).SetOutputs(outputs);
    compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  }
};


TEST_F(confusion_matrix_fusion_test, confusion_matrix_fusion_test_1){
  ge::ComputeGraphPtr compute_graph_ptr;
  std::vector<int32_t>  input_values{100};
  std::vector<int32_t>  input_indices{100};
  std::vector<int32_t>  input_shape{3};
  BuildGraph(compute_graph_ptr, input_values,input_indices,input_shape);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ConfusionMatrixFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool confusionMatrix_fusion = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ConfusionMatrix") {
        confusionMatrix_fusion = true;
    }
  }
  EXPECT_EQ(confusionMatrix_fusion, true);
}
TEST_F(confusion_matrix_fusion_test, confusion_matrix_fusion_test_2){
  ge::ComputeGraphPtr compute_graph_ptr;
  std::vector<float>  input_values{100};
  std::vector<float>  input_indices{100};
  std::vector<float>  input_shape{3};
  BuildGraph(compute_graph_ptr, input_values,input_indices,input_shape);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ConfusionMatrixFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  bool confusionMatrix_fusion = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ConfusionMatrix") {
        confusionMatrix_fusion = true;
    }
  }
  EXPECT_EQ(confusionMatrix_fusion, true);
}