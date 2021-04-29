#include <type_traits>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "image_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class remap_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "remap_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "remap_fusion_test TearDown" << std::endl;
    }
};

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

TEST_F(remap_fusion_test, remap_fusion_test_2) {
    ge::Graph graph("remap_fusion_test_2");
    auto image_data = op::Data("image_data");
    std::vector<int64_t> dims_x{1, 1024, 4000, 3};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND,  DT_FLOAT16);
    image_data.update_input_desc_x(tensorDescX);
    image_data.update_output_desc_y(tensorDescX);

    auto offset_data = op::Data("offset_data");
    std::vector<int64_t> dims_offset{1, 1024, 4000, 2};
    ge::Shape shape_offset(dims_offset);
    ge::TensorDesc tensorDescScore(shape_offset, FORMAT_ND,  DT_FLOAT);
    offset_data.update_input_desc_x(tensorDescScore);
    offset_data.update_output_desc_y(tensorDescScore);

    auto remap_op = op::Remap("Remap_1");
    remap_op.set_input_img(image_data);
    remap_op.set_input_map_offset(offset_data);

    std::vector<Operator> inputs{image_data, offset_data};
    std::vector<Operator> outputs{remap_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("RemapFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Muls") {
            findTranspose = true;
            std::cout << "test pass!" << std::endl;
        }
    }
    EXPECT_EQ(findTranspose, true);
}

TEST_F(remap_fusion_test, remap_fusion_test_3) {
    ge::Graph graph("remap_fusion_test_3");
    auto image_data = op::Data("image_data");
    std::vector<int64_t> dims_x{1, 2024, 4000, 3};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND,  DT_UINT8);
    image_data.update_input_desc_x(tensorDescX);
    image_data.update_output_desc_y(tensorDescX);

    auto offset_data = op::Data("offset_data");
    std::vector<int64_t> dims_offset{1, 1024, 4000, 2};
    ge::Shape shape_offset(dims_offset);
    ge::TensorDesc tensorDescScore(shape_offset, FORMAT_ND,  DT_FLOAT);
    offset_data.update_input_desc_x(tensorDescScore);
    offset_data.update_output_desc_y(tensorDescScore);

    auto remap_op = op::Remap("Remap_3");
    remap_op.set_input_img(image_data);
    remap_op.set_input_map_offset(offset_data);

    std::vector<Operator> inputs{image_data, offset_data};
    std::vector<Operator> outputs{remap_op};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("RemapFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = false;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Muls") {
            findTranspose = true;
            std::cout << "test pass!" << std::endl;
        }
    }
    EXPECT_EQ(findTranspose, true);
}

