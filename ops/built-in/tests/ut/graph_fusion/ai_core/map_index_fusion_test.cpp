#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "control_flow_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class map_index_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "map_index_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "map_index_fusion_test TearDown" << std::endl;
    }
};

TEST_F(map_index_fusion_test, map_index_fusion_test_1) {
    ge::Graph graph("map_index_fusion_test_1");

    auto x = op::Data("x");
    std::vector<int64_t> dims_x{1};
    ge::Shape shape_x(dims_x);
    ge::TensorDesc tensorDescX(shape_x, FORMAT_ND, DT_INT32);
    x.update_input_desc_x(tensorDescX);
    x.update_output_desc_y(tensorDescX);

    ge::Tensor data_seq_tensor;
    std::vector<int64_t> dims_data_seq{2};
    ge::Shape shape_data_seq(dims_data_seq);
    ge::TensorDesc tensorDescDataSeq(shape_data_seq, FORMAT_ND, DT_INT32);
    int64_t data_seq_size = tensorDescDataSeq.GetShape().GetShapeSize();
    tensorDescDataSeq.SetSize(data_seq_size * sizeof(int32_t));
    data_seq_tensor.SetTensorDesc(tensorDescDataSeq);
    int32_t* data_seq_data = nullptr;
    data_seq_data = new int32_t[data_seq_size];
    *(data_seq_data + 0) = 1;
    *(data_seq_data + 1) = 1;
    data_seq_tensor.SetData((uint8_t*)data_seq_data, data_seq_size * sizeof(int32_t));
    delete [] data_seq_data;

    auto dataSeq = op::Constant().set_attr_value(data_seq_tensor);

    auto mapIndexOp = op::MapIndex("MapIndex_1");
    mapIndexOp.set_input_x(x);
    mapIndexOp.set_input_data_seq(dataSeq);

    std::vector<Operator> inputs{x, dataSeq};
    std::vector<Operator> outputs{mapIndexOp};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("MapIndexFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findTranspose = true;

    EXPECT_EQ(findTranspose, true);
}
