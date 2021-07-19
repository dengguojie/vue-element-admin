#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "all_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class resize_bilinear_cast_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "resize_bilinear SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "resize_bilinear TearDown" << std::endl;
    }
};
TEST_F(resize_bilinear_cast_fusion_test, resize_bilinear_cast_fusion_test_1) {
  ge::Graph graph("resize_bilinear_cast_fusion_test_1");
  ge::Tensor size_tensor;
  std::vector<int64_t> size_vec{2};
  ge::Shape size_shape(size_vec);
  ge::TensorDesc size_desc(size_shape, FORMAT_ND, DT_INT32);
  int32_t input_size = size_desc.GetShape().GetShapeSize();
  size_desc.SetSize(input_size * sizeof(int32_t));
  size_tensor.SetTensorDesc(size_desc);
  int32_t* size_data = nullptr;
  size_data = new int32_t[input_size];
  *(size_data + 0) = 10;
  *(size_data + 1) = 10;
  size_tensor.SetData((uint8_t*)size_data, input_size * sizeof(int32_t));
  delete [] size_data;

  auto size = op::Constant().set_attr_value(size_tensor);
  auto data_x = op::Data().set_attr_index(0);
  auto resize_bilinear_op = op::ResizeBilinearV2("resize_bilinear_op")
                            .set_input_x(data_x)
                            .set_input_size(size);

  auto cast_op = op::Cast("cast_op");   
  cast_op.set_input_x(resize_bilinear_op)
         .set_attr_dst_type(1);

  size.update_output_desc_y(size_desc);

  std::vector<int64_t> data_x_vec{1,1,100,100,16};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);
  std::vector<int64_t> output_vec{1,1,10,10,16};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
  resize_bilinear_op.update_input_desc_x(data_x_desc);
  resize_bilinear_op.update_output_desc_y(output_desc);
 
  ge::TensorDesc tensorDescCastOut(output_shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  cast_op.update_input_desc_x(output_desc);
  cast_op.update_output_desc_y(tensorDescCastOut);

  std::vector<Operator> inputs{data_x, size};
  std::vector<Operator> outputs{cast_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ResizeBilinearV2CastFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Cast") {
            findOp = true;
        }
    }
  EXPECT_EQ(findOp, false);
}
TEST_F(resize_bilinear_cast_fusion_test, resize_bilinear_cast_fusion_test_2) {
  ge::Graph graph("resize_bilinear_cast_fusion_test_1");
  ge::Tensor size_tensor;
  std::vector<int64_t> size_vec{2};
  ge::Shape size_shape(size_vec);
  ge::TensorDesc size_desc(size_shape, FORMAT_ND, DT_INT32);
  int32_t input_size = size_desc.GetShape().GetShapeSize();
  size_desc.SetSize(input_size * sizeof(int32_t));
  size_tensor.SetTensorDesc(size_desc);
  int32_t* size_data = nullptr;
  size_data = new int32_t[input_size];
  *(size_data + 0) = 10;
  *(size_data + 1) = 10;
  size_tensor.SetData((uint8_t*)size_data, input_size * sizeof(int32_t));
  delete [] size_data;

  auto size = op::Constant().set_attr_value(size_tensor);
  auto data_x = op::Data().set_attr_index(0);
  auto resize_bilinear_op = op::ResizeBilinearV2("resize_bilinear_op")
                            .set_input_x(data_x)
                            .set_input_size(size);

  auto cast_op = op::Cast("cast_op");   
  cast_op.set_input_x(resize_bilinear_op)
         .set_attr_dst_type(1);

  size.update_output_desc_y(size_desc);

  std::vector<int64_t> data_x_vec{1,100,100,16};
  ge::Shape data_x_shape(data_x_vec);
  ge::TensorDesc data_x_desc(data_x_shape, ge::FORMAT_NHWC, ge::DT_FLOAT16);
  data_x.update_input_desc_x(data_x_desc);
  data_x.update_output_desc_y(data_x_desc);
  std::vector<int64_t> output_vec{1,10,10,16};
  ge::Shape output_shape(output_vec);
  ge::TensorDesc output_desc(output_shape, ge::FORMAT_NHWC, ge::DT_FLOAT);
  resize_bilinear_op.update_input_desc_x(data_x_desc);
  resize_bilinear_op.update_output_desc_y(output_desc);
 
  ge::TensorDesc tensorDescCastOut(output_shape, ge::FORMAT_NHWC, ge::DT_FLOAT16);
  cast_op.update_input_desc_x(output_desc);
  cast_op.update_output_desc_y(tensorDescCastOut);

  std::vector<Operator> inputs{data_x, size};
  std::vector<Operator> outputs{cast_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ResizeBilinearV2CastFusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findOp = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "Cast") {
            findOp = true;
        }
    }
  EXPECT_EQ(findOp, true);
}
