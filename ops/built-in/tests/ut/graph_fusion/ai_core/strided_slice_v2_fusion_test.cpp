#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "selection_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class strided_slice_v2_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "strided_slice_v2 SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "strided_slice_v2 TearDown" << std::endl;
    }

    template<typename T>
    void test(std::string case_name,
              std::vector<T> vbegin,
              std::vector<T> vend,
              std::vector<T> vaxes,
              std::vector<T> vstrides,
              std::vector<int64_t> x_shape,
              std::vector<int64_t> y_shape,
              std::string expect_optype,
              bool &findOp) {
      ge::Graph graph(case_name);

      auto construct_a_node = [](std::vector<T> x) {
        if (x.empty()) {
          return static_cast<ge::Operator>(op::Data().set_attr_index(0));
        } else {
          int32_t begin_size = x.size();
          ge::Tensor begin_tensor;
          std::vector<int64_t> begin_vec{begin_size};
          ge::Shape begin_shape(begin_vec);
          ge::TensorDesc begin_desc(begin_shape, FORMAT_ND, DT_INT32);
          begin_desc.SetSize(begin_size * sizeof(int32_t));
          begin_tensor.SetTensorDesc(begin_desc);
          int32_t* begin_data = nullptr;
          begin_data = new int32_t[begin_size];
          for (int32_t i = 0; i < begin_size; i++) {
            *(begin_data + i) = x[i];
          }
          begin_tensor.SetData((uint8_t*)begin_data, begin_size * sizeof(int32_t));
          delete [] begin_data;
          auto begin_node = op::Const().set_attr_value(begin_tensor);
          begin_node.update_output_desc_y(begin_desc);
          return static_cast<ge::Operator>(begin_node);
        }
      };

      ge::Operator begin = construct_a_node(vbegin);
      ge::Operator end = construct_a_node(vend);
      ge::Operator axes = construct_a_node(vaxes);
      ge::Operator strides = construct_a_node(vstrides);
      auto data_x = op::Data().set_attr_index(0);
      auto strided_slice_op = op::StridedSliceV2("strided_slice_op")
                              .set_input_x(data_x)
                              .set_input_begin(begin)
                              .set_input_end(end)
                              .set_input_axes(axes)
                              .set_input_strides(strides);

      ge::TensorDesc data_x_desc(ge::Shape(x_shape), FORMAT_ND, DT_INT32);
      data_x.update_input_desc_x(data_x_desc);
      data_x.update_output_desc_y(data_x_desc);

      ge::TensorDesc output_desc(ge::Shape(y_shape), FORMAT_ND, DT_INT32);
      strided_slice_op.update_input_desc_x(output_desc);
      strided_slice_op.update_output_desc_y(output_desc);

      std::vector<Operator> inputs{data_x, begin, end, strides};
      std::vector<Operator> outputs{strided_slice_op};
      graph.SetInputs(inputs).SetOutputs(outputs);

      ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
      fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
      // GE_DUMP(compute_graph_ptr, "strided_slice_v2_fusion_test_1_before");
      fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrStridedSliceV2Fusion", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

      for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == expect_optype) {
          findOp = true;
        }
      }
    }
};

TEST_F(strided_slice_v2_fusion_test, strided_slice_v2_fusion_test_1) {
  bool findOp = false;
  test<int32_t>("strided_slice_v2_fusion_test_1",
                {0, 0, 0}, // begin
                {3, 2, 4}, // end
                {0, 1, 2}, // axes
                {1, 1, 1}, // strides
                {10, 12, 12}, // x shape
                {3, 2, 4}, // y shape
                "StridedSliceD",
                findOp);
  EXPECT_EQ(findOp, true);
}

TEST_F(strided_slice_v2_fusion_test, strided_slice_v2_fusion_test_2) {
    bool findOp = false;
  test<int32_t>("strided_slice_v2_fusion_test_2",
                {-1, -1, -1}, // begin
                {-1000, -1000, -1000}, // end
                {0, 1, 2}, // axes
                {-1, -1, -1}, // strides
                {10, 12, 12}, // x shape
                {10, 12, 12}, // y shape
                "ReverseV2D",
                findOp);
  EXPECT_EQ(findOp, true);
}

TEST_F(strided_slice_v2_fusion_test, strided_slice_v2_fusion_test_3) {
  bool findOp = false;
  test<int32_t>("strided_slice_v2_fusion_test_3",
                {0, 0, 0}, // begin
                {3, 2, 4}, // end
                {0, 1, 2}, // axes
                {}, // strides
                {-1, 12, 12}, // x shape
                {-1, -1, -1}, // y shape
                "StridedSliceV3",
                findOp);
  EXPECT_EQ(findOp, true);
}

TEST_F(strided_slice_v2_fusion_test, last_dim_stride_greater_than_one) {
  bool findOp = false;
  test<int32_t>("last_dim_stride_greater_than_one",
                {0, 0, 0}, // begin
                {3, 2, 4}, // end
                {0, 1, 2}, // axes
                {1, 1, 2}, // strides
                {10, 12, 12}, // x shape
                {3, 2, 2}, // y shape
                "StridedSliceD",
                findOp);
  EXPECT_EQ(findOp, true);
}

TEST_F(strided_slice_v2_fusion_test, strided_slice_v2_fusion_test_5) {
  bool findOp = false;
  test<int64_t>("strided_slice_v2_fusion_test_5",
                {0, 0}, // begin
                {2, 4}, // end
                {1, 2}, // axes
                {1, 1}, // strides
                {10, 12, 12}, // x shape
                {10, 2, 4}, // y shape
                "StridedSliceD",
                findOp);
  EXPECT_EQ(findOp, true);
}

TEST_F(strided_slice_v2_fusion_test, negative_stride) {
  bool findOp = false;
  test<int32_t>("negative_stride",
                {0, 0, 0}, // begin
                {3, 2, 4}, // end
                {0, 1, 2}, // axes
                {1, 1, -2}, // strides
                {10, 12, 12}, // x shape
                {3, 2, 2}, // y shape
                "StridedSlice",
                findOp);
  EXPECT_EQ(findOp, true);
}
