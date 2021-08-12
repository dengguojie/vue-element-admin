#include <gtest/gtest.h>
#include <iostream>
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "op_proto_test_util.h"
#include "nn_pooling_ops.h"

using namespace ge;
using namespace op;

class global_lppool:public testing::Test {
    protected:
        static void SetUpTestCase() {
            std::cout <<"global_lppool Proto Test SetUp" <<std::endl;
        }

        static void TearDownTestCase() {
            std::cout <<"global_lppool Proto Test TearDown" <<std::endl;
        }
};


TEST_F(global_lppool,global_lppool_infershape_diff_test) {
    ge::op::GlobalLpPool op;
    std::vector<std::pair<int64_t,int64_t>> range_x1 = {{1, 100}, {1, 100}, {1, 100}, {1, 100}};
    std::vector<std::pair<int64_t,int64_t>> expected_range = {{1, 100}, {1, 100}, {1, 1}, {1, 1}};
    auto shape_x1 = vector<int64_t>({-1, -1, -1, -1});
    auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT, ge::FORMAT_ND,
                                                  shape_x1, ge::FORMAT_NCHW, range_x1);

    op.UpdateInputDesc("x", tensor_desc_x1);
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret,ge::GRAPH_SUCCESS);
    auto output_y1_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_y1_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {-1, -1, 1, 1};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_output_shape);
    std::vector<std::pair<int64_t,int64_t>> output_shape_range;
    EXPECT_EQ(output_y1_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output_shape_range, expected_range);
}
