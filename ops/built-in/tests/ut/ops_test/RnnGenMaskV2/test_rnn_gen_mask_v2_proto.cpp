#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "rnn.h"

class RnnGenMaskV2Test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "RnnGenMaskV2Test Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "RnnGenMaskV2Test Proto Test TearDown" << std::endl;
    }
};

TEST_F(RnnGenMaskV2Test, rnn_gen_mask_tsest_1) {
    ge::op::RnnGenMaskV2 rnn_gen_mask_op;

    auto seq_length = create_desc_shape_range({-1},
                                                ge::DT_FLOAT16, ge::FORMAT_ND,
                                                {32},
                                                ge::FORMAT_ND, {{32, 32}});
    auto b = create_desc_shape_range({-1},
                                         ge::DT_FLOAT16, ge::FORMAT_ND,
                                         {256},
                                         ge::FORMAT_ND, {{256, 256}});
    auto x = create_desc_shape_range({-1, 32, -1},
                                                ge::DT_FLOAT16, ge::FORMAT_ND,
                                                {2, 32, 64},
                                                ge::FORMAT_ND, {{2, 2},{32,32},{64,64}});

    rnn_gen_mask_op.UpdateInputDesc("seq_length", seq_length);
    rnn_gen_mask_op.UpdateInputDesc("b", b);
    rnn_gen_mask_op.UpdateInputDesc("x", x);


    // infer
    auto ret = rnn_gen_mask_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    // compare
    auto output_desc = rnn_gen_mask_op.GetOutputDescByName("seq_mask");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);

    std::vector<int64_t> expected_output_shape = {-1, 32, -1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);

    std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{2, 2},{32,32},{64,64}};
    EXPECT_EQ(output_shape_range, expected_shape_range);
}
