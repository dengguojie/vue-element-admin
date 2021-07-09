#include <gtest/gtest.h>
#include <vector>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"

class LRUCacheV2Test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "lru_cache_v2 test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "lru_cache_v2 test TearDown" << std::endl;
    }
};

TEST_F(LRUCacheV2Test, lru_cache_v2_test_case_1) {
    ge::op::LRUCacheV2 lru_cache_v2_op;
    lru_cache_v2_op.UpdateInputDesc("index_list", create_desc_with_ori({32}, ge::DT_INT32, ge::FORMAT_ND, {32}, ge::FORMAT_ND));
    lru_cache_v2_op.UpdateInputDesc("data", create_desc_with_ori({100, 4}, ge::DT_FLOAT, ge::FORMAT_ND, {100, 4}, ge::FORMAT_ND));
    lru_cache_v2_op.UpdateInputDesc("cache", create_desc_with_ori({2048}, ge::DT_FLOAT, ge::FORMAT_ND, {2048}, ge::FORMAT_ND));
    lru_cache_v2_op.UpdateInputDesc("tag", create_desc_with_ori({512}, ge::DT_INT32, ge::FORMAT_ND, {512}, ge::FORMAT_ND));
    lru_cache_v2_op.UpdateInputDesc("is_last_call", create_desc_with_ori({1}, ge::DT_INT32, ge::FORMAT_ND, {1}, ge::FORMAT_ND));
    lru_cache_v2_op.SetAttr("pre_route_count", (int)4);
    auto ret = lru_cache_v2_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_0_desc = lru_cache_v2_op.GetOutputDesc("data");
    auto output_1_desc = lru_cache_v2_op.GetOutputDesc("cache");
    auto output_2_desc = lru_cache_v2_op.GetOutputDesc("tag");
    auto output_3_desc = lru_cache_v2_op.GetOutputDesc("index_offset_list");
    auto output_4_desc = lru_cache_v2_op.GetOutputDesc("not_in_cache_index_list");
    auto output_5_desc = lru_cache_v2_op.GetOutputDesc("not_in_cache_number");
    std::vector<int64_t> expected_output_0_shape = {100, 4};
    std::vector<int64_t> expected_output_1_shape = {2048};
    std::vector<int64_t> expected_output_2_shape = {512};
    std::vector<int64_t> expected_output_3_shape = {32};
    std::vector<int64_t> expected_output_4_shape = {32};
    std::vector<int64_t> expected_output_5_shape = {1};
    EXPECT_EQ(output_0_desc.GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(output_1_desc.GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(output_2_desc.GetDataType(), ge::DT_INT32);
    EXPECT_EQ(output_3_desc.GetDataType(), ge::DT_INT32);
    EXPECT_EQ(output_4_desc.GetDataType(), ge::DT_INT32);
    EXPECT_EQ(output_5_desc.GetDataType(), ge::DT_INT32);
    EXPECT_EQ(output_0_desc.GetShape().GetDims(), expected_output_0_shape);
    EXPECT_EQ(output_1_desc.GetShape().GetDims(), expected_output_1_shape);
    EXPECT_EQ(output_2_desc.GetShape().GetDims(), expected_output_2_shape);
    EXPECT_EQ(output_3_desc.GetShape().GetDims(), expected_output_3_shape);
    EXPECT_EQ(output_4_desc.GetShape().GetDims(), expected_output_4_shape);
    EXPECT_EQ(output_5_desc.GetShape().GetDims(), expected_output_5_shape);
}
