#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "array_ops.h"
#include "op_proto_test_util.h"
#include "data_flow_ops.h"


class lruCacheTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "lruCacheTest SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "lruCacheTest TearDown" << std::endl;
    }
};
TEST_F(lruCacheTest, lruCacheTest_1) {
    ge::op::LruCache op;
    op.set_attr_cache_size(200000);
    op.set_attr_container("Hello");
    op.set_attr_dtype(ge::DT_FLOAT);
    op.set_attr_load_factor(2);
    op.set_attr_shared_name("LruCache");
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
TEST_F(lruCacheTest, lruCacheTest_2) {
    ge::op::LruCache op;
    op.set_attr_cache_size(5);
    op.set_attr_container("Hello");
    op.set_attr_dtype(ge::DT_RESOURCE);
    op.set_attr_load_factor(0.5);
    op.set_attr_shared_name("LruCache");
    std::vector<int64_t> expected_shape = {};
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("cache");
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}
TEST_F(lruCacheTest, lruCacheTest_3) {
    ge::op::LruCache op;
    op.set_attr_cache_size(-1);
    op.set_attr_container("Hello");
    op.set_attr_dtype(ge::DT_FLOAT);
    op.set_attr_load_factor(0.5);
    op.set_attr_shared_name("LruCache");
    std::vector<int64_t> expected_shape = {};
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

