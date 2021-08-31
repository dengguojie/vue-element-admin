#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "vector_search.h"

class ScanPQCodesProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ScanPQCodes Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ScanPQCodes Proto Test TearDown" << std::endl;
  }
};

TEST_F(ScanPQCodesProtoTest, ScanPQCodesVerifyTest_1) {
  ge::op::ScanPQCodes op;
  op.UpdateInputDesc("ivf", create_desc({20480, 16}, ge::DT_UINT8));
  op.UpdateInputDesc("bucket_list", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_base_distance", create_desc({8}, ge::DT_FLOAT16));
  op.UpdateInputDesc("bucket_limits", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_offsets", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("adc_tables", create_desc({8, 16, 256}, ge::DT_FLOAT16));
  op.SetAttr("total_limit", 20480);
  op.SetAttr("group_size", 64);
  op.SetAttr("extreme_mode", 0);
  op.SetAttr("split_count", 1);
  op.SetAttr("split_index", 0);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(ScanPQCodesProtoTest, ScanPQCodesVerifyTest_2) {
  ge::op::ScanPQCodes op;
  op.UpdateInputDesc("ivf", create_desc({20480, 15}, ge::DT_UINT8));
  op.UpdateInputDesc("bucket_list", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_base_distance", create_desc({8}, ge::DT_FLOAT16));
  op.UpdateInputDesc("bucket_limits", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_offsets", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("adc_tables", create_desc({8, 16, 256}, ge::DT_FLOAT16));
  op.SetAttr("total_limit", 20480);
  op.SetAttr("group_size", 64);
  op.SetAttr("extreme_mode", 0);
  op.SetAttr("split_count", 1);
  op.SetAttr("split_index", 0);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ScanPQCodesProtoTest, ScanPQCodesVerifyTest_3) {
  ge::op::ScanPQCodes op;
  op.UpdateInputDesc("ivf", create_desc({20480, 16}, ge::DT_UINT8));
  op.UpdateInputDesc("bucket_list", create_desc({8, 8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_base_distance", create_desc({8}, ge::DT_FLOAT16));
  op.UpdateInputDesc("bucket_limits", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_offsets", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("adc_tables", create_desc({8, 16, 256}, ge::DT_FLOAT16));
  op.SetAttr("total_limit", 20480);
  op.SetAttr("group_size", 64);
  op.SetAttr("extreme_mode", 0);
  op.SetAttr("split_count", 1);
  op.SetAttr("split_index", 0);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ScanPQCodesProtoTest, ScanPQCodesVerifyTest_4) {
  ge::op::ScanPQCodes op;
  op.UpdateInputDesc("ivf", create_desc({20480, 16}, ge::DT_UINT8));
  op.UpdateInputDesc("bucket_list", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_base_distance", create_desc({8, 8}, ge::DT_FLOAT16));
  op.UpdateInputDesc("bucket_limits", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_offsets", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("adc_tables", create_desc({8, 16, 256}, ge::DT_FLOAT16));
  op.SetAttr("total_limit", 20480);
  op.SetAttr("group_size", 64);
  op.SetAttr("extreme_mode", 0);
  op.SetAttr("split_count", 1);
  op.SetAttr("split_index", 0);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ScanPQCodesProtoTest, ScanPQCodesVerifyTest_5) {
  ge::op::ScanPQCodes op;
  op.UpdateInputDesc("ivf", create_desc({20480, 16}, ge::DT_UINT8));
  op.UpdateInputDesc("bucket_list", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_base_distance", create_desc({8}, ge::DT_FLOAT16));
  op.UpdateInputDesc("bucket_limits", create_desc({8, 8, 8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_offsets", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("adc_tables", create_desc({8, 16, 256}, ge::DT_FLOAT16));
  op.SetAttr("total_limit", 20480);
  op.SetAttr("group_size", 64);
  op.SetAttr("extreme_mode", 0);
  op.SetAttr("split_count", 1);
  op.SetAttr("split_index", 0);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ScanPQCodesProtoTest, ScanPQCodesVerifyTest_6) {
  ge::op::ScanPQCodes op;
  op.UpdateInputDesc("ivf", create_desc({20480, 16}, ge::DT_UINT8));
  op.UpdateInputDesc("bucket_list", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_base_distance", create_desc({8}, ge::DT_FLOAT16));
  op.UpdateInputDesc("bucket_limits", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_offsets", create_desc({8, 8, 8}, ge::DT_INT32));
  op.UpdateInputDesc("adc_tables", create_desc({8, 16, 256}, ge::DT_FLOAT16));
  op.SetAttr("total_limit", 20480);
  op.SetAttr("group_size", 64);
  op.SetAttr("extreme_mode", 0);
  op.SetAttr("split_count", 1);
  op.SetAttr("split_index", 0);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ScanPQCodesProtoTest, ScanPQCodesVerifyTest_7) {
  ge::op::ScanPQCodes op;
  op.UpdateInputDesc("ivf", create_desc({20480, 16}, ge::DT_UINT8));
  op.UpdateInputDesc("bucket_list", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_base_distance", create_desc({8}, ge::DT_FLOAT16));
  op.UpdateInputDesc("bucket_limits", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_offsets", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("adc_tables", create_desc({8, 1, 256}, ge::DT_FLOAT16));
  op.SetAttr("total_limit", 20480);
  op.SetAttr("group_size", 64);
  op.SetAttr("extreme_mode", 0);
  op.SetAttr("split_count", 1);
  op.SetAttr("split_index", 0);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ScanPQCodesProtoTest, ScanPQCodesVerifyTest_8) {
  ge::op::ScanPQCodes op;
  op.UpdateInputDesc("ivf", create_desc({20480, 16}, ge::DT_UINT8));
  op.UpdateInputDesc("bucket_list", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_base_distance", create_desc({8}, ge::DT_FLOAT16));
  op.UpdateInputDesc("bucket_limits", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_offsets", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("adc_tables", create_desc({8, 16, 2}, ge::DT_FLOAT16));
  op.SetAttr("total_limit", 20480);
  op.SetAttr("group_size", 64);
  op.SetAttr("extreme_mode", 0);
  op.SetAttr("split_count", 1);
  op.SetAttr("split_index", 0);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ScanPQCodesProtoTest, ScanPQCodesVerifyTest_9) {
  ge::op::ScanPQCodes op;
  op.UpdateInputDesc("ivf", create_desc({20480, 16}, ge::DT_UINT8));
  op.UpdateInputDesc("bucket_list", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_base_distance", create_desc({8}, ge::DT_FLOAT16));
  op.UpdateInputDesc("bucket_limits", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_offsets", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("adc_tables", create_desc({8, 16, 256}, ge::DT_FLOAT16));
  op.SetAttr("total_limit", 20480);
  op.SetAttr("group_size", 60);
  op.SetAttr("extreme_mode", 0);
  op.SetAttr("split_count", 1);
  op.SetAttr("split_index", 0);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ScanPQCodesProtoTest, ScanPQCodesVerifyTest_10) {
  ge::op::ScanPQCodes op;
  op.UpdateInputDesc("ivf", create_desc({20480, 16}, ge::DT_UINT8));
  op.UpdateInputDesc("bucket_list", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_base_distance", create_desc({8}, ge::DT_FLOAT16));
  op.UpdateInputDesc("bucket_limits", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_offsets", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("adc_tables", create_desc({8, 16, 256}, ge::DT_FLOAT16));
  op.SetAttr("total_limit", 20480);
  op.SetAttr("group_size", 20544);
  op.SetAttr("extreme_mode", 0);
  op.SetAttr("split_count", 1);
  op.SetAttr("split_index", 0);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ScanPQCodesProtoTest, ScanPQCodesVerifyTest_11) {
  ge::op::ScanPQCodes op;
  op.UpdateInputDesc("ivf", create_desc({20480, 16}, ge::DT_UINT8));
  op.UpdateInputDesc("bucket_list", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_base_distance", create_desc({8}, ge::DT_FLOAT16));
  op.UpdateInputDesc("bucket_limits", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_offsets", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("adc_tables", create_desc({8, 16, 256}, ge::DT_FLOAT16));
  op.SetAttr("total_limit", 20480);
  op.SetAttr("group_size", 64);
  op.SetAttr("extreme_mode", 2);
  op.SetAttr("split_count", 1);
  op.SetAttr("split_index", 0);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ScanPQCodesProtoTest, ScanPQCodesVerifyTest_12) {
  ge::op::ScanPQCodes op;
  op.UpdateInputDesc("ivf", create_desc({20480, 16}, ge::DT_UINT8));
  op.UpdateInputDesc("bucket_list", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_base_distance", create_desc({8}, ge::DT_FLOAT16));
  op.UpdateInputDesc("bucket_limits", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_offsets", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("adc_tables", create_desc({8, 16, 256}, ge::DT_FLOAT16));
  op.SetAttr("total_limit", 20480);
  op.SetAttr("group_size", 64);
  op.SetAttr("extreme_mode", 0);
  op.SetAttr("split_count", 0);
  op.SetAttr("split_index", 0);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ScanPQCodesProtoTest, ScanPQCodesVerifyTest_13) {
  ge::op::ScanPQCodes op;
  op.UpdateInputDesc("ivf", create_desc({20480, 16}, ge::DT_UINT8));
  op.UpdateInputDesc("bucket_list", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_base_distance", create_desc({8}, ge::DT_FLOAT16));
  op.UpdateInputDesc("bucket_limits", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("bucket_offsets", create_desc({8}, ge::DT_INT32));
  op.UpdateInputDesc("adc_tables", create_desc({8, 16, 256}, ge::DT_FLOAT16));
  op.SetAttr("total_limit", 20480);
  op.SetAttr("group_size", 64);
  op.SetAttr("extreme_mode", 0);
  op.SetAttr("split_count", 2);
  op.SetAttr("split_index", 2);
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(ScanPQCodesProtoTest, ScanPQCodesInferShapeTest_1) {
  ge::op::ScanPQCodes op;
  op.UpdateInputDesc("ivf", create_desc_shape_range({2048, 16}, ge::DT_UINT8, 
                            ge::FORMAT_ND, {2048, 16}, ge::FORMAT_ND, {{2048, 20480}, {16, 16}}));
  op.UpdateInputDesc("bucket_list", create_desc_shape_range({1}, ge::DT_INT32, ge::FORMAT_ND, {8},
                                    ge::FORMAT_ND, {{1, 1}}));
  op.UpdateInputDesc("bucket_base_distance", create_desc_shape_range({1}, ge::DT_FLOAT16, ge::FORMAT_ND, {8},
                                             ge::FORMAT_ND, {{1, 1}}));
  op.UpdateInputDesc("bucket_limits", create_desc_shape_range({8}, ge::DT_INT32, ge::FORMAT_ND, {8},
                                      ge::FORMAT_ND, {{1, 1}}));
  op.UpdateInputDesc("bucket_offsets", create_desc_shape_range({1}, ge::DT_INT32, ge::FORMAT_ND, {8},
                                       ge::FORMAT_ND, {{1, 1}}));
  op.UpdateInputDesc("adc_tables", create_desc_shape_range({1}, ge::DT_FLOAT16, ge::FORMAT_ND, {8},
                                   ge::FORMAT_ND, {{1, 1}}));

  op.SetAttr("total_limit", 2048);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto actual_count_desc = op.GetOutputDescByName("actual_count");
  auto pq_distance_desc = op.GetOutputDescByName("pq_distance");
  auto grouped_extreme_distance_desc = op.GetOutputDescByName("grouped_extreme_distance");
  auto pq_ivf_desc = op.GetOutputDescByName("pq_ivf");
  auto pq_index_desc = op.GetOutputDescByName("pq_index");
  EXPECT_EQ(actual_count_desc.GetDataType(), ge::DT_INT32);
  EXPECT_EQ(pq_distance_desc.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(grouped_extreme_distance_desc.GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(pq_ivf_desc.GetDataType(), ge::DT_INT32);
  EXPECT_EQ(pq_index_desc.GetDataType(), ge::DT_INT32);

  std::vector<int64_t> actual_count_expect_shape = {1};
  std::vector<int64_t> pq_distance_expect_shape = {3072};
  std::vector<int64_t> grouped_extreme_distance_expect_shape = {48};
  std::vector<int64_t> pq_ivf_expect_shape = {3072};
  std::vector<int64_t> pq_index_expect_shape = {3072};
  EXPECT_EQ(actual_count_desc.GetShape().GetDims(), actual_count_expect_shape);
  EXPECT_EQ(pq_distance_desc.GetShape().GetDims(), pq_distance_expect_shape);
  EXPECT_EQ(grouped_extreme_distance_desc.GetShape().GetDims(), grouped_extreme_distance_expect_shape);
  EXPECT_EQ(pq_ivf_desc.GetShape().GetDims(), pq_ivf_expect_shape);
  EXPECT_EQ(pq_index_desc.GetShape().GetDims(), pq_index_expect_shape);

}

TEST_F(ScanPQCodesProtoTest, ScanPQCodesInferShapeTest_2) {
  ge::op::ScanPQCodes op;
  op.UpdateInputDesc("ivf", create_desc_shape_range({2048, 16}, ge::DT_UINT8,
                            ge::FORMAT_ND, {2048, 16}, ge::FORMAT_ND, {{2048, 20480}, {16, 16}}));
  op.UpdateInputDesc("bucket_list", create_desc_shape_range({8}, ge::DT_INT32, ge::FORMAT_ND, {8},
                                    ge::FORMAT_ND, {{8, 8}}));
  op.UpdateInputDesc("bucket_base_distance", create_desc_shape_range({8}, ge::DT_FLOAT16, ge::FORMAT_ND, {8},
                                             ge::FORMAT_ND, {{8, 8}}));
  op.UpdateInputDesc("bucket_limits", create_desc_shape_range({8}, ge::DT_INT32, ge::FORMAT_ND, {8},
                                      ge::FORMAT_ND, {{8, 8}}));
  op.UpdateInputDesc("bucket_offsets", create_desc_shape_range({8}, ge::DT_INT32, ge::FORMAT_ND, {8},
                                       ge::FORMAT_ND, {{8, 8}}));
  op.UpdateInputDesc("adc_tables", create_desc_shape_range({8}, ge::DT_FLOAT16, ge::FORMAT_ND, {8},
                                   ge::FORMAT_ND, {{8, 8}}));

  op.SetAttr("total_limit", 2048);
  op.SetAttr("group_size", 0);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
