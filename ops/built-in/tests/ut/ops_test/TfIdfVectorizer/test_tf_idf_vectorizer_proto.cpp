#include <gtest/gtest.h>

#include <iostream>

#include "array_ops.h"
#include "op_proto_test_util.h"
#include "transformation_ops.h"

class TfIdfVectorizerTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TfIdfVectorizerTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TfIdfVectorizerTest TearDown" << std::endl;
  }
};

TEST_F(TfIdfVectorizerTest, TfIdfVectorizer_verify_test1_failed) {
  ge::op::TfIdfVectorizer op;
  op.UpdateInputDesc("input",
                     create_desc_with_ori({12}, ge::DT_INT8, ge::FORMAT_ND, {12}, ge::FORMAT_ND));
  auto verify_fail_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res, ge::GRAPH_FAILED);
}

TEST_F(TfIdfVectorizerTest, TfIdfVectorizer_verify_test2_failed) {
  ge::op::TfIdfVectorizer op;
  op.UpdateInputDesc("input",
                     create_desc_with_ori({2,2,2}, ge::DT_INT64, ge::FORMAT_ND, {2,2,2}, ge::FORMAT_ND));
  auto verify_fail_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res, ge::GRAPH_FAILED);
}

TEST_F(TfIdfVectorizerTest, TfIdfVectorizer_verify_test3_failed) {
  ge::op::TfIdfVectorizer op;
  op.UpdateInputDesc("input",
                     create_desc_with_ori({-1,-1}, ge::DT_INT64, ge::FORMAT_ND, {-1,-1}, ge::FORMAT_ND));
  auto verify_fail_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res, ge::GRAPH_FAILED);
}

TEST_F(TfIdfVectorizerTest, TfIdfVectorizer_verify_test4_failed) {
  ge::op::TfIdfVectorizer op;
  op.UpdateInputDesc("input",
                     create_desc_with_ori({2}, ge::DT_INT64, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  auto verify_fail_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res, ge::GRAPH_FAILED);
}

TEST_F(TfIdfVectorizerTest, TfIdfVectorizer_verify_test5_failed) {
  ge::op::TfIdfVectorizer op;
  op.UpdateInputDesc("input",
                     create_desc_with_ori({2}, ge::DT_INT64, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.SetAttr("max_gram_length", -3);
  auto verify_fail_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res, ge::GRAPH_FAILED);
}

TEST_F(TfIdfVectorizerTest, TfIdfVectorizer_verify_test6_failed) {
  ge::op::TfIdfVectorizer op;
  op.UpdateInputDesc("input",
                     create_desc_with_ori({2}, ge::DT_INT64, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.SetAttr("max_gram_length", 2);
  op.SetAttr("max_skip_count", -2);
  auto verify_fail_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res, ge::GRAPH_FAILED);
}

TEST_F(TfIdfVectorizerTest, TfIdfVectorizer_verify_test7_failed) {
  ge::op::TfIdfVectorizer op;
  op.UpdateInputDesc("input",
                     create_desc_with_ori({2}, ge::DT_INT64, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.SetAttr("max_gram_length", 2);
  op.SetAttr("max_skip_count", 0);
  op.SetAttr("min_gram_length", -2);
  auto verify_fail_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res, ge::GRAPH_FAILED);
}

TEST_F(TfIdfVectorizerTest, TfIdfVectorizer_verify_test8_failed) {
  ge::op::TfIdfVectorizer op;
  op.UpdateInputDesc("input",
                     create_desc_with_ori({2}, ge::DT_INT64, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.SetAttr("max_gram_length", 1);
  op.SetAttr("max_skip_count", 0);
  op.SetAttr("min_gram_length", 2);
  op.SetAttr("mode", "TF");
  auto verify_fail_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res, ge::GRAPH_FAILED);
}

TEST_F(TfIdfVectorizerTest, TfIdfVectorizer_verify_test9_failed) {
  ge::op::TfIdfVectorizer op;
  op.UpdateInputDesc("input",
                     create_desc_with_ori({2}, ge::DT_INT64, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.SetAttr("max_gram_length", 2);
  op.SetAttr("max_skip_count", 0);
  op.SetAttr("min_gram_length", 1);
  op.SetAttr("mode", "ERROR");
  auto verify_fail_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res, ge::GRAPH_FAILED);
}

TEST_F(TfIdfVectorizerTest, TfIdfVectorizer_verify_test10_failed) {
  ge::op::TfIdfVectorizer op;
  op.UpdateInputDesc("input",
                     create_desc_with_ori({2}, ge::DT_INT64, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.SetAttr("max_gram_length", 2);
  op.SetAttr("max_skip_count", 0);
  op.SetAttr("min_gram_length", 1);
  op.SetAttr("mode", "TF");
  auto verify_fail_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res, ge::GRAPH_FAILED);
}

TEST_F(TfIdfVectorizerTest, TfIdfVectorizer_verify_test11_failed) {
  ge::op::TfIdfVectorizer op;
  op.UpdateInputDesc("input",
                     create_desc_with_ori({2}, ge::DT_INT64, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.SetAttr("max_gram_length", 2);
  op.SetAttr("max_skip_count", 0);
  op.SetAttr("min_gram_length", 1);
  op.SetAttr("mode", "TF");
  op.SetAttr("ngram_counts", {0, 4});
  auto verify_fail_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res, ge::GRAPH_FAILED);
}

TEST_F(TfIdfVectorizerTest, TfIdfVectorizer_verify_test12_failed) {
  ge::op::TfIdfVectorizer op;
  op.UpdateInputDesc("input",
                     create_desc_with_ori({2}, ge::DT_INT64, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.SetAttr("max_gram_length", 2);
  op.SetAttr("max_skip_count", 0);
  op.SetAttr("min_gram_length", 1);
  op.SetAttr("mode", "TF");
  std::vector<int64_t> ngramCounts = {0, 4};
  op.SetAttr("ngram_counts", ngramCounts);
  std::vector<int64_t> ngramIndexes = {0, 1, 2};
  op.SetAttr("ngram_indexes", ngramIndexes);
  auto verify_fail_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res, ge::GRAPH_FAILED);
}

TEST_F(TfIdfVectorizerTest, TfIdfVectorizer_verify_test13_failed) {
  ge::op::TfIdfVectorizer op;
  op.UpdateInputDesc("input",
                     create_desc_with_ori({2}, ge::DT_INT64, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.SetAttr("max_gram_length", 2);
  op.SetAttr("max_skip_count", 0);
  op.SetAttr("min_gram_length", 1);
  op.SetAttr("mode", "TF");
  std::vector<int64_t> ngramCounts = {0, 4};
  op.SetAttr("ngram_counts", ngramCounts);
  std::vector<int64_t> ngramIndexes = {0, 1, 2};
  op.SetAttr("ngram_indexes", ngramIndexes);
  std::vector<std::string> poolStrings = {};
  op.SetAttr("pool_strings", poolStrings);
  auto verify_fail_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res, ge::GRAPH_FAILED);
}

TEST_F(TfIdfVectorizerTest, TfIdfVectorizer_verify_test14_failed) {
  ge::op::TfIdfVectorizer op;
  op.UpdateInputDesc("input",
                     create_desc_with_ori({2}, ge::DT_INT64, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.SetAttr("max_gram_length", 2);
  op.SetAttr("max_skip_count", 0);
  op.SetAttr("min_gram_length", 1);
  op.SetAttr("mode", "TF");
  std::vector<int64_t> ngramCounts = {0, 5};
  op.SetAttr("ngram_counts", ngramCounts);
  std::vector<int64_t> ngramIndexes = {0, 1, 2};
  op.SetAttr("ngram_indexes", ngramIndexes);
  std::vector<int64_t> poolInt64s = {2, 3, 5, 4};
  op.SetAttr("pool_int64s", poolInt64s);
  auto verify_fail_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res, ge::GRAPH_FAILED);
}

TEST_F(TfIdfVectorizerTest, TfIdfVectorizer_verify_test15_success) {
  ge::op::TfIdfVectorizer op;
  op.UpdateInputDesc("input", create_desc_with_ori({2}, ge::DT_INT64, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.SetAttr("max_gram_length", 2);
  op.SetAttr("max_skip_count", 0);
  op.SetAttr("min_gram_length", 1);
  op.SetAttr("mode", "TF");
  std::vector<int64_t> ngramCounts = {0, 2};
  op.SetAttr("ngram_counts", ngramCounts);
  std::vector<int64_t> ngramIndexes = {0, 1, 2};
  op.SetAttr("ngram_indexes", ngramIndexes);
  std::vector<int64_t> poolInt64s = {2, 3, 5, 4};
  op.SetAttr("pool_int64s", poolInt64s);
  auto verify_fail_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res, ge::GRAPH_SUCCESS);
}

TEST_F(TfIdfVectorizerTest, TfIdfVectorizer_verify_test16_failed) {
  ge::op::TfIdfVectorizer op;
  op.UpdateInputDesc("input",
                     create_desc_with_ori({2}, ge::DT_INT64, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.SetAttr("max_gram_length", 2);
  op.SetAttr("max_skip_count", 0);
  op.SetAttr("min_gram_length", 1);
  op.SetAttr("mode", "TF");
  std::vector<int64_t> ngramCounts = {0, 2};
  op.SetAttr("ngram_counts", ngramCounts);
  std::vector<int64_t> ngramIndexes = {0, 1, 2};
  op.SetAttr("ngram_indexes", ngramIndexes);
  std::vector<int64_t> poolInt64s = {2, 3, 5, 4};
  op.SetAttr("pool_int64s", poolInt64s);
  std::vector<float> weights = {1.0, 2.0};
  op.SetAttr("weights", weights);
  auto verify_fail_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res, ge::GRAPH_FAILED);
}

TEST_F(TfIdfVectorizerTest, TfIdfVectorizer_verify_test17_success) {
  ge::op::TfIdfVectorizer op;
  op.UpdateInputDesc("input", create_desc_with_ori({2}, ge::DT_INT64, ge::FORMAT_ND, {2}, ge::FORMAT_ND));
  op.SetAttr("max_gram_length", 2);
  op.SetAttr("max_skip_count", 0);
  op.SetAttr("min_gram_length", 1);
  op.SetAttr("mode", "TF");
  std::vector<int64_t> ngramCounts = {0, 2};
  op.SetAttr("ngram_counts", ngramCounts);
  std::vector<int64_t> ngramIndexes = {0, 1, 2};
  op.SetAttr("ngram_indexes", ngramIndexes);
  std::vector<int64_t> poolInt64s = {2, 3, 5, 4};
  op.SetAttr("pool_int64s", poolInt64s);
  std::vector<float> weights = {1.0, 2.0, 3.0};
  op.SetAttr("weights", weights);
  auto verify_fail_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res, ge::GRAPH_SUCCESS);
}


TEST_F(TfIdfVectorizerTest, TfIdfVectorizer_static_infer_test1_success) {
  ge::op::TfIdfVectorizer op;
  op.UpdateInputDesc("input",
                     create_desc_with_ori({2, 12}, ge::DT_INT64, ge::FORMAT_ND, {2, 12}, ge::FORMAT_ND));
  op.SetAttr("max_gram_length", 2);
  op.SetAttr("max_skip_count", 0);
  op.SetAttr("min_gram_length", 1);
  op.SetAttr("mode", "TF");
  std::vector<int64_t> ngramCounts = {0, 2};
  op.SetAttr("ngram_counts", ngramCounts);
  std::vector<int64_t> ngramIndexes = {0, 1, 2};
  op.SetAttr("ngram_indexes", ngramIndexes);
  std::vector<int64_t> poolInt64s = {2, 3, 5, 4};
  op.SetAttr("pool_int64s", poolInt64s);
  std::vector<float> weights = {1.0, 2.0, 3.0};
  op.SetAttr("weights", weights);
  auto infer_res = op.InferShapeAndType();
  EXPECT_EQ(infer_res, ge::GRAPH_SUCCESS);
  auto out_var_desc = op.GetOutputDesc("output");
  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_var_output_shape = {2, 3};
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
}

