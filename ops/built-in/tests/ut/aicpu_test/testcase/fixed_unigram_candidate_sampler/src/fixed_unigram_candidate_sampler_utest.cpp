/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "aicpu_read_file.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_FIXED_UNIGRAM_CANDIDATE_SAMPLER_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, num_true, num_sampled,      \
                       unique, range_max, vocab_file, unigrams, seed)         \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();            \
  NodeDefBuilder(node_def.get(), "FixedUnigramCandidateSampler",              \
                 "FixedUnigramCandidateSampler")                              \
      .Input({"true_classes", data_types[0], shapes[0], datas[0]})            \
      .Output({"sampled_candidates", data_types[1], shapes[1], datas[1]})     \
      .Output({"true_expected_count", data_types[2], shapes[2], datas[2]})    \
      .Output({"sampled_expected_count", data_types[3], shapes[3], datas[3]}) \
      .Attr("num_true", num_true)                                             \
      .Attr("num_sampled", num_sampled)                                       \
      .Attr("unique", unique)                                                 \
      .Attr("range_max", range_max)                                           \
      .Attr("vocab_file", vocab_file)                                         \
      .Attr("distortion", 1.0)                                                \
      .Attr("num_reserved_ids", 0)                                            \
      .Attr("num_shards", 1)                                                  \
      .Attr("shard", 0)                                                       \
      .Attr("unigrams", unigrams)                                             \
      .Attr("seed", seed[0])                                                  \
      .Attr("seed2", seed[1])

TEST_F(TEST_FIXED_UNIGRAM_CANDIDATE_SAMPLER_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT64, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 5}, {2}, {1, 5}, {2}};
  int32_t true_classes[5] = {1, 2, 3, 4, 6};
  int64_t sampled_candidates[2] = {0};
  float true_expected_count[5] = {0.0};
  float sampled_expected_count[2] = {0.0};

  int32_t num_true = 5;
  int32_t num_sampled = 2;
  bool unique = false;
  int32_t range_max = 5;
  string vocab_file = "";
  vector<float> unigrams{0.1, 0.2, 0.3, 0.1, 0.3};
  vector<int> seed{87654321, 0};

  vector<void *> datas = {(void *)true_classes, (void *)sampled_candidates,
                          (void *)true_expected_count,
                          (void *)sampled_expected_count};

  CREATE_NODEDEF(shapes, data_types, datas, num_true, num_sampled, unique,
                 range_max, vocab_file, unigrams, seed);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_FIXED_UNIGRAM_CANDIDATE_SAMPLER_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 1, 5}, {2}, {1, 5}, {2}};
  int64_t true_classes[5] = {1, 2, 3, 4, 6};
  int64_t sampled_candidates[2] = {0};
  float true_expected_count[5] = {0.0};
  float sampled_expected_count[2] = {0.0};

  int32_t num_true = 5;
  int32_t num_sampled = 2;
  bool unique = false;
  int32_t range_max = 5;
  string vocab_file = "";
  vector<float> unigrams{0.1, 0.2, 0.3, 0.1, 0.3};
  vector<int> seed{87654321, 0};

  vector<void *> datas = {(void *)true_classes, (void *)sampled_candidates,
                          (void *)true_expected_count,
                          (void *)sampled_expected_count};

  CREATE_NODEDEF(shapes, data_types, datas, num_true, num_sampled, unique,
                 range_max, vocab_file, unigrams, seed);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_FIXED_UNIGRAM_CANDIDATE_SAMPLER_UT, NUM_TRUE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 5}, {2}, {1, 5}, {2}};
  int64_t true_classes[5] = {1, 2, 3, 4, 6};
  int64_t sampled_candidates[2] = {0};
  float true_expected_count[5] = {0.0};
  float sampled_expected_count[2] = {0.0};

  int32_t num_true = 4;
  int32_t num_sampled = 2;
  bool unique = false;
  int32_t range_max = 5;
  string vocab_file = "";
  vector<float> unigrams{0.1, 0.2, 0.3, 0.1, 0.3};
  vector<int> seed{87654321, 0};

  vector<void *> datas = {(void *)true_classes, (void *)sampled_candidates,
                          (void *)true_expected_count,
                          (void *)sampled_expected_count};

  CREATE_NODEDEF(shapes, data_types, datas, num_true, num_sampled, unique,
                 range_max, vocab_file, unigrams, seed);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_FIXED_UNIGRAM_CANDIDATE_SAMPLER_UT, RANGE_MAX_EXCEPTION) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 5}, {2}, {1, 5}, {2}};
  int64_t true_classes[5] = {1, 2, 3, 4, 6};
  int64_t sampled_candidates[2] = {0};
  float true_expected_count[5] = {0.0};
  float sampled_expected_count[2] = {0.0};

  int32_t num_true = 5;
  int32_t num_sampled = 2;
  bool unique = false;
  int32_t range_max = 6;
  string vocab_file = "";
  vector<float> unigrams{0.1, 0.2, 0.3, 0.1, 0.3};
  vector<int> seed{87654321, 0};

  vector<void *> datas = {(void *)true_classes, (void *)sampled_candidates,
                          (void *)true_expected_count,
                          (void *)sampled_expected_count};

  CREATE_NODEDEF(shapes, data_types, datas, num_true, num_sampled, unique,
                 range_max, vocab_file, unigrams, seed);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_FIXED_UNIGRAM_CANDIDATE_SAMPLER_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 5}, {2}, {1, 5}, {2}};
  int64_t sampled_candidates[2] = {0};
  float true_expected_count[5] = {0.0};
  float sampled_expected_count[2] = {0.0};

  int32_t num_true = 5;
  int32_t num_sampled = 2;
  bool unique = false;
  int32_t range_max = 5;
  string vocab_file = "";
  vector<float> unigrams{0.1, 0.2, 0.3, 0.1, 0.3};
  vector<int> seed{87654321, 0};

  vector<void *> datas = {(void *)nullptr, (void *)sampled_candidates,
                          (void *)true_expected_count,
                          (void *)sampled_expected_count};

  CREATE_NODEDEF(shapes, data_types, datas, num_true, num_sampled, unique,
                 range_max, vocab_file, unigrams, seed);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_FIXED_UNIGRAM_CANDIDATE_SAMPLER_UT, NUM_SAMPLED_EXCEPTION) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 5}, {2}, {1, 5}, {2}};
  int32_t true_classes[5] = {1, 2, 3, 4, 6};
  int64_t sampled_candidates[2] = {0};
  float true_expected_count[5] = {0.0};
  float sampled_expected_count[2] = {0.0};

  int32_t num_true = 5;
  int32_t num_sampled = 6;
  bool unique = true;
  int32_t range_max = 5;
  string vocab_file = "";
  vector<float> unigrams{0.1, 0.2, 0.3, 0.1, 0.3};
  vector<int> seed{87654321, 0};

  vector<void *> datas = {(void *)true_classes, (void *)sampled_candidates,
                          (void *)true_expected_count,
                          (void *)sampled_expected_count};

  CREATE_NODEDEF(shapes, data_types, datas, num_true, num_sampled, unique,
                 range_max, vocab_file, unigrams, seed);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_FIXED_UNIGRAM_CANDIDATE_SAMPLER_UT, IMIT_PHILOX_RANDOM) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 5}, {2}, {1, 5}, {2}};
  int64_t true_classes[5] = {1, 2, 3, 4, 6};
  int64_t sampled_candidates[2] = {0};
  float true_expected_count[5] = {0.0};
  float sampled_expected_count[2] = {0.0};

  int32_t num_true = 5;
  int32_t num_sampled = 2;
  bool unique = false;
  int32_t range_max = 5;
  string vocab_file = "";
  vector<float> unigrams{0.1, 0.2, 0.3, 0.1, 0.3};
  vector<int> seed{0, 0};

  vector<void *> datas = {(void *)true_classes, (void *)sampled_candidates,
                          (void *)true_expected_count,
                          (void *)sampled_expected_count};

  CREATE_NODEDEF(shapes, data_types, datas, num_true, num_sampled, unique,
                 range_max, vocab_file, unigrams, seed);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_FIXED_UNIGRAM_CANDIDATE_SAMPLER_UT, SEED_0_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 5}, {2}, {1, 5}, {2}};
  vector<string> data_files{
      "fixed_unigram_candidate_sampler/data/fucs_true_classes_1.txt",
      "fixed_unigram_candidate_sampler/data/fucs_unigrams_1.txt",
      "fixed_unigram_candidate_sampler/data/fucs_sampled_candidates_1.txt",
      "fixed_unigram_candidate_sampler/data/fucs_true_expected_count_1.txt",
      "fixed_unigram_candidate_sampler/data/fucs_sampled_expected_count_1.txt"};

  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  int64_t *input1 = new int64_t[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  data_path = ktestcaseFilePath + data_files[1];
  uint64_t unigrams_size = 5;
  float *unigrams_array = new float[unigrams_size];
  status = ReadFile(data_path, unigrams_array, unigrams_size);
  EXPECT_EQ(status, true);
  vector<float> unigrams;
  for (int i = 0; i < unigrams_size; i++) {
    unigrams.push_back(unigrams_array[i]);
  }

  uint64_t output1_size = CalTotalElements(shapes, 1);
  int64_t *output1 = new int64_t[output1_size];
  uint64_t output2_size = CalTotalElements(shapes, 2);
  float *output2 = new float[output2_size];
  uint64_t output3_size = CalTotalElements(shapes, 3);
  float *output3 = new float[output3_size];

  int32_t num_true = 5;
  int32_t num_sampled = 2;
  bool unique = true;
  int32_t range_max = 5;
  string vocab_file = "";
  vector<int> seed{87654321, 0};

  vector<void *> datas = {(void *)input1, (void *)output1, (void *)output2,
                          (void *)output3};

  CREATE_NODEDEF(shapes, data_types, datas, num_true, num_sampled, unique,
                 range_max, vocab_file, unigrams, seed);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[2];
  int64_t *output1_exp = new int64_t[output1_size];
  status = ReadFile(data_path, output1_exp, output1_size);
  EXPECT_EQ(status, true);

  data_path = ktestcaseFilePath + data_files[3];
  float *output2_exp = new float[output2_size];
  status = ReadFile(data_path, output2_exp, output2_size);
  EXPECT_EQ(status, true);

  data_path = ktestcaseFilePath + data_files[4];
  float *output3_exp = new float[output3_size];
  status = ReadFile(data_path, output3_exp, output3_size);
  EXPECT_EQ(status, true);

  bool compare1 = CompareResult(output1, output1_exp, output1_size);
  bool compare2 = CompareResult(output2, output2_exp, output2_size);
  bool compare3 = CompareResult(output3, output3_exp, output3_size);
  EXPECT_EQ(compare1 & compare2 & compare3, true);
  delete [] input1;
  delete [] unigrams_array;
  delete [] output1;
  delete [] output2;
  delete [] output3;
  delete [] output1_exp;
  delete [] output2_exp;
  delete [] output3_exp;
}

TEST_F(TEST_FIXED_UNIGRAM_CANDIDATE_SAMPLER_UT, SEED_1_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 6}, {3}, {1, 6}, {3}};
  vector<string> data_files{
      "fixed_unigram_candidate_sampler/data/fucs_true_classes_2.txt",
      "fixed_unigram_candidate_sampler/data/fucs_unigrams_2.txt",
      "fixed_unigram_candidate_sampler/data/fucs_sampled_candidates_2.txt",
      "fixed_unigram_candidate_sampler/data/fucs_true_expected_count_2.txt",
      "fixed_unigram_candidate_sampler/data/fucs_sampled_expected_count_2.txt"};

  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  int64_t *input1 = new int64_t[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  data_path = ktestcaseFilePath + data_files[1];
  uint64_t unigrams_size = 6;
  float *unigrams_array = new float[unigrams_size];
  status = ReadFile(data_path, unigrams_array, unigrams_size);
  EXPECT_EQ(status, true);
  vector<float> unigrams;
  for (int i = 0; i < unigrams_size; i++) {
    unigrams.push_back(unigrams_array[i]);
  }

  uint64_t output1_size = CalTotalElements(shapes, 1);
  int64_t *output1 = new int64_t[output1_size];
  uint64_t output2_size = CalTotalElements(shapes, 2);
  float *output2 = new float[output2_size];
  uint64_t output3_size = CalTotalElements(shapes, 3);
  float *output3 = new float[output3_size];

  int32_t num_true = 6;
  int32_t num_sampled = 3;
  bool unique = true;
  int32_t range_max = 6;
  string vocab_file = "";
  vector<int> seed{87654321, 1};

  vector<void *> datas = {(void *)input1, (void *)output1, (void *)output2,
                          (void *)output3};

  CREATE_NODEDEF(shapes, data_types, datas, num_true, num_sampled, unique,
                 range_max, vocab_file, unigrams, seed);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[2];
  int64_t *output1_exp = new int64_t[output1_size];
  status = ReadFile(data_path, output1_exp, output1_size);
  EXPECT_EQ(status, true);

  data_path = ktestcaseFilePath + data_files[3];
  float *output2_exp = new float[output2_size];
  status = ReadFile(data_path, output2_exp, output2_size);
  EXPECT_EQ(status, true);

  data_path = ktestcaseFilePath + data_files[4];
  float *output3_exp = new float[output3_size];
  status = ReadFile(data_path, output3_exp, output3_size);
  EXPECT_EQ(status, true);

  bool compare1 = CompareResult(output1, output1_exp, output1_size);
  bool compare2 = CompareResult(output2, output2_exp, output2_size);
  bool compare3 = CompareResult(output3, output3_exp, output3_size);
  EXPECT_EQ(compare1 & compare2 & compare3, true);
  delete [] input1;
  delete [] unigrams_array;
  delete [] output1;
  delete [] output2;
  delete [] output3;
  delete [] output1_exp;
  delete [] output2_exp;
  delete [] output3_exp;
}

TEST_F(TEST_FIXED_UNIGRAM_CANDIDATE_SAMPLER_UT, UNIQUE_FALSE_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 6}, {3}, {1, 6}, {3}};
  vector<string> data_files{
      "fixed_unigram_candidate_sampler/data/fucs_true_classes_3.txt",
      "fixed_unigram_candidate_sampler/data/fucs_unigrams_3.txt",
      "fixed_unigram_candidate_sampler/data/fucs_sampled_candidates_3.txt",
      "fixed_unigram_candidate_sampler/data/fucs_true_expected_count_3.txt",
      "fixed_unigram_candidate_sampler/data/fucs_sampled_expected_count_3.txt"};

  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  int64_t *input1 = new int64_t[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  data_path = ktestcaseFilePath + data_files[1];
  uint64_t unigrams_size = 6;
  float *unigrams_array = new float[unigrams_size];
  status = ReadFile(data_path, unigrams_array, unigrams_size);
  EXPECT_EQ(status, true);
  vector<float> unigrams;
  for (int i = 0; i < unigrams_size; i++) {
    unigrams.push_back(unigrams_array[i]);
  }

  uint64_t output1_size = CalTotalElements(shapes, 1);
  int64_t *output1 = new int64_t[output1_size];
  uint64_t output2_size = CalTotalElements(shapes, 2);
  float *output2 = new float[output2_size];
  uint64_t output3_size = CalTotalElements(shapes, 3);
  float *output3 = new float[output3_size];

  int32_t num_true = 6;
  int32_t num_sampled = 3;
  bool unique = false;
  int32_t range_max = 6;
  string vocab_file = "";
  vector<int> seed{87654321, 2};

  vector<void *> datas = {(void *)input1, (void *)output1, (void *)output2,
                          (void *)output3};

  CREATE_NODEDEF(shapes, data_types, datas, num_true, num_sampled, unique,
                 range_max, vocab_file, unigrams, seed);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[2];
  int64_t *output1_exp = new int64_t[output1_size];
  status = ReadFile(data_path, output1_exp, output1_size);
  EXPECT_EQ(status, true);

  data_path = ktestcaseFilePath + data_files[3];
  float *output2_exp = new float[output2_size];
  status = ReadFile(data_path, output2_exp, output2_size);
  EXPECT_EQ(status, true);

  data_path = ktestcaseFilePath + data_files[4];
  float *output3_exp = new float[output3_size];
  status = ReadFile(data_path, output3_exp, output3_size);
  EXPECT_EQ(status, true);

  bool compare1 = CompareResult(output1, output1_exp, output1_size);
  bool compare2 = CompareResult(output2, output2_exp, output2_size);
  bool compare3 = CompareResult(output3, output3_exp, output3_size);
  EXPECT_EQ(compare1 & compare2 & compare3, true);
  delete [] input1;
  delete [] unigrams_array;
  delete [] output1;
  delete [] output2;
  delete [] output3;
  delete [] output1_exp;
  delete [] output2_exp;
  delete [] output3_exp;
}

TEST_F(TEST_FIXED_UNIGRAM_CANDIDATE_SAMPLER_UT, LOAD_FILE_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 6}, {3}, {1, 6}, {3}};
  vector<string> data_files{
      "fixed_unigram_candidate_sampler/data/fucs_true_classes_4.txt",
      "fixed_unigram_candidate_sampler/data/fucs_unigrams_4.csv",
      "fixed_unigram_candidate_sampler/data/fucs_sampled_candidates_4.txt",
      "fixed_unigram_candidate_sampler/data/fucs_true_expected_count_4.txt",
      "fixed_unigram_candidate_sampler/data/fucs_sampled_expected_count_4.txt"};

  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  int64_t *input1 = new int64_t[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  vector<float> unigrams;

  uint64_t output1_size = CalTotalElements(shapes, 1);
  int64_t *output1 = new int64_t[output1_size];
  uint64_t output2_size = CalTotalElements(shapes, 2);
  float *output2 = new float[output2_size];
  uint64_t output3_size = CalTotalElements(shapes, 3);
  float *output3 = new float[output3_size];

  int32_t num_true = 6;
  int32_t num_sampled = 3;
  bool unique = false;
  int32_t range_max = 6;
  string vocab_file = ktestcaseFilePath + data_files[1];
  vector<int> seed{87654321, 2};

  vector<void *> datas = {(void *)input1, (void *)output1, (void *)output2,
                          (void *)output3};

  CREATE_NODEDEF(shapes, data_types, datas, num_true, num_sampled, unique,
                 range_max, vocab_file, unigrams, seed);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[2];
  int64_t *output1_exp = new int64_t[output1_size];
  status = ReadFile(data_path, output1_exp, output1_size);
  EXPECT_EQ(status, true);

  data_path = ktestcaseFilePath + data_files[3];
  float *output2_exp = new float[output2_size];
  status = ReadFile(data_path, output2_exp, output2_size);
  EXPECT_EQ(status, true);

  data_path = ktestcaseFilePath + data_files[4];
  float *output3_exp = new float[output3_size];
  status = ReadFile(data_path, output3_exp, output3_size);
  EXPECT_EQ(status, true);

  bool compare1 = CompareResult(output1, output1_exp, output1_size);
  bool compare2 = CompareResult(output2, output2_exp, output2_size);
  bool compare3 = CompareResult(output3, output3_exp, output3_size);
  EXPECT_EQ(compare1 & compare2 & compare3, true);
  delete [] input1;
  delete [] output1;
  delete [] output2;
  delete [] output3;
  delete [] output1_exp;
  delete [] output2_exp;
  delete [] output3_exp;
}