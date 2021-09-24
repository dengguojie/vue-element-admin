#include <vector>

#include <gtest/gtest.h>
#include <graph/utils/type_utils.h>
#include "register/op_tiling_registry.h"
#include "all_ops.h"
#include "test_common.h"
#include "op_tiling/op_tiling_util.h"
#include "common/utils/ut_op_util.h"

using namespace std;
using namespace ge;

class EmbeddingDenseGradTiling : public testing::Test {
  protected:
    static void SetUpTestCase() {
        std::cout << "EmbeddingDenseGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "EmbeddingDenseGradTiling TearDown" << std::endl;
    }
};

const int64_t profiling_test_num = 0;
static void run_case(std::vector<int64_t> input_shape_0, std::string data_dtype_0, 
                     std::vector<int64_t> input_shape_1, std::string data_dtype_1, 
                     std::string src_ori_format, std::string src_format, 
                     std::string compile_info, std::string expect_tiling, 
                     std::string case_name) {
  using namespace ut_util;
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find("EmbeddingDenseGrad");
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  auto test_op = op::EmbeddingDenseGrad("EmbeddingDenseGrad");

  TENSOR_INPUT_WITH_SHAPE(test_op, grad, input_shape_0, StringToDtype(data_dtype_0),
                          TypeUtils::SerialStringToFormat(src_ori_format), {});
  TransformerOpBaseFormat(test_op, "grad", TypeUtils::SerialStringToFormat(src_format));
  TENSOR_INPUT_WITH_SHAPE(test_op, indices, input_shape_1, StringToDtype(data_dtype_1),
                          TypeUtils::SerialStringToFormat(src_ori_format), {});
  TransformerOpBaseFormat(test_op, "indices", TypeUtils::SerialStringToFormat(src_format));

  optiling::utils::OpCompileInfo op_compile_info(case_name.c_str(), compile_info);

  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(test_op, op_compile_info, runInfo));
  if (expect_tiling != "") {
    EXPECT_EQ(to_string_int32(runInfo.GetAllTilingData()), expect_tiling);
  }
  for (int64_t i = 0; i < profiling_test_num; i++) {
    iter->second(test_op, op_compile_info, runInfo);
  }
}

TEST_F(EmbeddingDenseGradTiling, embedding_dense_grad_tiling_0) {
  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"num_weights\": 20000, \"padding_idx\": 20, \"scale_grad_by_freq\": 1}}";
  std::vector<int64_t> input_0{20000, 512};
  std::vector<int64_t> input_1{20000};
  std::vector<int64_t> output{20000, 512};
  std::string expect_tiling = "20000 512 1 32 ";
  std::string input_dtype_0 = "float32";
  std::string input_dtype_1 = "int32";
  std::string format = "ND";
  run_case(input_0, input_dtype_0, input_1, input_dtype_1, format, format, compileInfo, expect_tiling, this->test_info_->name());
}

TEST_F(EmbeddingDenseGradTiling, embedding_dense_grad_tiling_1) {
  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"num_weights\": 20000, \"padding_idx\": 20, \"scale_grad_by_freq\": 1}}";
  std::vector<int64_t> input_0{30000, 1024};
  std::vector<int64_t> input_1{30000};
  std::vector<int64_t> output{20000, 1024};
  std::string input_dtype_0 = "float32";
  std::string input_dtype_1 = "int32";
  std::string format = "ND";
  std::string expect_tiling = "30000 1024 1 32 ";
  run_case(input_0, input_dtype_0, input_1, input_dtype_1, format, format, compileInfo, expect_tiling, this->test_info_->name());
}

TEST_F(EmbeddingDenseGradTiling, embedding_dense_grad_tiling_2) {
  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"num_weights\": 20000, \"padding_idx\": 10, \"scale_grad_by_freq\": 0}}";
  std::vector<int64_t> input_0{10000, 768};
  std::vector<int64_t> input_1{10000};
  std::vector<int64_t> output{20000, 768};
  std::string input_dtype_0 = "float32";
  std::string input_dtype_1 = "int32";
  std::string format = "ND";
  std::string expect_tiling = "10000 768 1 32 ";
  run_case(input_0, input_dtype_0, input_1, input_dtype_1, format, format, compileInfo, expect_tiling, this->test_info_->name());
}
