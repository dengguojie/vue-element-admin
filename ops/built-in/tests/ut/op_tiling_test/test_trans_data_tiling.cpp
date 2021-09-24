#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "external/graph/operator.h"
#include "graph/utils/op_desc_utils.h"
#include "transformation_ops.h"
#include "array_ops.h"
#include "register/op_tiling_registry.h"
#include "op_log.h"
#include "op_tiling/op_tiling_util.h"

using namespace std;
using namespace ge;

class TransDataTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TransDataTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TransDataTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream& tiling_data) {
  auto data = tiling_data.str();
  string result;
  int32_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }

  return result;
}

static string to_string_int64(const std::stringstream& tiling_data) {
  auto data = tiling_data.str();
  string result;
  int64_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }

  return result;
}

const int64_t profiling_test_num = 0;
const std::map<std::string, ge::Format> string_to_format_map = {{"NCHW", FORMAT_NCHW},
                                                                {"NHWC", FORMAT_NHWC},
                                                                {"NCDHW", FORMAT_NCDHW},
                                                                {"HWCN", FORMAT_HWCN},
                                                                {"DHWCN", FORMAT_DHWCN},
                                                                {"NDHWC", FORMAT_NDHWC},
                                                                {"CHWN", FORMAT_CHWN},
                                                                {"ND", FORMAT_ND},
                                                                {"NC1HWC0", FORMAT_NC1HWC0},
                                                                {"FRACTAL_NZ", FORMAT_FRACTAL_NZ},
                                                                {"FRACTAL_Z", FORMAT_FRACTAL_Z},
                                                                {"FRACTAL_ZN", FORMAT_FRACTAL_Z},
                                                                {"FRACTAL_Z_3D", FORMAT_FRACTAL_Z_3D},
                                                                {"NDC1HWC0", FORMAT_NDC1HWC0}};

static ge::Format StringToFormat(std::string format_string) {
  auto find_it = string_to_format_map.find(format_string);
  if (find_it != string_to_format_map.end()) {
    return find_it->second;
  }
  return FORMAT_ND;
}

static DataType StringToDtype(std::string dtype_string) {
  auto find_it = optiling::STR_TO_DATATYPE.find(dtype_string);
  if (find_it != optiling::STR_TO_DATATYPE.end()) {
    return find_it->second;
  }
  return ge::DT_FLOAT16;
}

static void add_input_desc_by_idx(Operator& op, int64_t idx, std::vector<int64_t> input_shape,
                                  std::vector<int64_t> input_ori_shape, std::string data_dtype, std::string src_format,
                                  std::string src_ori_format) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  op_info->MutableInputDesc(idx)->SetShape(GeShape(input_shape));
  op_info->MutableInputDesc(idx)->SetOriginShape(GeShape(input_ori_shape));
  op_info->MutableInputDesc(idx)->SetFormat(StringToFormat(src_format));
  op_info->MutableInputDesc(idx)->SetOriginFormat(StringToFormat(src_ori_format));
  op_info->MutableInputDesc(idx)->SetDataType(StringToDtype(data_dtype));
}

static void add_output_desc_by_idx(Operator& op, int64_t idx, std::vector<int64_t> input_shape,
                                   std::vector<int64_t> input_ori_shape, std::string data_dtype, std::string src_format,
                                   std::string src_ori_format) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  op_info->MutableOutputDesc(idx)->SetShape(GeShape(input_shape));
  op_info->MutableOutputDesc(idx)->SetOriginShape(GeShape(input_ori_shape));
  op_info->MutableOutputDesc(idx)->SetFormat(StringToFormat(src_format));
  op_info->MutableOutputDesc(idx)->SetOriginFormat(StringToFormat(src_ori_format));
  op_info->MutableOutputDesc(idx)->SetDataType(StringToDtype(data_dtype));
}

static void run_case(std::vector<int64_t> input_shape, std::vector<int64_t> output_shape, std::string data_dtype,
                     std::string src_format, std::string dst_format, std::string compile_info,
                     std::string expect_tiling, std::string case_name) {
  OP_EVENT("OP_TILING_UTEST", "case_name = %s", case_name.c_str());
  OP_EVENT("OP_TILING_UTEST", "case_mode = %s_TO_%s", src_format.c_str(), dst_format.c_str());
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find("TransData");
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  auto test_op = op::TransData("TransData");
  add_input_desc_by_idx(test_op, 0, input_shape, input_shape, data_dtype, src_format, src_format);
  add_output_desc_by_idx(test_op, 0, output_shape, output_shape, data_dtype, dst_format, dst_format);

  optiling::utils::OpCompileInfo op_compile_info(case_name.c_str(), compile_info);

  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(test_op, op_compile_info, runInfo));
  if (expect_tiling != "") {
    EXPECT_EQ(to_string_int64(runInfo.GetAllTilingData()), expect_tiling);
  }
  for (int64_t i = 0; i < profiling_test_num; i++) {
    iter->second(test_op, op_compile_info, runInfo);
  }
}

TEST_F(TransDataTiling, TransData_tiling1) {
  std::vector<int64_t> input_shape = {1, 16, 7, 7};
  std::vector<int64_t> output_shape = {1, 1, 7, 7, 16};
  std::string dtype = "float16";
  std::string src_format = "NCHW";
  std::string dst_format = "NC1HWC0";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", \"ub_size\": 126464, "
      "\"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}";
  std::string expect_tiling =
      "1001 63232 0 1 784 784 3952 0 16 1 1 1 784 784 1 784 784 49 16 784 784 1 16 49 49 784 1 0 1 0 1 0 1 0 1 0 1 0 1 "
      "1 784 1 1 0 49 1 16 1 1 0 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling2) {
  std::vector<int64_t> input_shape = {2, 35, 68, 3};
  std::vector<int64_t> output_shape = {2, 1, 35, 68, 16};
  std::string dtype = "float16";
  std::string src_format = "NHWC";
  std::string dst_format = "NC1HWC0";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"NHWC\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", \"ub_size\": 126464, "
      "\"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}";
  std::string expect_tiling =
      "1010 63232 2 7140 38080 7140 38080 7140 38080 11856 63232 3 0 3952 1 247 16 3 38080 38080 16 3 3 1 0 1 1 10 157 "
      "1 0 1 1 1 1 10 157 1 0 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling3) {
  std::vector<int64_t> input_shape = {2, 17, 10, 11095};
  std::vector<int64_t> output_shape = {2, 17, 694, 1, 16, 16};
  std::string dtype = "float16";
  std::string src_format = "ND";
  std::string dst_format = "FRACTAL_NZ";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"ND\", \"dstFormat\": \"FRACTAL_NZ\", \"dType\": \"float16\", \"ub_size\": 126464, "
      "\"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}";
  std::string expect_tiling =
      "1010 63232 17 221900 355328 110950 177664 110950 177664 177520 256 11095 0 3952 1 1 16 3952 63232 256 16 7 3952 "
      "2 0 1 1 10 1 3 3191 2 1 1 1 10 1 3 3191 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling4) {
  std::vector<int64_t> input_shape = {2, 2, 1, 1, 16};
  std::vector<int64_t> output_shape = {2, 1, 1, 31};
  std::string dtype = "float16";
  std::string src_format = "NC1HWC0";
  std::string dst_format = "NHWC";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"NC1HWC0\", \"dstFormat\": \"NHWC\", \"dType\": \"float16\", \"ub_size\": 126464, "
      "\"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "2012 63232 0 1 1 16 64 62 1 1 1 0 0 0 1 1 1 0 0 0 1 16 31 16 31 2 1 16 1 32 32 15 2 32 31 64 62 1 1 16 1 1 0 1 "
      "3952 1 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling5) {
  std::vector<int64_t> input_shape = {100, 3, 7, 16, 16};
  std::vector<int64_t> output_shape = {100, 107, 42};
  std::string dtype = "float16";
  std::string src_format = "FRACTAL_NZ";
  std::string dst_format = "ND";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"FRACTAL_NZ\", \"dstFormat\": \"ND\", \"dType\": \"float16\", \"ub_size\": 126464, "
      "\"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "2011 63232 0 7 1 16 86016 71904 2 1 1 27 0 0 2 1 1 27 0 4 80 16 42 1280 3360 3 1 1792 1 5376 48 10 16 5376 4494 "
      "86016 71904 112 1 16 1 1 0 1 3952 0 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling6) {
  std::vector<int64_t> input_shape = {100, 2, 16, 16};
  std::vector<int64_t> output_shape = {19, 5, 1, 5, 63};
  std::string dtype = "float16";
  std::string src_format = "FRACTAL_Z_3D";
  std::string dst_format = "NDHWC";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"FRACTAL_Z_3D\", \"dstFormat\": \"NDHWC\", \"dType\": \"float16\", \"ub_size\": "
      "126464, \"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "2011 63232 0 2 0 16 256 25200 1 1 1 0 0 0 1 1 1 0 0 3 25 0 63 0 1575 4 1 2560 1 10240 64 15 16 16 1575 256 "
      "25200 5 1 512 5 5 10240 2 3952 1 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling7) {
  std::vector<int64_t> input_shape = {2, 2, 49, 49, 16};
  std::vector<int64_t> output_shape = {2, 30, 49, 49};
  std::string dtype = "float16";
  std::string src_format = "NC1HWC0";
  std::string dst_format = "NCHW";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"NC1HWC0\", \"dstFormat\": \"NCHW\", \"dType\": \"float16\", \"ub_size\": 126464, "
      "\"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "2001 63232 0 2 16 76832 72030 1 2 1 0 0 0 1 2 1 0 0 0 2401 1 1 3952 16 1 38416 2401 30 38416 2401 38416 38416 1 "
      "76832 72030 76832 72030 14 1 1 0 1 1 0 2 1 76832 1 1 0 2401 1 16 1 1 0 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling8) {
  std::vector<int64_t> input_shape = {8, 32, 2, 16, 16};
  std::vector<int64_t> output_shape = {8, 2, 2, 16, 16, 16};
  std::string dtype = "float16";
  std::string src_format = "NCDHW";
  std::string dst_format = "NDC1HWC0";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"NCDHW\", \"dstFormat\": \"NDC1HWC0\", \"dType\": \"float16\", \"ub_size\": 126976, "
      "\"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "1001 63488 0 8 16384 16384 3968 0 16 1 2 1 16384 16384 1 16384 16384 512 16 8192 4096 1 0 512 512 0 1 0 2 0 1 0 "
      "1 0 2 0 1 0 8 1 16384 1 1 0 256 1 16 2 256 8192 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling9) {
  std::vector<int64_t> input_shape = {3, 34, 34, 16, 2};
  std::vector<int64_t> output_shape = {3468, 1, 16, 16};
  std::string dtype = "float16";
  std::string src_format = "DHWCN";
  std::string dst_format = "FRACTAL_Z_3D";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"DHWCN\", \"dstFormat\": \"FRACTAL_Z_3D\", \"dType\": \"float16\", \"ub_size\": "
      "126976, \"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "1000 63488 0 31 3584 0 3968 0 16 2 1 1 32 0 16 512 0 2 16 32 295936 1 16 2 2 32 7 0 1 0 1 0 7 12 1 0 1 0 1156 1 "
      "256 3 1156 295936 2 1 16 1 1 0 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling10) {
  std::vector<int64_t> input_shape = {79, 79, 23, 31};
  std::vector<int64_t> output_shape = {12482, 2, 16, 16};
  std::string dtype = "float16";
  std::string src_format = "HWCN";
  std::string dst_format = "FRACTAL_Z";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"HWCN\", \"dstFormat\": \"FRACTAL_Z\", \"dType\": \"float16\", \"ub_size\": 126976, "
      "\"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "1000 63488 0 31 148304 106496 3968 7 16 1 1 1 713 512 16 11408 8192 31 16 496 3195392 1 16 31 31 496 13 0 2 7 1 "
      "0 1 1 2 7 1 0 6241 1 512 1 1 0 31 1 16 1 1 0 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

/*
TEST_F(TransDataTiling, TransData_tiling_NCHW2NHWC) {
  vector<int64_t> input_shape = {100, 17, 16, 16};
  vector<int64_t> output_shape = {100, 16, 16, 17};
  std::string src_format = "NCHW";
  std::string dtype = "float16";
  std::string dst_format = "NHWC";
  std::string compile_info = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  std::string expect_tiling = "";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling_NCHW2HWCN) {
  vector<int64_t> input_shape = {100, 17, 16, 16};
  vector<int64_t> output_shape = {16, 16, 17, 100};
  std::string src_format = "NCHW";
  std::string dtype = "float16";
  std::string dst_format = "HWCN";
  std::string compile_info = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  std::string expect_tiling = "";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling_NHWC2NCHW) {
  vector<int64_t> input_shape = {100, 17, 16, 16};
  vector<int64_t> output_shape = {100, 16, 17, 16};
  std::string src_format = "NHWC";
  std::string dtype = "float16";
  std::string dst_format = "NCHW";
  std::string compile_info = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  std::string expect_tiling = "";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling_NHWC2HWCN) {
  vector<int64_t> input_shape = {100, 17, 16, 16};
  vector<int64_t> output_shape = {17, 16, 16, 100};
  std::string src_format = "NHWC";
  std::string dtype = "float16";
  std::string dst_format = "HWCN";
  std::string compile_info = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  std::string expect_tiling = "";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling_HWCN2NCHW) {
  vector<int64_t> input_shape = {100, 17, 16, 16};
  vector<int64_t> output_shape = {16, 16, 100, 17};
  std::string src_format = "HWCN";
  std::string dtype = "float16";
  std::string dst_format = "NCHW";
  std::string compile_info = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  std::string expect_tiling = "";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling_HWCN2NHWC) {
  vector<int64_t> input_shape = {100, 17, 16, 16};
  vector<int64_t> output_shape = {16, 100, 17, 16};
  std::string src_format = "HWCN";
  std::string dtype = "float16";
  std::string dst_format = "NHWC";
  std::string compile_info = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  std::string expect_tiling = "";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling_CHWN2NCHW) {
  vector<int64_t> input_shape = {100, 17, 16, 16};
  vector<int64_t> output_shape = {16, 100, 17, 16};
  std::string src_format = "CHWN";
  std::string dtype = "float16";
  std::string dst_format = "NCHW";
  std::string compile_info = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  std::string expect_tiling = "";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling_CHWN2NHWC) {
  vector<int64_t> input_shape = {100, 17, 16, 16};
  vector<int64_t> output_shape = {16, 17, 16, 100};
  std::string src_format = "CHWN";
  std::string dtype = "float16";
  std::string dst_format = "NHWC";
  std::string compile_info = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  std::string expect_tiling = "";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling_CHWN2HWCN) {
  vector<int64_t> input_shape = {100, 17, 16, 16};
  vector<int64_t> output_shape = {17, 16, 100, 16};
  std::string src_format = "CHWN";
  std::string dtype = "float16";
  std::string dst_format = "HWCN";
  std::string compile_info = "{\"vars\": {\"core_num\":32, \"ub_size\":8192, \"dtype\":\"float16\"}}";
  std::string expect_tiling = "";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
this->test_info_->name());
}
*/
TEST_F(TransDataTiling, TransData_tiling11) {
  std::vector<int64_t> input_shape = {2, 7, 11, 11, 16};
  std::vector<int64_t> output_shape = {7, 1, 11, 11, 1, 16, 16};
  std::string dtype = "float16";
  std::string src_format = "NDHWC";
  std::string dst_format = "FRACTAL_Z_3D";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"NDHWC\", \"dstFormat\": \"FRACTAL_Z_3D\", \"dType\": \"float16\", \"ub_size\": "
      "126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}";
  std::string expect_tiling =
      "1011 63232 4 1 3952 0 27104 32 13552 2 3952 3952 247 0 16 30976 30976 16 0 16 1 0 1 0 1 0 1 0 1 106 1 0 121 1 "
      "256 7 121 30976 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling12) {
  std::vector<int64_t> input_shape = {2, 7, 11, 11, 16};
  std::vector<int64_t> output_shape = {2, 7, 1, 11, 11, 16};
  std::string dtype = "float16";
  std::string src_format = "NDHWC";
  std::string dst_format = "NDC1HWC0";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"NDHWC\", \"dstFormat\": \"NDC1HWC0\", \"dType\": \"float16\", \"ub_size\": 126464, "
      "\"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}";
  std::string expect_tiling =
      "1010 63232 14 1936 1936 1936 1936 1936 1936 63232 63232 16 5 3952 1 247 16 16 1936 1936 16 0 16 1 0 0 1 1 121 1 "
      "0 1 0 0 1 1 121 1 0 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling13) {
  std::vector<int64_t> input_shape = {20, 3, 300, 300, 16};
  std::vector<int64_t> output_shape = {270000, 2, 16, 16};
  std::string dtype = "float16";
  std::string src_format = "NC1HWC0";
  std::string dst_format = "FRACTAL_Z";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"NC1HWC0\", \"dstFormat\": \"FRACTAL_Z\", \"dType\": \"float16\", \"ub_size\": "
      "126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}";
  std::string expect_tiling =
      "1011 63232 32 1 138320 0 69120000 256 4320000 16 3952 3952 247 0 16 138240000 138240000 16 0 16 2 4 35 0 1 0 2 "
      "4 9 29 1 0 270000 1 512 1 270000 138240000 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling_NCDHW2NDC1HWC0) {
  std::vector<int64_t> input_shape = {2, 7, 11, 11, 16};
  std::vector<int64_t> output_shape = {2, 11, 1, 11, 16, 16};
  std::string dtype = "float16";
  std::string src_format = "NCDHW";
  std::string dst_format = "NDC1HWC0";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"NCDHW\", \"dstFormat\": \"NDC1HWC0\", \"dType\": \"float16\", \"ub_size\": 126976, "
      "\"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "1001 63488 0 2 13552 30976 3968 7 16 1 2 1 13552 30976 1 13552 30976 1936 16 30976 2816 1 0 1936 1936 0 1 0 1 7 "
      "1 0 1 0 1 7 1 0 2 1 30976 1 1 0 176 1 16 11 176 2816 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling_NCDHW2NDC1HWC0_C) {
  std::vector<int64_t> input_shape = {2, 42767, 11, 11, 16};
  std::vector<int64_t> output_shape = {2, 11, 2673, 11, 16, 16};
  std::string dtype = "float";
  std::string src_format = "NCDHW";
  std::string dst_format = "NDC1HWC0";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"NCDHW\", \"dstFormat\": \"NDC1HWC0\", \"dType\": \"float\", \"ub_size\": 63488, "
      "\"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "1000 31744 1 32 2601984 236544 1984 15 16 1 2 1 82796912 82798848 2 165593824 165597696 1936 16 30976 2816 1 0 "
      "120 120 0 1 0 84 0 17 16 1 0 69 15 17 16 2 1 82798848 1 1 0 176 1 16 11 176 7527168 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling_NCDHW2NDC1HWC0_Cl) {
  std::vector<int64_t> input_shape = {42767, 7, 11, 11, 16};
  std::vector<int64_t> output_shape = {42767, 11, 1, 11, 16, 16};
  std::string dtype = "int32";
  std::string src_format = "NCDHW";
  std::string dst_format = "NDC1HWC0";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"NCDHW\", \"dstFormat\": \"NDC1HWC0\", \"dType\": \"int32\", \"ub_size\": 63488, "
      "\"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "1000 31744 0 32 18213888 41631744 1984 7 16 1 2 1 13552 30976 16 216832 495616 1936 16 30976 2816 1 0 120 120 0 "
      "84 0 1 7 17 16 69 15 1 7 17 16 42767 1 30976 1 1 0 176 1 16 11 176 2816 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling_NCDHW2NDC1HWC0_Cr) {
  std::vector<int64_t> input_shape = {2, 7, 42767, 11, 16};
  std::vector<int64_t> output_shape = {2, 42767, 1, 11, 16, 32};
  std::string dtype = "int8";
  std::string src_format = "NCDHW";
  std::string dst_format = "NDC1HWC0";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"NCDHW\", \"dstFormat\": \"NDC1HWC0\", \"dType\": \"int8\", \"ub_size\": 253952, "
      "\"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "1001 126976 2 32 238080 0 3968 7 32 1 2 1 52688944 240863744 1 52688944 240863744 7526992 32 240863744 5632 1 0 "
      "3968 3968 0 2 0 1 7 60 0 2 0 1 7 37 3664 2 1 240863744 1 1 0 176 1 32 42767 176 5632 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling_HWCN2FRACTALZN) {
  std::vector<int64_t> input_shape = {79, 79, 23, 31};
  std::vector<int64_t> output_shape = {12482, 2, 16, 16};
  std::string dtype = "float16";
  std::string src_format = "HWCN";
  std::string dst_format = "FRACTAL_ZN";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"HWCN\", \"dstFormat\": \"FRACTAL_ZN\", \"dType\": \"float16\", \"ub_size\": 126976, "
      "\"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "1000 63488 0 31 148304 106496 3968 7 16 1 1 1 713 512 16 11408 8192 31 16 496 3195392 1 16 31 31 496 13 0 2 7 1 "
      "0 1 1 2 7 1 0 6241 1 512 1 1 0 31 1 16 1 1 0 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling_ND2FRACTALZ_001) {
  std::vector<int64_t> input_shape = {79, 23, 13, 71};
  std::vector<int64_t> output_shape = {1817, 5, 16, 16};
  std::string dtype = "float16";
  std::string src_format = "ND";
  std::string dst_format = "FRACTAL_Z";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"ND\", \"dstFormat\": \"FRACTAL_Z\", \"dType\": \"float16\", \"ub_size\": 126976, "
      "\"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "1001 63488 0 32 52611 72960 3968 13 16 1 1 1 923 1280 1 923 1280 71 16 1136 1280 1 16 71 71 1136 57 0 1 13 1 0 "
      "50 0 1 13 1 0 1817 1 1280 1 1 0 71 1 16 1 1 0 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling_ND2FRACTALZN) {
  std::vector<int64_t> input_shape = {42767};
  std::vector<int64_t> output_shape = {1, 2673, 16, 16};
  std::string dtype = "float16";
  std::string src_format = "ND";
  std::string dst_format = "FRACTAL_ZN";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"ND\", \"dstFormat\": \"FRACTAL_ZN\", \"dType\": \"float16\", \"ub_size\": 126976, "
      "\"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "1001 63488 2 11 3968 63488 3968 1 16 1 1 1 42767 684288 1 42767 684288 42767 16 684272 684288 1 16 3968 3968 "
      "63488 1 0 1 1 1 0 1 0 1 1 1 3087 1 1 684288 1 1 0 42767 1 16 1 1 0 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling_ND2FRACTALZ_002) {
  std::vector<int64_t> input_shape = {1280, 1280};
  std::vector<int64_t> output_shape = {80, 80, 16, 16};
  std::string dtype = "float16";
  std::string src_format = "ND";
  std::string dst_format = "FRACTAL_Z";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"ND\", \"dstFormat\": \"FRACTAL_Z\", \"dType\": \"float16\", \"ub_size\": 126976, "
      "\"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "1001 63488 1 27 61440 61440 3968 0 16 1 1 1 1638400 1638400 1 1638400 1638400 1280 16 20480 20480 1 16 1280 "
      "1280 20480 1 0 3 0 1 0 1 0 2 0 1 0 1 1 1638400 1 1 0 1280 1 16 1 1 0 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling_NCHW2FRACTALZ) {
  std::vector<int64_t> input_shape = {1280, 42767, 31, 4};
  std::vector<int64_t> output_shape = {331452, 80, 16, 16};
  std::string dtype = "float16";
  std::string src_format = "NCHW";
  std::string dst_format = "FRACTAL_Z";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"FRACTAL_Z\", \"dType\": \"float16\", \"ub_size\": 126976, "
      "\"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "1001 63488 1 32 166656 213319680 3968 15 16 1 1 0 5303108 16 31 164396348 496 124 16 1984 2539520 1 20480 124 "
      "124 2539520 42 9 84 0 1 0 42 9 69 15 1 0 1280 1 16 1 1 0 124 1 20480 1 1 0 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling_NCHW2FRACTALZN) {
  std::vector<int64_t> input_shape = {1280, 42767, 31, 4};
  std::vector<int64_t> output_shape = {331452, 80, 16, 16};
  std::string dtype = "float16";
  std::string src_format = "NCHW";
  std::string dst_format = "FRACTAL_ZN";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"FRACTAL_ZN\", \"dType\": \"float16\", \"ub_size\": 126976, "
      "\"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "1001 63488 1 32 166656 213319680 3968 15 16 1 1 0 5303108 16 31 164396348 496 124 16 1984 2539520 1 20480 124 "
      "124 2539520 42 9 84 0 1 0 42 9 69 15 1 0 1280 1 16 1 1 0 124 1 20480 1 1 0 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_tiling_NCDHW2FRACTALZ3D) {
  std::vector<int64_t> input_shape = {1280, 2, 427, 31, 4};
  std::vector<int64_t> output_shape = {52948, 80, 16, 16};
  std::string dtype = "float16";
  std::string src_format = "NCDHW";
  std::string dst_format = "FRACTAL_Z_3D";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"NCDHW\", \"dstFormat\": \"FRACTAL_Z_3D\", \"dType\": \"float16\", \"ub_size\": "
      "126976, \"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "1001 63488 0 32 4235840 640 3968 2 16 1 2 0 105896 16 1 105896 16 52948 16 847168 2539520 1 0 3968 3968 0 40 0 "
      "1 2 14 1364 40 0 1 2 14 1364 1280 1 16 1 1 0 124 1 20480 427 124 2539520 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_NDC1HWC02NCDHW) {
  std::vector<int64_t> input_shape = {3, 4, 5, 17, 17, 16};
  std::vector<int64_t> output_shape = {3, 80, 4, 17, 17};
  std::string dtype = "float16";
  std::string src_format = "NDC1HWC0";
  std::string dst_format = "NCDHW";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"NDC1HWC0\", \"dstFormat\": \"NCDHW\", \"dType\": \"float16\", \"ub_size\": 126464, "
      "\"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "2001 63232 0 3 16 92480 92480 1 2 1 0 2 0 1 2 1 0 2 0 1156 3 1 3952 0 1 0 1156 80 4624 1156 13872 55488 1 92480 "
      "92480 92480 92480 0 2 1 0 1 1 0 3 1 92480 1 1 0 289 1 16 4 289 23120 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_FRAZ3D2NCDHW) {
  std::vector<int64_t> input_shape = {5780, 1, 16, 16};
  std::vector<int64_t> output_shape = {3, 80, 4, 17, 17};
  std::string dtype = "float16";
  std::string src_format = "FRACTAL_Z_3D";
  std::string dst_format = "NCDHW";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"FRACTAL_Z_3D\", \"dstFormat\": \"NCDHW\", \"dType\": \"float16\", \"ub_size\": "
      "126464, \"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "2001 63232 0 3 16 16 92480 1 2 1 0 2 0 1 2 1 0 2 0 1156 3 1 3952 0 1 0 1156 80 73984 1156 221952 55488 1 16 "
      "92480 16 92480 0 2 1 0 1 0 0 3 1 16 1 1 0 289 1 256 4 289 369920 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_FRAZ2HWCN) {
  std::vector<int64_t> input_shape = {1445, 1, 16, 16};
  std::vector<int64_t> output_shape = {17, 17, 80, 3};
  std::string dtype = "float16";
  std::string src_format = "FRACTAL_Z";
  std::string dst_format = "HWCN";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"FRACTAL_Z\", \"dstFormat\": \"HWCN\", \"dType\": \"float16\", \"ub_size\": 126464, "
      "\"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "2003 63232 0 19 16 4096 3840 1 1 1 0 0 0 1 1 1 0 0 1 3 5 16 3952 16 1 48 3 80 73984 3 369920 240 1 256 240 4096 "
      "3840 0 1 1 0 1 1 240 289 1 256 1 1 0 3 1 16 1 1 0 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_FRAZ3D2DHWCN) {
  std::vector<int64_t> input_shape = {5780, 1, 16, 16};
  std::vector<int64_t> output_shape = {4, 17, 17, 80, 3};
  std::string dtype = "float16";
  std::string src_format = "FRACTAL_Z_3D";
  std::string dst_format = "DHWCN";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"FRACTAL_Z_3D\", \"dstFormat\": \"DHWCN\", \"dType\": \"float16\", \"ub_size\": "
      "126464, \"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "2003 63232 0 25 16 0 11520 1 1 3 0 0 0 1 1 1 0 0 4 3 5 16 3952 16 1 48 3 80 73984 3 369920 240 1 0 240 0 3840 0 "
      "1 2 0 1 1 960 289 1 256 4 289 369920 3 1 16 1 1 0 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_FRAZ2NCHW) {
  std::vector<int64_t> input_shape = {1445, 1, 16, 16};
  std::vector<int64_t> output_shape = {3, 80, 17, 17};
  std::string dtype = "float16";
  std::string src_format = "FRACTAL_Z";
  std::string dst_format = "NCHW";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"FRACTAL_Z\", \"dstFormat\": \"NCHW\", \"dType\": \"float16\", \"ub_size\": 126464, "
      "\"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "2001 63232 0 2 16 32 46240 1 1 1 0 0 0 1 1 1 0 0 1 289 5 2 3952 256 1 73984 289 80 73984 289 369920 23120 1 16 "
      "23120 32 46240 0 1 1 0 1 0 23120 3 1 16 1 1 0 289 1 256 1 1 0 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_FRAZ2ND) {
  std::vector<int64_t> input_shape = {40, 3, 16, 16};
  std::vector<int64_t> output_shape = {640, 37};
  std::string dtype = "float16";
  std::string src_format = "FRACTAL_Z";
  std::string dst_format = "ND";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"FRACTAL_Z\", \"dstFormat\": \"ND\", \"dType\": \"float16\", \"ub_size\": 126464, "
      "\"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "2002 63232 1 8 16 3840 2960 1 1 1 0 0 0 1 1 1 0 0 0 37 5 1 3952 16 1 592 37 640 768 37 3840 2960 1 30720 23680 "
      "30720 23680 0 1 1 0 0 1 0 1 1 30720 1 1 0 37 1 16 1 1 0 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_NZ2NC1HWC0) {
  std::vector<int64_t> input_shape = {9, 3, 16, 16};
  std::vector<int64_t> output_shape = {35, 1, 3, 3, 16};
  std::string dtype = "float16";
  std::string src_format = "FRACTAL_NZ";
  std::string dst_format = "NC1HWC0";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"FRACTAL_NZ\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", \"ub_size\": "
      "126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}";
  std::string expect_tiling =
      "2010 63232 0 18 0 16 32 288 1 1 2 0 0 0 1 1 1 0 0 0 9 0 16 0 144 1 1 6912 1 6912 16 0 1 16 144 16 144 9 1 768 1 "
      "9 6912 2 3952 1 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_NDC1HWC02NDHWC) {
  std::vector<int64_t> input_shape = {1, 2, 2, 1, 1, 16};
  std::vector<int64_t> output_shape = {1, 2, 1, 1, 31};
  std::string dtype = "float16";
  std::string src_format = "NDC1HWC0";
  std::string dst_format = "NDHWC";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"NDC1HWC0\", \"dstFormat\": \"NDHWC\", \"dType\": \"float16\", \"ub_size\": 126464, "
      "\"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "2012 63232 0 1 1 16 64 62 1 1 1 0 0 0 1 1 1 0 0 0 1 16 31 16 31 2 1 16 1 32 32 15 2 32 31 64 62 1 1 16 1 1 0 1 "
      "3952 1 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_NZ3D_2_NDHWC) {
  std::vector<int64_t> input_shape = {1024, 1, 16, 16};
  std::vector<int64_t> output_shape = {10, 8, 16, 8, 11};
  std::string dtype = "float";
  std::string src_format = "FRACTAL_Z_3D";
  std::string dst_format = "NDHWC";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"FRACTAL_Z_3D\", \"dstFormat\": \"NDHWC\", \"dType\": \"float\", \"ub_size\": "
      "65280, \"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "2011 32640 2 9 0 16 0 1320 1 1 1 0 0 0 1 1 1 64 0 0 120 0 11 0 1320 1 1 32768 1 32768 16 11 10 16 11264 160 "
      "112640 128 1 256 8 128 32768 2 2040 0 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_NZ3D_2_NDHWC_1) {
  std::vector<int64_t> input_shape = {6250, 1, 16, 16};
  std::vector<int64_t> output_shape = {1, 1, 1, 1, 100000};
  std::string dtype = "float16";
  std::string src_format = "FRACTAL_Z_3D";
  std::string dst_format = "NDHWC";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"FRACTAL_Z_3D\", \"dstFormat\": \"NDHWC\", \"dType\": \"float16\", \"ub_size\": "
      "130560, \"block_dim\": 32, \"input_size\": -1, \"hidden_size\": -1, \"group\": 1}}";
  std::string expect_tiling =
      "2010 65280 1 2 0 16 1044480 65280 1 1 1 0 0 0 1 1 1 0 2170 0 1 0 100000 0 100000 4080 0 256 1 1044480 65280 0 1 "
      "16 100000 16 100000 1 1 256 1 1 1600000 2 4080 1 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, ND_2_NZ) {
  std::vector<int64_t> input_shape = {38400, 54};
  std::vector<int64_t> output_shape = {4, 2400, 16, 16};
  std::string dtype = "float16";
  std::string src_format = "ND";
  std::string dst_format = "FRACTAL_NZ";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"ND\", \"dstFormat\": \"FRACTAL_NZ\", \"dType\": \"float16\", \"ub_size\": 130560, "
      "\"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}";
  std::string expect_tiling =
      "1010 65280 20 108864 32256 2073600 2457600 2073600 2457600 54432 16128 54 0 4080 1 63 16 54 2457600 614400 16 6 "
      "54 1 1 1 2 0 63 1 0 1 1 1 1 2 33 1 0 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, ND_2_NZ_1) {
  std::vector<int64_t> input_shape = {2, 10000};
  std::vector<int64_t> output_shape = {625, 1, 16, 16};
  std::string dtype = "float16";
  std::string src_format = "ND";
  std::string dst_format = "FRACTAL_NZ";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"ND\", \"dstFormat\": \"FRACTAL_NZ\", \"dType\": \"float16\", \"ub_size\": 130560, "
      "\"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}";
  std::string expect_tiling =
      "1010 65280 3 4080 65280 20000 160000 20000 160000 10000 16 10000 5 4080 1 1 1 4080 65280 256 16 0 4080 1 0 0 2 "
      "0 1 1 0 1 0 0 2 0 1 1 1840 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_NDHWC_2_NZ3D) {
  std::vector<int64_t> input_shape = {1, 1, 1, 1, 47};
  std::vector<int64_t> output_shape = {1, 3, 1, 1, 1, 16, 16};
  std::string dtype = "float16";
  std::string src_format = "NDHWC";
  std::string dst_format = "FRACTAL_Z_3D";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"NDHWC\", \"dstFormat\": \"FRACTAL_Z_3D\", \"dType\": \"float16\", \"ub_size\": "
      "130560, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}";
  std::string expect_tiling =
      "1011 65280 1 0 47 16 47 16 47 1 47 4080 1 0 47 768 256 16 15 47 1 0 1 0 1 0 1 0 1 0 1 0 1 1 256 1 1 768 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}

TEST_F(TransDataTiling, TransData_NDHWC_2_NZ3D_1) {
  std::vector<int64_t> input_shape = {100, 1, 1, 1, 47};
  std::vector<int64_t> output_shape = {1, 3, 1, 1, 7, 16, 16};
  std::string dtype = "float16";
  std::string src_format = "NDHWC";
  std::string dst_format = "FRACTAL_Z_3D";
  std::string compile_info =
      "{\"vars\": {\"srcFormat\": \"NDHWC\", \"dstFormat\": \"FRACTAL_Z_3D\", \"dType\": \"float16\", \"ub_size\": "
      "130560, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}";
  std::string expect_tiling =
      "1011 65280 7 0 752 256 752 256 47 16 47 4080 1 0 47 5376 1792 16 15 47 1 0 1 0 1 0 1 4 1 0 1 0 1 1 1792 1 1 "
      "5376 ";
  run_case(input_shape, output_shape, dtype, src_format, dst_format, compile_info, expect_tiling,
           this->test_info_->name());
}
