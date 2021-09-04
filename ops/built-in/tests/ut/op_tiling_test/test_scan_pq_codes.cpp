#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class ScanPQCodesTiling : public testing::Test {
  protected:
  static void SetUpTestCase() {
    std::cout << "ScanPQCodesTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ScanPQCodesTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream &tiling_data) {
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

TEST_F(ScanPQCodesTiling, scan_pq_codes_tiling_0) {
  using namespace optiling;
  std::string op_name = "ScanPQCodes";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"core_nums\": 8, \"split_count\": 1, \"split_index\": 0}}";

  std::vector<int64_t> ivfShape{20480, 16};
  std::vector<int64_t> bucketListShape{4};
  std::vector<int64_t> pqDistanceShape{20480};

  TeOpTensor tensorIvf;
  tensorIvf.shape = ivfShape;
  tensorIvf.dtype = "uint8";

  TeOpTensor tensorBucketList;
  tensorBucketList.shape = bucketListShape;
  tensorBucketList.dtype = "int32";

  TeOpTensor tensorPQDistance;
  tensorPQDistance.shape = pqDistanceShape;
  tensorPQDistance.dtype = "float16";

  TeOpTensorArg tensorIvfArg;
  tensorIvfArg.tensor.push_back(tensorIvf);
  tensorIvfArg.arg_type = TA_SINGLE;
  
  TeOpTensorArg tensorBucketListArg;
  tensorBucketListArg.tensor.push_back(tensorBucketList);
  tensorBucketListArg.arg_type = TA_SINGLE;

  TeOpTensorArg tensorPQDistanceArg;
  tensorPQDistanceArg.tensor.push_back(tensorPQDistance);
  tensorPQDistanceArg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensorIvfArg);
  opParas.inputs.push_back(tensorBucketListArg);
  opParas.outputs.push_back(tensorPQDistanceArg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "scanPQCodes.key0";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "Tiling_data is: " << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string(runInfo.tiling_data), "4 1 0 4 0 ");
}

TEST_F(ScanPQCodesTiling, scan_pq_codes_tiling_1) {
  using namespace optiling;
  std::string op_name = "ScanPQCodes";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"core_nums\": 8, \"split_count\": 1, \"split_index\": 0}}";

  std::vector<int64_t> ivfShape{20480, 16};
  std::vector<int64_t> bucketListShape{8};
  std::vector<int64_t> pqDistanceShape{20480};

  TeOpTensor tensorIvf;
  tensorIvf.shape = ivfShape;
  tensorIvf.dtype = "uint8";

  TeOpTensor tensorBucketList;
  tensorBucketList.shape = bucketListShape;
  tensorBucketList.dtype = "int32";

  TeOpTensor tensorPQDistance;
  tensorPQDistance.shape = pqDistanceShape;
  tensorPQDistance.dtype = "float16";

  TeOpTensorArg tensorIvfArg;
  tensorIvfArg.tensor.push_back(tensorIvf);
  tensorIvfArg.arg_type = TA_SINGLE;
  
  TeOpTensorArg tensorBucketListArg;
  tensorBucketListArg.tensor.push_back(tensorBucketList);
  tensorBucketListArg.arg_type = TA_SINGLE;

  TeOpTensorArg tensorPQDistanceArg;
  tensorPQDistanceArg.tensor.push_back(tensorPQDistance);
  tensorPQDistanceArg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensorIvfArg);
  opParas.inputs.push_back(tensorBucketListArg);
  opParas.outputs.push_back(tensorPQDistanceArg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "scanPQCodes.key1";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "Tiling_data is: " << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string(runInfo.tiling_data), "8 1 0 8 0 ");
}

TEST_F(ScanPQCodesTiling, scan_pq_codes_tiling_2) {
  using namespace optiling;
  std::string op_name = "ScanPQCodes";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"core_nums\": 8, \"split_count\": 2, \"split_index\": 0}}";

  std::vector<int64_t> ivfShape{40960, 16};
  std::vector<int64_t> bucketListShape{2};
  std::vector<int64_t> pqDistanceShape{40960};

  TeOpTensor tensorIvf;
  tensorIvf.shape = ivfShape;
  tensorIvf.dtype = "uint8";

  TeOpTensor tensorBucketList;
  tensorBucketList.shape = bucketListShape;
  tensorBucketList.dtype = "int32";

  TeOpTensor tensorPQDistance;
  tensorPQDistance.shape = pqDistanceShape;
  tensorPQDistance.dtype = "float16";

  TeOpTensorArg tensorIvfArg;
  tensorIvfArg.tensor.push_back(tensorIvf);
  tensorIvfArg.arg_type = TA_SINGLE;
  
  TeOpTensorArg tensorBucketListArg;
  tensorBucketListArg.tensor.push_back(tensorBucketList);
  tensorBucketListArg.arg_type = TA_SINGLE;

  TeOpTensorArg tensorPQDistanceArg;
  tensorPQDistanceArg.tensor.push_back(tensorPQDistance);
  tensorPQDistanceArg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensorIvfArg);
  opParas.inputs.push_back(tensorBucketListArg);
  opParas.outputs.push_back(tensorPQDistanceArg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "scanPQCodes.key0";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "Tiling_data is: " << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string(runInfo.tiling_data), "2 1 0 2 0 ");
}
TEST_F(ScanPQCodesTiling, scan_pq_codes_tiling_3) {
  using namespace optiling;
  std::string op_name = "ScanPQCodes";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"core_nums\": 8, \"split_count\": 2, \"split_index\": 1}}";

  std::vector<int64_t> ivfShape{20480, 16};
  std::vector<int64_t> bucketListShape{2};
  std::vector<int64_t> pqDistanceShape{20480};

  TeOpTensor tensorIvf;
  tensorIvf.shape = ivfShape;
  tensorIvf.dtype = "uint8";

  TeOpTensor tensorBucketList;
  tensorBucketList.shape = bucketListShape;
  tensorBucketList.dtype = "int32";

  TeOpTensor tensorPQDistance;
  tensorPQDistance.shape = pqDistanceShape;
  tensorPQDistance.dtype = "float16";

  TeOpTensorArg tensorIvfArg;
  tensorIvfArg.tensor.push_back(tensorIvf);
  tensorIvfArg.arg_type = TA_SINGLE;
  
  TeOpTensorArg tensorBucketListArg;
  tensorBucketListArg.tensor.push_back(tensorBucketList);
  tensorBucketListArg.arg_type = TA_SINGLE;

  TeOpTensorArg tensorPQDistanceArg;
  tensorPQDistanceArg.tensor.push_back(tensorPQDistance);
  tensorPQDistanceArg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensorIvfArg);
  opParas.inputs.push_back(tensorBucketListArg);
  opParas.outputs.push_back(tensorPQDistanceArg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "scanPQCodes.key0";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  std::cout << "Tiling_data is: " << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string(runInfo.tiling_data), "2 1 0 2 0 ");
}
/*
TEST_F(ScanPQCodesTiling, scan_pq_codes_tiling_2) {
  using namespace optiling;
  std::string op_name = "ScanPQCodes";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"core_nums\": 8, \"split_count\": 1, \"split_index\": 0}}";

  std::vector<int64_t> ivfShape{20480, 16};
  std::vector<int64_t> bucketListShape{8};
  std::vector<int64_t> pqDistanceShape{20480};

  TeOpTensor tensorIvf;
  tensorIvf.shape = ivfShape;
  tensorIvf.dtype = "uint8";

  TeOpTensor tensorBucketList;
  tensorBucketList.shape = bucketListShape;
  tensorBucketList.dtype = "int32";

  TeOpTensor tensorPQDistance;
  tensorPQDistance.shape = pqDistanceShape;
  tensorPQDistance.dtype = "float16";

  TeOpTensorArg tensorIvfArg;
  tensorIvfArg.tensor.push_back(tensorIvf);
  tensorIvfArg.arg_type = TA_SINGLE;
  
  TeOpTensorArg tensorBucketListArg;
  tensorBucketListArg.tensor.push_back(tensorBucketList);
  tensorBucketListArg.arg_type = TA_SINGLE;

  TeOpTensorArg tensorPQDistanceArg;
  tensorPQDistanceArg.tensor.push_back(tensorPQDistance);
  tensorPQDistanceArg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensorIvfArg);
  opParas.inputs.push_back(tensorBucketListArg);
  opParas.outputs.push_back(tensorPQDistanceArg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "scanPQCodes.key3";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}
*/
TEST_F(ScanPQCodesTiling, scan_pq_codes_tiling_4) {
  using namespace optiling;
  std::string op_name = "ScanPQCodes";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {}}";

  std::vector<int64_t> ivfShape{20480, 16};
  std::vector<int64_t> bucketListShape{8};
  std::vector<int64_t> pqDistanceShape{20480};

  TeOpTensor tensorIvf;
  tensorIvf.shape = ivfShape;
  tensorIvf.dtype = "uint8";

  TeOpTensor tensorBucketList;
  tensorBucketList.shape = bucketListShape;
  tensorBucketList.dtype = "int32";

  TeOpTensor tensorPQDistance;
  tensorPQDistance.shape = pqDistanceShape;
  tensorPQDistance.dtype = "float16";

  TeOpTensorArg tensorIvfArg;
  tensorIvfArg.tensor.push_back(tensorIvf);
  tensorIvfArg.arg_type = TA_SINGLE;
  
  TeOpTensorArg tensorBucketListArg;
  tensorBucketListArg.tensor.push_back(tensorBucketList);
  tensorBucketListArg.arg_type = TA_SINGLE;

  TeOpTensorArg tensorPQDistanceArg;
  tensorPQDistanceArg.tensor.push_back(tensorPQDistance);
  tensorPQDistanceArg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensorIvfArg);
  opParas.inputs.push_back(tensorBucketListArg);
  opParas.outputs.push_back(tensorPQDistanceArg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "scanPQCodes.key2";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}
/*
TEST_F(ScanPQCodesTiling, scan_pq_codes_tiling_4) {
  using namespace optiling;
  std::string op_name = "ScanPQCodes";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"core_nums\": 8, \"split_count\": 1, \"split_index\": 0}}";

  std::vector<int64_t> ivfShape{20480, 16};
  std::vector<int64_t> bucketListShape{8};
  std::vector<int64_t> pqDistanceShape{20480};

  TeOpTensor tensorIvf;
  tensorIvf.shape = ivfShape;
  tensorIvf.dtype = "uint8";

  TeOpTensor tensorBucketList;
  tensorBucketList.shape = bucketListShape;
  tensorBucketList.dtype = "int32";

  TeOpTensor tensorPQDistance;
  tensorPQDistance.shape = pqDistanceShape;
  tensorPQDistance.dtype = "float16";

  TeOpTensorArg tensorIvfArg;
  tensorIvfArg.tensor.push_back(tensorIvf);
  tensorIvfArg.arg_type = TA_SINGLE;
  
  TeOpTensorArg tensorBucketListArg;
  tensorBucketListArg.tensor.push_back(tensorBucketList);
  tensorBucketListArg.arg_type = TA_SINGLE;

  TeOpTensorArg tensorPQDistanceArg;
  tensorPQDistanceArg.tensor.push_back(tensorPQDistance);
  tensorPQDistanceArg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensorIvfArg);
  // opParas.inputs.push_back(tensorBucketListArg);
  opParas.outputs.push_back(tensorPQDistanceArg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "scanPQCodes.key4";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(ScanPQCodesTiling, scan_pq_codes_tiling_5) {
  using namespace optiling;
  std::string op_name = "ScanPQCodes";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"core_nums\": 8, \"split_count\": 1, \"split_index\": 0}}";

  std::vector<int64_t> ivfShape{20480, 16};
  std::vector<int64_t> bucketListShape{4};
  std::vector<int64_t> pqDistanceShape{20480};

  TeOpTensor tensorIvf;
  tensorIvf.shape = ivfShape;
  tensorIvf.dtype = "uint8";

  TeOpTensor tensorBucketList;
  tensorBucketList.shape = bucketListShape;
  tensorBucketList.dtype = "int32";

  TeOpTensor tensorPQDistance;
  tensorPQDistance.shape = pqDistanceShape;
  tensorPQDistance.dtype = "float16";

  TeOpTensorArg tensorIvfArg;
  tensorIvfArg.tensor.push_back(tensorIvf);
  tensorIvfArg.arg_type = TA_SINGLE;
  
  TeOpTensorArg tensorBucketListArg;
  tensorBucketListArg.tensor.push_back(tensorBucketList);
  tensorBucketListArg.arg_type = TA_SINGLE;

  TeOpTensorArg tensorPQDistanceArg;
  tensorPQDistanceArg.tensor.push_back(tensorPQDistance);
  tensorPQDistanceArg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  // opParas.inputs.push_back(tensorIvfArg);
  // opParas.inputs.push_back(tensorBucketListArg);
  opParas.outputs.push_back(tensorPQDistanceArg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "scanPQCodes.key5";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}

TEST_F(ScanPQCodesTiling, scan_pq_codes_tiling_6) {
  using namespace optiling;
  std::string op_name = "ScanPQCodes";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"core_nums\": 8, \"split_count\": 1, \"split_index\": 0}}";

  std::vector<int64_t> ivfShape{20480, 16};
  std::vector<int64_t> bucketListShape{4};
  std::vector<int64_t> pqDistanceShape{20480};

  TeOpTensor tensorIvf;
  tensorIvf.shape = ivfShape;
  tensorIvf.dtype = "uint8";

  TeOpTensor tensorBucketList;
  tensorBucketList.shape = bucketListShape;
  tensorBucketList.dtype = "int32";

  TeOpTensor tensorPQDistance;
  tensorPQDistance.shape = pqDistanceShape;
  tensorPQDistance.dtype = "float16";

  TeOpTensorArg tensorIvfArg;
  tensorIvfArg.tensor.push_back(tensorIvf);
  tensorIvfArg.arg_type = TA_SINGLE;
  
  TeOpTensorArg tensorBucketListArg;
  // tensorBucketListArg.tensor.push_back(tensorBucketList);
  tensorBucketListArg.arg_type = TA_SINGLE;

  TeOpTensorArg tensorPQDistanceArg;
  tensorPQDistanceArg.tensor.push_back(tensorPQDistance);
  tensorPQDistanceArg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensorIvfArg);
  opParas.inputs.push_back(tensorBucketListArg);
  opParas.outputs.push_back(tensorPQDistanceArg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "scanPQCodes.key6";
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second(opParas, op_compile_info, runInfo));
}*/
