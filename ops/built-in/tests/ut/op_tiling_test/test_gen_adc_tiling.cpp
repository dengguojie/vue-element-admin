#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class GenADCTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GenADCTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GenADCTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream& tiling_data) {
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

static void RunTestTiling(const std::vector<int64_t>& queryShape, const std::string& queryDtype,
                          const std::vector<int64_t>& codeBookShape, const std::string& codeBookDtype,
                          const std::vector<int64_t>& centroidsShape, const std::string& centroidsDtype,
                          const std::vector<int64_t>& bucketListShape, const std::string& bucketListDtype,
                          const std::vector<int64_t>& adcTablesShape, const std::string& adcTablesDtype,
                          const std::string& compileInfo, const std::string& compileInfoKey,
                          const std::string& expectTiling) {
  using namespace optiling;
  std::string opName = "GenADC";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find("GenADC");
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  TeOpTensor queryTensor;
  queryTensor.shape = queryShape;
  queryTensor.dtype = queryDtype;

  TeOpTensor codeBookTensor;
  codeBookTensor.shape = codeBookShape;
  codeBookTensor.dtype = codeBookDtype;

  TeOpTensor centroidsTensor;
  centroidsTensor.shape = centroidsShape;
  centroidsTensor.dtype = centroidsDtype;

  TeOpTensor bucketListTensor;
  bucketListTensor.shape = bucketListShape;
  bucketListTensor.dtype = bucketListDtype;

  TeOpTensor adcTablesTensor;
  adcTablesTensor.shape = adcTablesShape;
  adcTablesTensor.dtype = adcTablesDtype;

  TeOpTensorArg tensorArgQuery;
  tensorArgQuery.tensor.push_back(queryTensor);
  tensorArgQuery.arg_type = TA_SINGLE;

  TeOpTensorArg tensorArgCodeBook;
  tensorArgCodeBook.tensor.push_back(codeBookTensor);
  tensorArgCodeBook.arg_type = TA_SINGLE;

  TeOpTensorArg tensorArgCentroids;
  tensorArgCentroids.tensor.push_back(centroidsTensor);
  tensorArgCentroids.arg_type = TA_SINGLE;

  TeOpTensorArg tensorArgBucketList;
  tensorArgBucketList.tensor.push_back(bucketListTensor);
  tensorArgBucketList.arg_type = TA_SINGLE;

  TeOpTensorArg tensorArgAdcTables;
  tensorArgAdcTables.tensor.push_back(adcTablesTensor);
  tensorArgAdcTables.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.inputs.push_back(tensorArgQuery);
  opParas.inputs.push_back(tensorArgCodeBook);
  opParas.inputs.push_back(tensorArgCentroids);
  opParas.inputs.push_back(tensorArgBucketList);
  opParas.outputs.push_back(tensorArgAdcTables);

  opParas.op_type = opName;
  OpCompileInfo opCompileInfo;
  opCompileInfo.str = compileInfo;
  opCompileInfo.key = compileInfoKey;
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, opCompileInfo, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), expectTiling);
}

TEST_F(GenADCTiling, gen_adc_tiling_001) {
  std::vector<int64_t> queryShape{32};
  std::string queryDtype = "float16";

  std::vector<int64_t> codeBookShape{16, 256, 2};
  std::string codeBookDtype = "float16";

  std::vector<int64_t> centroidsShape{1000000, 32};
  std::string centroidsDtype = "float16";

  std::vector<int64_t> bucketListShape{10};
  std::string bucketListDtype = "int32";

  std::vector<int64_t> adcTablesShape{10, 16, 256};
  std::string adcTablesDtype = "float16";

  std::string compileInfo = "{\"vars\": {\"core_num\": 8}}";
  std::string compileInfoKey = "gen_adc.key.001";

  std::string expectTiling = "2 2 1 1 5 10 ";

  RunTestTiling(queryShape, queryDtype, codeBookShape, codeBookDtype, centroidsShape, centroidsDtype, bucketListShape,
                bucketListDtype, adcTablesShape, adcTablesDtype, compileInfo, compileInfoKey, expectTiling);
}

TEST_F(GenADCTiling, gen_adc_tiling_002) {
  std::vector<int64_t> queryShape{64};
  std::string queryDtype = "float32";

  std::vector<int64_t> codeBookShape{32, 256, 2};
  std::string codeBookDtype = "float32";

  std::vector<int64_t> centroidsShape{1000000, 32};
  std::string centroidsDtype = "float32";

  std::vector<int64_t> bucketListShape{77};
  std::string bucketListDtype = "int64";

  std::vector<int64_t> adcTablesShape{77, 32, 256};
  std::string adcTablesDtype = "float16";

  std::string compileInfo = "{\"vars\": {\"core_num\": 8}}";
  std::string compileInfoKey = "gen_adc.key.002";

  std::string expectTiling = "10 7 3 2 8 77 ";

  RunTestTiling(queryShape, queryDtype, codeBookShape, codeBookDtype, centroidsShape, centroidsDtype, bucketListShape,
                bucketListDtype, adcTablesShape, adcTablesDtype, compileInfo, compileInfoKey, expectTiling);
}

TEST_F(GenADCTiling, gen_adc_tiling_003) {
  std::vector<int64_t> queryShape{256};
  std::string queryDtype = "float32";

  std::vector<int64_t> codeBookShape{64, 512, 4};
  std::string codeBookDtype = "float32";

  std::vector<int64_t> centroidsShape{1000000, 512};
  std::string centroidsDtype = "float32";

  std::vector<int64_t> bucketListShape{64};
  std::string bucketListDtype = "int32";

  std::vector<int64_t> adcTablesShape{64, 64, 512};
  std::string adcTablesDtype = "float16";

  std::string compileInfo = "{\"vars\": {\"core_num\": 8}}";
  std::string compileInfoKey = "gen_adc.key.003";

  std::string expectTiling = "8 8 1 1 8 64 ";

  RunTestTiling(queryShape, queryDtype, codeBookShape, codeBookDtype, centroidsShape, centroidsDtype, bucketListShape,
                bucketListDtype, adcTablesShape, adcTablesDtype, compileInfo, compileInfoKey, expectTiling);
}

TEST_F(GenADCTiling, gen_adc_tiling_004) {
  std::vector<int64_t> queryShape{1024};
  std::string queryDtype = "float16";

  std::vector<int64_t> codeBookShape{256, 256, 4};
  std::string codeBookDtype = "float16";

  std::vector<int64_t> centroidsShape{1000000, 1024};
  std::string centroidsDtype = "float16";

  std::vector<int64_t> bucketListShape{128};
  std::string bucketListDtype = "int64";

  std::vector<int64_t> adcTablesShape{128, 256, 256};
  std::string adcTablesDtype = "float16";

  std::string compileInfo = "{\"vars\": {\"core_num\": 32}}";
  std::string compileInfoKey = "gen_adc.key.004";

  std::string expectTiling = "4 4 1 1 32 128 ";

  RunTestTiling(queryShape, queryDtype, codeBookShape, codeBookDtype, centroidsShape, centroidsDtype, bucketListShape,
                bucketListDtype, adcTablesShape, adcTablesDtype, compileInfo, compileInfoKey, expectTiling);
}

TEST_F(GenADCTiling, gen_adc_tiling_005) {
  std::vector<int64_t> queryShape{32};
  std::string queryDtype = "float16";

  std::vector<int64_t> codeBookShape{16, 256, 2};
  std::string codeBookDtype = "float16";

  std::vector<int64_t> centroidsShape{1000000, 32};
  std::string centroidsDtype = "float16";

  std::vector<int64_t> bucketListShape{1};
  std::string bucketListDtype = "int64";

  std::vector<int64_t> adcTablesShape{1, 16, 256};
  std::string adcTablesDtype = "float16";

  std::string compileInfo = "{\"vars\": {\"core_num\": 8}}";
  std::string compileInfoKey = "gen_adc.key.005";

  std::string expectTiling = "2 1 1 1 1 1 ";

  RunTestTiling(queryShape, queryDtype, codeBookShape, codeBookDtype, centroidsShape, centroidsDtype, bucketListShape,
                bucketListDtype, adcTablesShape, adcTablesDtype, compileInfo, compileInfoKey, expectTiling);
}

TEST_F(GenADCTiling, gen_adc_tiling_006) {
  std::vector<int64_t> queryShape{32};
  std::string queryDtype = "float16";

  std::vector<int64_t> codeBookShape{16, 256, 2};
  std::string codeBookDtype = "float16";

  std::vector<int64_t> centroidsShape{1000000, 32};
  std::string centroidsDtype = "float16";

  std::vector<int64_t> bucketListShape{2};
  std::string bucketListDtype = "int64";

  std::vector<int64_t> adcTablesShape{2, 16, 256};
  std::string adcTablesDtype = "float16";

  std::string compileInfo = "{\"vars\": {\"core_num\": 8}}";
  std::string compileInfoKey = "gen_adc.key.006";

  std::string expectTiling = "2 2 1 1 1 2 ";

  RunTestTiling(queryShape, queryDtype, codeBookShape, codeBookDtype, centroidsShape, centroidsDtype, bucketListShape,
                bucketListDtype, adcTablesShape, adcTablesDtype, compileInfo, compileInfoKey, expectTiling);
}

TEST_F(GenADCTiling, gen_adc_tiling_007) {
  std::vector<int64_t> queryShape{32};
  std::string queryDtype = "float16";

  std::vector<int64_t> codeBookShape{16, 256, 2};
  std::string codeBookDtype = "float16";

  std::vector<int64_t> centroidsShape{1000000, 32};
  std::string centroidsDtype = "float16";

  std::vector<int64_t> bucketListShape{3};
  std::string bucketListDtype = "int64";

  std::vector<int64_t> adcTablesShape{3, 16, 256};
  std::string adcTablesDtype = "float16";

  std::string compileInfo = "{\"vars\": {\"core_num\": 8}}";
  std::string compileInfoKey = "gen_adc.key.007";

  std::string expectTiling = "2 1 1 1 2 3 ";

  RunTestTiling(queryShape, queryDtype, codeBookShape, codeBookDtype, centroidsShape, centroidsDtype, bucketListShape,
                bucketListDtype, adcTablesShape, adcTablesDtype, compileInfo, compileInfoKey, expectTiling);
}
