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

class TEST_ADD_BLOCK_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Add", "Add")                     \
      .Input({"x1", data_types[0], shapes[0], datas[0]})           \
      .Input({"x2", data_types[1], shapes[1], datas[1]})           \
      .Output({"y", data_types[2], shapes[2], datas[2]});

TEST_F(TEST_ADD_BLOCK_UT, BROADCAST_INPUT_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 12}, {12}, {3, 12}};

  // read data from file for input1
  string data_path = ktestcaseFilePath + "add/data/add_data_input1_1.txt";
  int32_t input1[36] = {0};
  bool status = ReadFile(data_path, input1, 36);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = ktestcaseFilePath + "add/data/add_data_input2_1.txt";
  int32_t input2[12] = {0};
  status = ReadFile(data_path, input2, 12);
  EXPECT_EQ(status, true);

  int32_t output[36] = {0};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  auto blockInfo_ptr = new (std::nothrow) BlkDimInfo();
  blockInfo_ptr->blockNum = 2;
  blockInfo_ptr->blockId = 0;
  RUN_KERNEL_WITHBLOCK(node_def, HOST, KERNEL_STATUS_OK, blockInfo_ptr);
  delete blockInfo_ptr;
  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "add/data/add_data_output1_1.txt";
  int32_t output_exp[36] = {0};
  status = ReadFile(data_path, output_exp, 36);
  EXPECT_EQ(status, true);
  bool compare = CompareResult(output, output_exp, 36);
  EXPECT_EQ(compare, true);
}
