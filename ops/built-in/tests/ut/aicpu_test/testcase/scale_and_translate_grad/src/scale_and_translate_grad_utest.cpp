#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_read_file.h"
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"
#include <random>

using namespace std;
using namespace aicpu;

class TEST_SCALEANDTRANSLATEGRAD_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias)      \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();                 \
  NodeDefBuilder(node_def.get(), "ScaleAndTranslateGrad", "ScaleAndTranslateGrad") \
      .Input({"grads", (data_types)[0], (shapes)[0], (datas)[0]})                  \
      .Input({"original_image", (data_types)[1], (shapes)[1], (datas)[1]})         \
      .Input({"scale", (data_types)[2], (shapes)[2], (datas)[2]})                  \
      .Input({"translation", (data_types)[3], (shapes)[3], (datas)[3]})            \
      .Output({"y", (data_types)[4], (shapes)[4], (datas)[4]})                     \
      .Attr("kernel_type", (kernel_type_str))                                      \
      .Attr("antialias", (antialias));

// read input and output data from files which generate by your python file

void RunScaleAndTranslateKernel(vector<string> data_files, vector<DataType> data_types,
                         vector<vector<int64_t>> &shapes, std::string kernel_type_str, bool antialias) {
  // read data from file for input1
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t grads_size = CalTotalElements(shapes, 0);
  float *grads = new float[grads_size];
  bool status = ReadFile(data_path, grads, grads_size);
  EXPECT_EQ(status, true);

  // read data from file for size
  data_path = ktestcaseFilePath + data_files[1];
  uint64_t original_image_size = CalTotalElements(shapes, 1);
  float *original_image_data = new float[original_image_size];
  status = ReadFile(data_path, original_image_data, original_image_size);
  EXPECT_EQ(status, true);

  // read data from file for scale
  data_path = ktestcaseFilePath + data_files[2];
  uint64_t scale_size = CalTotalElements(shapes, 2);
  float *scale_data = new float[scale_size];
  status = ReadFile(data_path, scale_data, scale_size);
  EXPECT_EQ(status, true);

  // read data from file for translate
  data_path = ktestcaseFilePath + data_files[3];
  uint64_t translate_size = CalTotalElements(shapes, 3);
  float *translate_data = new float[translate_size];
  status = ReadFile(data_path, translate_data, translate_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 4);
  float *output = new float[output_size];
  // int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};

  vector<void *> datas = {(void *)grads, (void *)original_image_data, (void *)scale_data, (void *)translate_data, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[4];
  float *output_exp = new float[output_size];

  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  std::cout<<"output:"<<*output<<std::endl;
  std::cout<<"output_exp:"<<*output_exp<<std::endl;
  EXPECT_EQ(compare, true);
  delete[] grads;
  delete[] original_image_data;
  delete[] scale_data;
  delete[] translate_data;
  delete[] output;
  delete[] output_exp;
}

TEST_F(TEST_SCALEANDTRANSLATEGRAD_UT, DATA_TYPE_LANCZOS1_TRUE_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={2, 4, 6, 2};

  std::string num = "lanczos1";
  vector<vector<int64_t>> shapes = {in_shape, in_shape, {2}, {2}, in_shape};
  vector<string> files{"scale_and_translate_grad/data/scale_and_translate_grad_data_grad_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_original_image_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_scale_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_translate_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_output_" +num +"_true.txt"};
  
  std::string kernel_type_str = "lanczos1";
  bool antialias = true;                  
  RunScaleAndTranslateKernel(files, data_types, shapes, kernel_type_str, antialias);
}
TEST_F(TEST_SCALEANDTRANSLATEGRAD_UT, DATA_TYPE_LANCZOS3_TRUE_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={2, 4, 6, 2};

  std::string num = "lanczos3";
  vector<vector<int64_t>> shapes = {in_shape, in_shape, {2}, {2}, in_shape};
  vector<string> files{"scale_and_translate_grad/data/scale_and_translate_grad_data_grad_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_original_image_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_scale_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_translate_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_output_" +num +"_true.txt"};
  
  std::string kernel_type_str = "lanczos3";
  bool antialias = true;                  
  RunScaleAndTranslateKernel(files, data_types, shapes, kernel_type_str, antialias);
}
TEST_F(TEST_SCALEANDTRANSLATEGRAD_UT, DATA_TYPE_LANCZOS5_TRUE_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={2, 4, 6, 2};

  std::string num = "lanczos5";
  vector<vector<int64_t>> shapes = {in_shape, in_shape, {2}, {2}, in_shape};
  vector<string> files{"scale_and_translate_grad/data/scale_and_translate_grad_data_grad_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_original_image_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_scale_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_translate_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_output_" +num +"_true.txt"};
  
  std::string kernel_type_str = "lanczos5";
  bool antialias = true;                  
  RunScaleAndTranslateKernel(files, data_types, shapes, kernel_type_str, antialias);
}
TEST_F(TEST_SCALEANDTRANSLATEGRAD_UT, DATA_TYPE_BOX_TRUE_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={2, 4, 6, 2};

  std::string num = "box";
  vector<vector<int64_t>> shapes = {in_shape, in_shape, {2}, {2}, in_shape};
  vector<string> files{"scale_and_translate_grad/data/scale_and_translate_grad_data_grad_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_original_image_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_scale_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_translate_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_output_" +num +"_true.txt"};
  
  std::string kernel_type_str = "box";
  bool antialias = true;                  
  RunScaleAndTranslateKernel(files, data_types, shapes, kernel_type_str, antialias);
}
TEST_F(TEST_SCALEANDTRANSLATEGRAD_UT, DATA_TYPE_GAUSSIAN_TRUE_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={2, 4, 6, 2};

  std::string num = "gaussian";
  vector<vector<int64_t>> shapes = {in_shape, in_shape, {2}, {2}, in_shape};
  vector<string> files{"scale_and_translate_grad/data/scale_and_translate_grad_data_grad_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_original_image_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_scale_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_translate_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_output_" +num +"_true.txt"};
  
  std::string kernel_type_str = "gaussian";
  bool antialias = true;                  
  RunScaleAndTranslateKernel(files, data_types, shapes, kernel_type_str, antialias);
}
TEST_F(TEST_SCALEANDTRANSLATEGRAD_UT, DATA_TYPE_MITCHELLCUBIC_TRUE_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={2, 4, 6, 2};

  std::string num = "mitchellcubic";
  vector<vector<int64_t>> shapes = {in_shape, in_shape, {2}, {2}, in_shape};
  vector<string> files{"scale_and_translate_grad/data/scale_and_translate_grad_data_grad_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_original_image_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_scale_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_translate_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_output_" +num +"_true.txt"};
  
  std::string kernel_type_str = "mitchellcubic";
  bool antialias = true;                  
  RunScaleAndTranslateKernel(files, data_types, shapes, kernel_type_str, antialias);
}
TEST_F(TEST_SCALEANDTRANSLATEGRAD_UT, DATA_TYPE_KEYSCUBIC_TRUE_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={2, 4, 6, 2};

  std::string num = "keyscubic";
  vector<vector<int64_t>> shapes = {in_shape, in_shape, {2}, {2}, in_shape};
  vector<string> files{"scale_and_translate_grad/data/scale_and_translate_grad_data_grad_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_original_image_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_scale_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_translate_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_output_" +num +"_true.txt"};
  
  std::string kernel_type_str = "keyscubic";
  bool antialias = true;                  
  RunScaleAndTranslateKernel(files, data_types, shapes, kernel_type_str, antialias);
}
TEST_F(TEST_SCALEANDTRANSLATEGRAD_UT, DATA_TYPE_TRIANGLE_TRUE_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={2, 4, 6, 2};

  std::string num = "triangle";
  vector<vector<int64_t>> shapes = {in_shape, in_shape, {2}, {2}, in_shape};
  vector<string> files{"scale_and_translate_grad/data/scale_and_translate_grad_data_grad_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_original_image_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_scale_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_translate_" +num +"_true.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_output_" +num +"_true.txt"};
  
  std::string kernel_type_str = "triangle";
  bool antialias = true;                  
  RunScaleAndTranslateKernel(files, data_types, shapes, kernel_type_str, antialias);
}

TEST_F(TEST_SCALEANDTRANSLATEGRAD_UT, DATA_TYPE_LANCZOS1_FALSE_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={2, 4, 6, 2};

  std::string num = "lanczos1";
  vector<vector<int64_t>> shapes = {in_shape, in_shape, {2}, {2}, in_shape};
  vector<string> files{"scale_and_translate_grad/data/scale_and_translate_grad_data_grad_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_original_image_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_scale_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_translate_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_output_" +num +"_false.txt"};
  
  std::string kernel_type_str = "lanczos1";
  bool antialias = false;                  
  RunScaleAndTranslateKernel(files, data_types, shapes, kernel_type_str, antialias);
}
TEST_F(TEST_SCALEANDTRANSLATEGRAD_UT, DATA_TYPE_LANCZOS3_FALSE_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={2, 4, 6, 2};

  std::string num = "lanczos3";
  vector<vector<int64_t>> shapes = {in_shape, in_shape, {2}, {2}, in_shape};
  vector<string> files{"scale_and_translate_grad/data/scale_and_translate_grad_data_grad_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_original_image_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_scale_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_translate_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_output_" +num +"_false.txt"};
  
  std::string kernel_type_str = "lanczos3";
  bool antialias = false;                  
  RunScaleAndTranslateKernel(files, data_types, shapes, kernel_type_str, antialias);
}
TEST_F(TEST_SCALEANDTRANSLATEGRAD_UT, DATA_TYPE_LANCZOS5_FALSE_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={2, 4, 6, 2};

  std::string num = "lanczos5";
  vector<vector<int64_t>> shapes = {in_shape, in_shape, {2}, {2}, in_shape};
  vector<string> files{"scale_and_translate_grad/data/scale_and_translate_grad_data_grad_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_original_image_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_scale_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_translate_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_output_" +num +"_false.txt"};
  
  std::string kernel_type_str = "lanczos5";
  bool antialias = false;                  
  RunScaleAndTranslateKernel(files, data_types, shapes, kernel_type_str, antialias);
}
TEST_F(TEST_SCALEANDTRANSLATEGRAD_UT, DATA_TYPE_BOX_FALSE_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={2, 4, 6, 2};

  std::string num = "box";
  vector<vector<int64_t>> shapes = {in_shape, in_shape, {2}, {2}, in_shape};
  vector<string> files{"scale_and_translate_grad/data/scale_and_translate_grad_data_grad_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_original_image_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_scale_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_translate_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_output_" +num +"_false.txt"};
  
  std::string kernel_type_str = "box";
  bool antialias = false;                  
  RunScaleAndTranslateKernel(files, data_types, shapes, kernel_type_str, antialias);
}
TEST_F(TEST_SCALEANDTRANSLATEGRAD_UT, DATA_TYPE_GAUSSIAN_FALSE_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={2, 4, 6, 2};

  std::string num = "gaussian";
  vector<vector<int64_t>> shapes = {in_shape, in_shape, {2}, {2}, in_shape};
  vector<string> files{"scale_and_translate_grad/data/scale_and_translate_grad_data_grad_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_original_image_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_scale_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_translate_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_output_" +num +"_false.txt"};
  
  std::string kernel_type_str = "gaussian";
  bool antialias = false;                  
  RunScaleAndTranslateKernel(files, data_types, shapes, kernel_type_str, antialias);
}
TEST_F(TEST_SCALEANDTRANSLATEGRAD_UT, DATA_TYPE_MITCHELLCUBIC_FALSE_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={2, 4, 6, 2};

  std::string num = "mitchellcubic";
  vector<vector<int64_t>> shapes = {in_shape, in_shape, {2}, {2}, in_shape};
  vector<string> files{"scale_and_translate_grad/data/scale_and_translate_grad_data_grad_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_original_image_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_scale_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_translate_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_output_" +num +"_false.txt"};
  
  std::string kernel_type_str = "mitchellcubic";
  bool antialias = false;                  
  RunScaleAndTranslateKernel(files, data_types, shapes, kernel_type_str, antialias);
}
TEST_F(TEST_SCALEANDTRANSLATEGRAD_UT, DATA_TYPE_KEYSCUBIC_FALSE_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={2, 4, 6, 2};

  std::string num = "keyscubic";
  vector<vector<int64_t>> shapes = {in_shape, in_shape, {2}, {2}, in_shape};
  vector<string> files{"scale_and_translate_grad/data/scale_and_translate_grad_data_grad_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_original_image_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_scale_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_translate_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_output_" +num +"_false.txt"};
  
  std::string kernel_type_str = "keyscubic";
  bool antialias = false;                  
  RunScaleAndTranslateKernel(files, data_types, shapes, kernel_type_str, antialias);
}
TEST_F(TEST_SCALEANDTRANSLATEGRAD_UT, DATA_TYPE_TRIANGLE_FALSE_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={2, 4, 6, 2};

  std::string num = "triangle";
  vector<vector<int64_t>> shapes = {in_shape, in_shape, {2}, {2}, in_shape};
  vector<string> files{"scale_and_translate_grad/data/scale_and_translate_grad_data_grad_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_original_image_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_scale_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_translate_" +num +"_false.txt",
                       "scale_and_translate_grad/data/scale_and_translate_data_grad_output_" +num +"_false.txt"};
  
  std::string kernel_type_str = "triangle";
  bool antialias = false;                  
  RunScaleAndTranslateKernel(files, data_types, shapes, kernel_type_str, antialias);
}

// exception instance
TEST_F(TEST_SCALEANDTRANSLATEGRAD_UT, INPUT_DATA_TYPE_EXCEPTION) {
  std::vector<uint64_t> data;
  vector<DataType> data_types = {DT_UINT64, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};

  u_int64_t grads[4] = {(u_int64_t)1};
  vector<int64_t> datashape = {1, 2 , 2 ,1};
  float original_image[4] = {(float)1};

  float scale_data[2] = {0.5, 0.5}; 
  float translate_data[2] = {0.5, 0.5}; 

  float output[4] = {(float)0};

  vector<vector<int64_t>> shapes = {datashape, datashape, {2}, {2}, datashape}; 

  vector<void *> datas = {(void *)grads, (void *)original_image, (void *)scale_data, (void *)translate_data, (void *)output};
  std::string kernel_type_str = "box";
  bool antialias = false;
  CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}