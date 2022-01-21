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

class TEST_SCALEANDTRANSLATE_UT : public testing::Test {};

template<typename T>
void SetCheckerboardImageInput(int64_t batch_size, int64_t num_row_squares,
                                 int64_t num_col_squares, int64_t square_size,
                                 int64_t num_channels, std::vector<T>& data) {
    // inputs_.clear();
    const int64_t row_size = num_col_squares * square_size * num_channels;
    const int64_t image_size = num_row_squares * square_size * row_size;
    data.resize(batch_size * image_size);
    // random::PhiloxRandom philox(42);
    // random::SimplePhilox rnd(&philox);
    typedef std::mt19937 RNG_Engine;
    RNG_Engine rng;
    rng.seed(0);
    std::uniform_real_distribution<float> Unifrom_01(0, 1);
    std::vector<float> col(num_channels);
    for (int b = 0; b < batch_size; ++b) {
      for (int y = 0; y < num_row_squares; ++y) {
        for (int x = 0; x < num_col_squares; ++x) {
          for (int n = 0; n < num_channels; ++n) {
            // col[n] = rnd.RandFloat();
            col[n] = Unifrom_01(rng);
          }
          for (int r = y * square_size; r < (y + 1) * square_size; ++r) {
            auto it = data.begin() + b * image_size + r * row_size +
                      x * square_size * num_channels;
            for (int n = 0; n < square_size; ++n) {
              for (int chan = 0; chan < num_channels; ++chan, ++it) {
                *it = static_cast<T>(col[chan] * 255.0f);
              }
            }
          }
        }
      }
    }

  }

#define CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias) \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();            \
  NodeDefBuilder(node_def.get(), "ScaleAndTranslate", "ScaleAndTranslate")    \
      .Input({"images", (data_types)[0], (shapes)[0], (datas)[0]})            \
      .Input({"size", (data_types)[1], (shapes)[1], (datas)[1]})              \
      .Input({"scale", (data_types)[2], (shapes)[2], (datas)[2]})             \
      .Input({"translation", (data_types)[3], (shapes)[3], (datas)[3]})       \
      .Output({"y", (data_types)[4], (shapes)[4], (datas)[4]})                \
      .Attr("kernel_type", (kernel_type_str))                                 \
      .Attr("antialias", (antialias));

  // read input and output data from files which generate by your python file
  template <typename T1>
  void RunScaleAndTranslateKernel(vector<string> data_files, vector<DataType> data_types,
                                  vector<vector<int64_t>>& shapes, std::string kernel_type_str, bool antialias) {
    // read data from file for input1
    string data_path = ktestcaseFilePath + data_files[0];
    uint64_t image_size = CalTotalElements(shapes, 0);
    T1* image = new T1[image_size];
    bool status = ReadFile(data_path, image, image_size);
    EXPECT_EQ(status, true);

    // read data from file for size
    data_path = ktestcaseFilePath + data_files[1];
    uint64_t size = CalTotalElements(shapes, 1);
    int32_t* size_data = new int32_t[size];
    status = ReadFile(data_path, size_data, size);
    EXPECT_EQ(status, true);

    // read data from file for scale
    data_path = ktestcaseFilePath + data_files[2];
    uint64_t scale_size = CalTotalElements(shapes, 2);
    float* scale_data = new float[scale_size];
    status = ReadFile(data_path, scale_data, scale_size);
    EXPECT_EQ(status, true);

    // read data from file for translate
    data_path = ktestcaseFilePath + data_files[3];
    uint64_t translate_size = CalTotalElements(shapes, 3);
    float* translate_data = new float[translate_size];
    status = ReadFile(data_path, translate_data, translate_size);
    EXPECT_EQ(status, true);

    uint64_t output_size = CalTotalElements(shapes, 4);
    float* output = new float[output_size];
    // int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};

    vector<void*> datas = {(void*)image, (void*)size_data, (void*)scale_data, (void*)translate_data, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    // read data from file for expect ouput
    data_path = ktestcaseFilePath + data_files[4];
    float* output_exp = new float[output_size];

    status = ReadFile(data_path, output_exp, output_size);
    EXPECT_EQ(status, true);

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
    delete[] image;
    delete[] size_data;
    delete[] scale_data;
    delete[] translate_data;
    delete[] output;
    delete[] output_exp;
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_BOX_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={1, 2, 3, 1};
  vector<int64_t> out_shape ={1, 4, 6, 1};
  std::string num = "float_box";
  vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};
  vector<string> files{"scale_and_translate/data/scale_and_translate_data_image_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_size_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_scale_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_translate_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_output_" +num +".txt"};
  
  std::string kernel_type_str = "box";
  bool antialias = true;                  
  RunScaleAndTranslateKernel<float>(files, data_types, shapes, kernel_type_str, antialias);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_KEYSCUBIC_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={1, 2, 3, 1};
  vector<int64_t> out_shape ={1, 4, 6, 1};
  std::string num = "float_keyscubic";
  vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};
  vector<string> files{"scale_and_translate/data/scale_and_translate_data_image_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_size_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_scale_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_translate_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_output_" +num +".txt"};
  
  std::string kernel_type_str = "keyscubic";
  bool antialias = true;                  
  RunScaleAndTranslateKernel<float>(files, data_types, shapes, kernel_type_str, antialias);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_MITCHELLCUBIC_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={1, 2, 3, 1};
  vector<int64_t> out_shape ={1, 4, 6, 1};
  std::string num = "float_mitchellcubic";
  vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};
  vector<string> files{"scale_and_translate/data/scale_and_translate_data_image_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_size_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_scale_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_translate_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_output_" +num +".txt"};
  
  std::string kernel_type_str = "mitchellcubic";
  bool antialias = true;                  
  RunScaleAndTranslateKernel<float>(files, data_types, shapes, kernel_type_str, antialias);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_GAUSSIAN_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={1, 2, 3, 1};
  vector<int64_t> out_shape ={1, 4, 6, 1};
  std::string num = "float_gaussian";
  vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};
  vector<string> files{"scale_and_translate/data/scale_and_translate_data_image_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_size_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_scale_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_translate_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_output_" +num +".txt"};
  
  std::string kernel_type_str = "gaussian";
  bool antialias = true;                  
  RunScaleAndTranslateKernel<float>(files, data_types, shapes, kernel_type_str, antialias);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT32_BOX_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={1, 2, 3, 1};
  vector<int64_t> out_shape ={1, 4, 6, 1};
  std::string num = "int32_box";
  vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};
  vector<string> files{"scale_and_translate/data/scale_and_translate_data_image_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_size_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_scale_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_translate_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_output_" +num +".txt"};
  
  std::string kernel_type_str = "box";
  bool antialias = false;                  
  RunScaleAndTranslateKernel<int32_t>(files, data_types, shapes, kernel_type_str, antialias);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT32_KEYSCUBIC_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={1, 2, 3, 1};
  vector<int64_t> out_shape ={1, 4, 6, 1};
  std::string num = "int32_keyscubic";
  vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};
  vector<string> files{"scale_and_translate/data/scale_and_translate_data_image_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_size_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_scale_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_translate_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_output_" +num +".txt"};
  
  std::string kernel_type_str = "keyscubic";
  bool antialias = false;                  
  RunScaleAndTranslateKernel<int32_t>(files, data_types, shapes, kernel_type_str, antialias);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT32_MITCHELLCUBIC_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={1, 2, 3, 1};
  vector<int64_t> out_shape ={1, 4, 6, 1};
  std::string num = "int32_mitchellcubic";
  vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};
  vector<string> files{"scale_and_translate/data/scale_and_translate_data_image_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_size_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_scale_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_translate_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_output_" +num +".txt"};
  
  std::string kernel_type_str = "mitchellcubic";
  bool antialias = false;                  
  RunScaleAndTranslateKernel<int32_t>(files, data_types, shapes, kernel_type_str, antialias);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT32_GAUSSIAN_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={1, 2, 3, 1};
  vector<int64_t> out_shape ={1, 4, 6, 1};
  std::string num = "int32_gaussian";
  vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};
  vector<string> files{"scale_and_translate/data/scale_and_translate_data_image_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_size_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_scale_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_translate_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_output_" +num +".txt"};
  
  std::string kernel_type_str = "gaussian";
  bool antialias = false;                  
  RunScaleAndTranslateKernel<int32_t>(files, data_types, shapes, kernel_type_str, antialias);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT32_LANCZOS1_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape = {2, 4, 6, 2}; 
  vector<int64_t> out_shape ={2, 8, 12, 2};
  std::string num = "int32_lanczos1";
  vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};
  vector<string> files{"scale_and_translate/data/scale_and_translate_data_image_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_size_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_scale_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_translate_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_output_" +num +".txt"};
  
  std::string kernel_type_str = "lanczos1";
  bool antialias = true;                  
  RunScaleAndTranslateKernel<int32_t>(files, data_types, shapes, kernel_type_str, antialias);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT64_LANCZOS1_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape = {2, 4, 6, 2}; 
  vector<int64_t> out_shape ={2, 8, 12, 2};
  std::string num = "int64_lanczos1";
  vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};
  vector<string> files{"scale_and_translate/data/scale_and_translate_data_image_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_size_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_scale_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_translate_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_output_" +num +".txt"};
  
  std::string kernel_type_str = "lanczos1";
  bool antialias = false;                  
  RunScaleAndTranslateKernel<int64_t>(files, data_types, shapes, kernel_type_str, antialias);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT16_LANCZOS3_SUCC) {
  vector<DataType> data_types = {DT_INT16, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape = {2, 4, 6, 2}; 
  vector<int64_t> out_shape ={2, 8, 12, 2};
  std::string num = "int16_lanczos3";
  vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};
  vector<string> files{"scale_and_translate/data/scale_and_translate_data_image_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_size_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_scale_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_translate_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_output_" +num +".txt"};
  
  std::string kernel_type_str = "lanczos3";
  bool antialias = false;                  
  RunScaleAndTranslateKernel<int16_t>(files, data_types, shapes, kernel_type_str, antialias);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_UINT16_LANCZOS3_SUCC) {
  vector<DataType> data_types = {DT_UINT16, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape = {2, 4, 6, 2}; 
  vector<int64_t> out_shape ={2, 8, 12, 2};
  std::string num = "uint16_lanczos3";
  vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};
  vector<string> files{"scale_and_translate/data/scale_and_translate_data_image_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_size_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_scale_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_translate_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_output_" +num +".txt"};
  
  std::string kernel_type_str = "lanczos3";
  bool antialias = false;                  
  RunScaleAndTranslateKernel<uint16_t>(files, data_types, shapes, kernel_type_str, antialias);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_LANCZOS3_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape = {2, 4, 6, 2}; 
  vector<int64_t> out_shape ={2, 8, 12, 2};
  std::string num = "float_lanczos3";
  vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};
  vector<string> files{"scale_and_translate/data/scale_and_translate_data_image_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_size_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_scale_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_translate_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_output_" +num +".txt"};
  
  std::string kernel_type_str = "lanczos3";
  bool antialias = true;                  
  RunScaleAndTranslateKernel<float>(files, data_types, shapes, kernel_type_str, antialias);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT8_LANCZOS5_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape = {2, 4, 6, 2}; 
  vector<int64_t> out_shape ={2, 8, 12, 2};
  std::string num = "int8_lanczos5";
  vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};
  vector<string> files{"scale_and_translate/data/scale_and_translate_data_image_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_size_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_scale_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_translate_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_output_" +num +".txt"};
  
  std::string kernel_type_str = "lanczos5";
  bool antialias = true;                  
  RunScaleAndTranslateKernel<int8_t>(files, data_types, shapes, kernel_type_str, antialias);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_UINT8_LANCZOS5_SUCC) {
  vector<DataType> data_types = {DT_UINT8, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape = {2, 4, 6, 2}; 
  vector<int64_t> out_shape ={2, 8, 12, 2};
  std::string num = "uint8_lanczos5";
  vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};
  vector<string> files{"scale_and_translate/data/scale_and_translate_data_image_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_size_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_scale_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_translate_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_output_" +num +".txt"};
  
  std::string kernel_type_str = "lanczos5";
  bool antialias = true;                  
  RunScaleAndTranslateKernel<uint8_t>(files, data_types, shapes, kernel_type_str, antialias);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_DOUBLE_TRIANGLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={1, 2, 3, 1};
  vector<int64_t> out_shape ={1, 4, 6, 1};
  std::string num = "double_triangle";
  vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};
  vector<string> files{"scale_and_translate/data/scale_and_translate_data_image_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_size_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_scale_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_translate_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_output_" +num +".txt"};
  
  std::string kernel_type_str = "triangle";
  bool antialias = false;                  
  RunScaleAndTranslateKernel<double>(files, data_types, shapes, kernel_type_str, antialias);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT16_TRIANGLE_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  
  vector<int64_t> in_shape ={1, 2, 3, 1};
  vector<int64_t> out_shape ={1, 4, 6, 1};
  std::string num = "float16_triangle";
  vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};
  vector<string> files{"scale_and_translate/data/scale_and_translate_data_image_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_size_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_scale_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_translate_" +num +".txt",
                       "scale_and_translate/data/scale_and_translate_data_output_" +num +".txt"};
  
  std::string kernel_type_str = "triangle";
  bool antialias = false;                  
  RunScaleAndTranslateKernel<Eigen::half>(files, data_types, shapes, kernel_type_str, antialias);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_METHOD_BOX_SUCC) {
  std::vector<float> data;
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  int64_t kBatchSize_exp = 2;
  int64_t kNumRowSquares_exp = 16;
  int64_t kNumColSquares_exp = 13;
  int64_t kSquareSize_exp = 12;
  int64_t kNumChannels_exp = 3;
  
  SetCheckerboardImageInput<float> (kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp,
                            kSquareSize_exp, kNumChannels_exp, data);
  vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                     kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

  const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
  const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

  float scale_data[2] = {0.5, 0.5}; 

  float translate_data[2] = {0.5, 0.5}; 
  int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
  vector<int64_t> sizeshape ={2};
  const int outputshape= kBatchSize_exp* kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
  float* output = new float[outputshape];
  
  vector<vector<int64_t>> shapes = {datashape, sizeshape, {2}, {2}, {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}}; // 

  float* data_arr = new float[data.size()];

  std::copy(data.begin(), data.end(), data_arr);
  

  vector<void *> datas = {(void *)data_arr, (void *)size_data, (void *)scale_data, (void *)translate_data, (void *)output};
  std::string kernel_type_str = "box";
  bool antialias = false;
  CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  delete [] data_arr;
  delete [] output;
  // float output_exp[kOutputImageHeight_exp * kOutputImageWidth_exp] = {0};
  // CalcExpect<float>(*node_def.get(), output_exp);

  // bool compare = CompareResult(output, output_exp, 6);
  // EXPECT_EQ(compare, true);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_METHOD_LANCZOS1_SUCC) {
  std::vector<float> data;
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  int64_t kBatchSize_exp = 4;
  int64_t kNumRowSquares_exp = 20;
  int64_t kNumColSquares_exp = 15;
  int64_t kSquareSize_exp = 13;
  int64_t kNumChannels_exp = 3;
  
  SetCheckerboardImageInput<float> (kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp,
                            kSquareSize_exp, kNumChannels_exp, data);
  vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                     kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

  const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
  const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

  float scale_data[2] = {0.5, 0.5}; 

  float translate_data[2] = {0.5, 0.5}; 
  int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
  vector<int64_t> sizeshape ={2};
  const int outputshape= kBatchSize_exp* kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
  float* output = new float[outputshape];
  
  vector<vector<int64_t>> shapes = {datashape, sizeshape, {2}, {2}, {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}}; // 

  float* data_arr = new float[data.size()];

  std::copy(data.begin(), data.end(), data_arr);
  

  vector<void *> datas = {(void *)data_arr, (void *)size_data, (void *)scale_data, (void *)translate_data, (void *)output};
  std::string kernel_type_str = "lanczos1";
  bool antialias = false;
  CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  delete [] data_arr;
  delete [] output;
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_METHOD_LANCZOS3_SUCC) {
  std::vector<float> data;
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  int64_t kBatchSize_exp = 4;
  int64_t kNumRowSquares_exp = 20;
  int64_t kNumColSquares_exp = 15;
  int64_t kSquareSize_exp = 13;
  int64_t kNumChannels_exp = 3;
  
  SetCheckerboardImageInput<float> (kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp,
                            kSquareSize_exp, kNumChannels_exp, data);
  vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                     kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

  const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
  const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

  float scale_data[2] = {0.5, 0.5}; 

  float translate_data[2] = {0.5, 0.5}; 
  int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
  vector<int64_t> sizeshape ={2};
  const int outputshape= kBatchSize_exp* kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
  float* output = new float[outputshape];
  
  vector<vector<int64_t>> shapes = {datashape, sizeshape, {2}, {2}, {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}}; // 

  float* data_arr = new float[data.size()];

  std::copy(data.begin(), data.end(), data_arr);
  

  vector<void *> datas = {(void *)data_arr, (void *)size_data, (void *)scale_data, (void *)translate_data, (void *)output};
  std::string kernel_type_str = "lanczos3";
  bool antialias = false;
  CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  delete [] data_arr;
  delete [] output;
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_METHOD_LANCZOS5_SUCC) {
  std::vector<float> data;
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  int64_t kBatchSize_exp = 4;
  int64_t kNumRowSquares_exp = 20;
  int64_t kNumColSquares_exp = 15;
  int64_t kSquareSize_exp = 13;
  int64_t kNumChannels_exp = 3;
  
  SetCheckerboardImageInput<float> (kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp,
                            kSquareSize_exp, kNumChannels_exp, data);
  vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                     kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

  const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
  const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

  float scale_data[2] = {0.5, 0.5}; 

  float translate_data[2] = {0.5, 0.5}; 
  int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
  vector<int64_t> sizeshape ={2};
  const int outputshape= kBatchSize_exp* kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
  float* output = new float[outputshape];
  
  vector<vector<int64_t>> shapes = {datashape, sizeshape, {2}, {2}, {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}}; // 

  float* data_arr = new float[data.size()];

  std::copy(data.begin(), data.end(), data_arr);
  

  vector<void *> datas = {(void *)data_arr, (void *)size_data, (void *)scale_data, (void *)translate_data, (void *)output};
  std::string kernel_type_str = "lanczos5";
  bool antialias = false;
  CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  delete [] data_arr;
  delete [] output;
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_METHOD_GAUSSIAN_SUCC) {
  std::vector<float> data;
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  int64_t kBatchSize_exp = 4;
  int64_t kNumRowSquares_exp = 20;
  int64_t kNumColSquares_exp = 15;
  int64_t kSquareSize_exp = 13;
  int64_t kNumChannels_exp = 3;
  
  SetCheckerboardImageInput<float> (kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp,
                            kSquareSize_exp, kNumChannels_exp, data);
  vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                     kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

  const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
  const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

  float scale_data[2] = {0.5, 0.5}; 

  float translate_data[2] = {0.5, 0.5}; 
  int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
  vector<int64_t> sizeshape ={2};
  const int outputshape= kBatchSize_exp* kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
  float* output = new float[outputshape];
  
  vector<vector<int64_t>> shapes = {datashape, sizeshape, {2}, {2}, {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}}; // 

  float* data_arr = new float[data.size()];

  std::copy(data.begin(), data.end(), data_arr);
  

  vector<void *> datas = {(void *)data_arr, (void *)size_data, (void *)scale_data, (void *)translate_data, (void *)output};
  std::string kernel_type_str = "gaussian";
  bool antialias = false;
  CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  delete [] data_arr;
  delete [] output;
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_METHOD_TRIANGLE_SUCC) {
  std::vector<float> data;
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  int64_t kBatchSize_exp = 4;
  int64_t kNumRowSquares_exp = 20;
  int64_t kNumColSquares_exp = 15;
  int64_t kSquareSize_exp = 13;
  int64_t kNumChannels_exp = 3;
  
  SetCheckerboardImageInput<float> (kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp,
                            kSquareSize_exp, kNumChannels_exp, data);
  vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                     kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

  const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
  const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

  float scale_data[2] = {0.5, 0.5}; 

  float translate_data[2] = {0.5, 0.5}; 
  int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
  vector<int64_t> sizeshape ={2};
  const int outputshape= kBatchSize_exp* kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
  float* output = new float[outputshape];
  
  vector<vector<int64_t>> shapes = {datashape, sizeshape, {2}, {2}, {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}}; // 

  float* data_arr = new float[data.size()];

  std::copy(data.begin(), data.end(), data_arr);
  

  vector<void *> datas = {(void *)data_arr, (void *)size_data, (void *)scale_data, (void *)translate_data, (void *)output};
  std::string kernel_type_str = "triangle";
  bool antialias = false;
  CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  delete [] data_arr;
  delete [] output;
}
TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_METHOD_KEYSCUBIC_SUCC) {
  std::vector<float> data;
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  int64_t kBatchSize_exp = 4;
  int64_t kNumRowSquares_exp = 20;
  int64_t kNumColSquares_exp = 15;
  int64_t kSquareSize_exp = 13;
  int64_t kNumChannels_exp = 3;
  
  SetCheckerboardImageInput<float> (kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp,
                            kSquareSize_exp, kNumChannels_exp, data);
  vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                     kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

  const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
  const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

  float scale_data[2] = {0.5, 0.5}; 

  float translate_data[2] = {0.5, 0.5}; 
  int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
  vector<int64_t> sizeshape ={2};
  const int outputshape= kBatchSize_exp* kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
  float* output = new float[outputshape];
  
  vector<vector<int64_t>> shapes = {datashape, sizeshape, {2}, {2}, {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}}; // 

  float* data_arr = new float[data.size()];

  std::copy(data.begin(), data.end(), data_arr);
  

  vector<void *> datas = {(void *)data_arr, (void *)size_data, (void *)scale_data, (void *)translate_data, (void *)output};
  std::string kernel_type_str = "keyscubic";
  bool antialias = false;
  CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  delete [] data_arr;
  delete [] output;
}
TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_METHOD_MITCHELLCUBIC_SUCC) {
  std::vector<float> data;
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  int64_t kBatchSize_exp = 4;
  int64_t kNumRowSquares_exp = 20;
  int64_t kNumColSquares_exp = 15;
  int64_t kSquareSize_exp = 13;
  int64_t kNumChannels_exp = 3;
  
  SetCheckerboardImageInput<float> (kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp,
                            kSquareSize_exp, kNumChannels_exp, data);
  vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                     kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

  const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
  const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

  float scale_data[2] = {0.5, 0.5}; 

  float translate_data[2] = {0.5, 0.5}; 
  int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
  vector<int64_t> sizeshape ={2};
  const int outputshape= kBatchSize_exp* kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
  float* output = new float[outputshape];
  
  vector<vector<int64_t>> shapes = {datashape, sizeshape, {2}, {2}, {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}}; // 

  float* data_arr = new float[data.size()];

  std::copy(data.begin(), data.end(), data_arr);
  

  vector<void *> datas = {(void *)data_arr, (void *)size_data, (void *)scale_data, (void *)translate_data, (void *)output};
  std::string kernel_type_str = "mitchellcubic";
  bool antialias = false;
  CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  delete [] data_arr;
  delete [] output;
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT8_METHOD_BOX_SUCC) {
  std::vector<int8_t> data;
  vector<DataType> data_types = {DT_INT8, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  int64_t kBatchSize_exp = 2;
  int64_t kNumRowSquares_exp = 16;
  int64_t kNumColSquares_exp = 13;
  int64_t kSquareSize_exp = 12;
  int64_t kNumChannels_exp = 3;
  
  SetCheckerboardImageInput<int8_t> (kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp,
                            kSquareSize_exp, kNumChannels_exp, data);
  vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                     kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

  const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
  const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

  float scale_data[2] = {0.5, 0.5}; 

  float translate_data[2] = {0.5, 0.5}; 
  int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
  vector<int64_t> sizeshape ={2};
  const int outputshape= kBatchSize_exp* kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
  float* output = new float[outputshape];
  
  vector<vector<int64_t>> shapes = {datashape, sizeshape, {2}, {2}, {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}}; // 

  int8_t* data_arr = new int8_t[data.size()];

  std::copy(data.begin(), data.end(), data_arr);
  

  vector<void *> datas = {(void *)data_arr, (void *)size_data, (void *)scale_data, (void *)translate_data, (void *)output};
  std::string kernel_type_str = "box";
  bool antialias = false;
  CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  delete [] data_arr;
  delete [] output;
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT16_METHOD_BOX_SUCC) {
  std::vector<int16_t> data;
  vector<DataType> data_types = {DT_INT16, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  int64_t kBatchSize_exp = 2;
  int64_t kNumRowSquares_exp = 16;
  int64_t kNumColSquares_exp = 13;
  int64_t kSquareSize_exp = 12;
  int64_t kNumChannels_exp = 3;
  
  SetCheckerboardImageInput<int16_t> (kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp,
                            kSquareSize_exp, kNumChannels_exp, data);
  vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                     kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

  const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
  const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

  float scale_data[2] = {0.5, 0.5}; 

  float translate_data[2] = {0.5, 0.5}; 
  int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
  vector<int64_t> sizeshape ={2};
  const int outputshape= kBatchSize_exp* kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
  float* output = new float[outputshape];
  
  vector<vector<int64_t>> shapes = {datashape, sizeshape, {2}, {2}, {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}}; // 

  int16_t* data_arr = new int16_t[data.size()];

  std::copy(data.begin(), data.end(), data_arr);
  

  vector<void *> datas = {(void *)data_arr, (void *)size_data, (void *)scale_data, (void *)translate_data, (void *)output};
  std::string kernel_type_str = "box";
  bool antialias = false;
  CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  delete [] data_arr;
  delete [] output;
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT32_METHOD_BOX_SUCC) {
  std::vector<int32_t> data;
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  int64_t kBatchSize_exp = 2;
  int64_t kNumRowSquares_exp = 16;
  int64_t kNumColSquares_exp = 13;
  int64_t kSquareSize_exp = 12;
  int64_t kNumChannels_exp = 3;
  
  SetCheckerboardImageInput<int32_t> (kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp,
                            kSquareSize_exp, kNumChannels_exp, data);
  vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                     kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

  const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
  const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

  float scale_data[2] = {0.5, 0.5}; 

  float translate_data[2] = {0.5, 0.5}; 
  int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
  vector<int64_t> sizeshape ={2};
  const int outputshape= kBatchSize_exp* kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
  float* output = new float[outputshape];
  
  vector<vector<int64_t>> shapes = {datashape, sizeshape, {2}, {2}, {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}}; // 

  int32_t* data_arr = new int32_t[data.size()];

  std::copy(data.begin(), data.end(), data_arr);
  

  vector<void *> datas = {(void *)data_arr, (void *)size_data, (void *)scale_data, (void *)translate_data, (void *)output};
  std::string kernel_type_str = "box";
  bool antialias = false;
  CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  delete [] data_arr;
  delete [] output;

}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_UINT8_METHOD_BOX_SUCC) {
  std::vector<uint8_t> data;
  vector<DataType> data_types = {DT_UINT8, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  int64_t kBatchSize_exp = 2;
  int64_t kNumRowSquares_exp = 16;
  int64_t kNumColSquares_exp = 13;
  int64_t kSquareSize_exp = 12;
  int64_t kNumChannels_exp = 3;
  
  SetCheckerboardImageInput<uint8_t> (kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp,
                            kSquareSize_exp, kNumChannels_exp, data);
  vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                     kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

  const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
  const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

  float scale_data[2] = {0.5, 0.5}; 

  float translate_data[2] = {0.5, 0.5}; 
  int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
  vector<int64_t> sizeshape ={2};
  const int outputshape= kBatchSize_exp* kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
  float* output = new float[outputshape];
  
  vector<vector<int64_t>> shapes = {datashape, sizeshape, {2}, {2}, {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}}; // 

  uint8_t* data_arr = new uint8_t[data.size()];

  std::copy(data.begin(), data.end(), data_arr);
  

  vector<void *> datas = {(void *)data_arr, (void *)size_data, (void *)scale_data, (void *)translate_data, (void *)output};
  std::string kernel_type_str = "box";
  bool antialias = false;
  CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  delete [] data_arr;
  delete [] output;
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_UINT16_METHOD_BOX_SUCC) {
  std::vector<uint16_t> data;
  vector<DataType> data_types = {DT_UINT16, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  int64_t kBatchSize_exp = 2;
  int64_t kNumRowSquares_exp = 16;
  int64_t kNumColSquares_exp = 13;
  int64_t kSquareSize_exp = 12;
  int64_t kNumChannels_exp = 3;
  
  SetCheckerboardImageInput<uint16_t> (kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp,
                            kSquareSize_exp, kNumChannels_exp, data);
  vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                     kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

  const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
  const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

  float scale_data[2] = {0.5, 0.5}; 

  float translate_data[2] = {0.5, 0.5}; 
  int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
  vector<int64_t> sizeshape ={2};
  const int outputshape= kBatchSize_exp* kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
  float* output = new float[outputshape];
  
  vector<vector<int64_t>> shapes = {datashape, sizeshape, {2}, {2}, {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}}; // 

  uint16_t* data_arr = new uint16_t[data.size()];

  std::copy(data.begin(), data.end(), data_arr);
  

  vector<void *> datas = {(void *)data_arr, (void *)size_data, (void *)scale_data, (void *)translate_data, (void *)output};
  std::string kernel_type_str = "box";
  bool antialias = false;
  CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  delete [] data_arr;
  delete [] output;
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT64_METHOD_BOX_SUCC) {
  std::vector<int64_t> data;
  vector<DataType> data_types = {DT_INT64, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  int64_t kBatchSize_exp = 2;
  int64_t kNumRowSquares_exp = 16;
  int64_t kNumColSquares_exp = 13;
  int64_t kSquareSize_exp = 12;
  int64_t kNumChannels_exp = 3;
  
  SetCheckerboardImageInput<int64_t> (kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp,
                            kSquareSize_exp, kNumChannels_exp, data);
  vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                     kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

  const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
  const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

  float scale_data[2] = {0.5, 0.5}; 

  float translate_data[2] = {0.5, 0.5}; 
  int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
  vector<int64_t> sizeshape ={2};
  const int outputshape= kBatchSize_exp* kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
  float* output = new float[outputshape];
  
  vector<vector<int64_t>> shapes = {datashape, sizeshape, {2}, {2}, {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}}; // 

  int64_t* data_arr = new int64_t[data.size()];

  std::copy(data.begin(), data.end(), data_arr);
  

  vector<void *> datas = {(void *)data_arr, (void *)size_data, (void *)scale_data, (void *)translate_data, (void *)output};
  std::string kernel_type_str = "box";
  bool antialias = false;
  CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  delete [] data_arr;
  delete [] output;

}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT16_METHOD_BOX_SUCC) {
  std::vector<Eigen::half> data;
  vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  int64_t kBatchSize_exp = 2;
  int64_t kNumRowSquares_exp = 16;
  int64_t kNumColSquares_exp = 13;
  int64_t kSquareSize_exp = 12;
  int64_t kNumChannels_exp = 3;
  
  SetCheckerboardImageInput<Eigen::half> (kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp,
                            kSquareSize_exp, kNumChannels_exp, data);
  vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                     kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

  const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
  const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

  float scale_data[2] = {0.5, 0.5}; 

  float translate_data[2] = {0.5, 0.5}; 
  int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
  vector<int64_t> sizeshape ={2};
  const int outputshape= kBatchSize_exp* kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
  float* output = new float[outputshape];
  
  vector<vector<int64_t>> shapes = {datashape, sizeshape, {2}, {2}, {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}}; // 

  Eigen::half* data_arr = new Eigen::half[data.size()];

  std::copy(data.begin(), data.end(), data_arr);
  

  vector<void *> datas = {(void *)data_arr, (void *)size_data, (void *)scale_data, (void *)translate_data, (void *)output};
  std::string kernel_type_str = "box";
  bool antialias = false;
  CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  delete [] data_arr;
  delete [] output;

}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_DOUBLE_METHOD_BOX_SUCC) {
  std::vector<double> data;
  vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  int64_t kBatchSize_exp = 2;
  int64_t kNumRowSquares_exp = 16;
  int64_t kNumColSquares_exp = 13;
  int64_t kSquareSize_exp = 12;
  int64_t kNumChannels_exp = 3;
  
  SetCheckerboardImageInput<double> (kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp,
                            kSquareSize_exp, kNumChannels_exp, data);
  vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                     kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

  const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
  const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

  float scale_data[2] = {0.5, 0.5}; 

  float translate_data[2] = {0.5, 0.5}; 
  int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
  vector<int64_t> sizeshape ={2};
  const int outputshape= kBatchSize_exp* kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
  float* output = new float[outputshape];
  
  vector<vector<int64_t>> shapes = {datashape, sizeshape, {2}, {2}, {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}}; // 

  double* data_arr = new double[data.size()];

  std::copy(data.begin(), data.end(), data_arr);
  

  vector<void *> datas = {(void *)data_arr, (void *)size_data, (void *)scale_data, (void *)translate_data, (void *)output};
  std::string kernel_type_str = "box";
  bool antialias = false;
  CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  delete [] data_arr;
  delete [] output;

}

// exception instance
TEST_F(TEST_SCALEANDTRANSLATE_UT, INPUT_DATA_TYPE_EXCEPTION) {
  std::vector<uint64_t> data;
  vector<DataType> data_types = {DT_UINT64, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};

  u_int64_t data_arr[4] = {(u_int64_t)1};
  vector<int64_t> datashape = {1, 2 , 2 ,1};
  float scale_data[2] = {0.5, 0.5}; 

  float translate_data[2] = {0.5, 0.5}; 
  int size_data[2] = {4, 4};
  vector<int64_t> sizeshape ={2};
  const int outputshape= 16;
  float* output = new float[outputshape];
  
  vector<vector<int64_t>> shapes = {datashape, sizeshape, {2}, {2}, {1, 4, 4, 1}}; // 

  vector<void *> datas = {(void *)data_arr, (void *)size_data, (void *)scale_data, (void *)translate_data, (void *)output};
  std::string kernel_type_str = "box";
  bool antialias = false;
  CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
  delete [] output;
}
