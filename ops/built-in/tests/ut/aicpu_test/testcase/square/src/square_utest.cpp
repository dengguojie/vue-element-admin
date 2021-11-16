/**
 * Copyright 2021 Huawei Technologies Co., Ltd.
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

#include <gtest/gtest.h>

#include <malloc.h>
#include <Eigen/Core>
#include <complex>
#include <iostream>
#include "aicpu_read_file.h"
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"

class TEST_SQUARE_UT : public testing::Test {
 protected:
  std::float_t *float_null_{nullptr};
  std::float_t float_0_[0];
  std::float_t float_12_[12]{1.0f};
  std::float_t float_16_[16]{0.0f};
  std::int32_t int32_22_[22]{1};
  std::int64_t int64_22_[22]{0L};
  bool bool_22_[22]{true};
};

template <typename T>
inline aicpu::DataType ToDataType() {
  return aicpu::DataType::DT_UNDEFINED;
}

template <>
inline aicpu::DataType ToDataType<bool>() {
  return aicpu::DataType::DT_BOOL;
}

template <>
inline aicpu::DataType ToDataType<Eigen::half>() {
  return aicpu::DataType::DT_FLOAT16;
}

template <>
inline aicpu::DataType ToDataType<std::float_t>() {
  return aicpu::DataType::DT_FLOAT;
}

template <>
inline aicpu::DataType ToDataType<std::double_t>() {
  return aicpu::DataType::DT_DOUBLE;
}

template <>
inline aicpu::DataType ToDataType<std::int8_t>() {
  return aicpu::DataType::DT_INT8;
}

template <>
inline aicpu::DataType ToDataType<std::int16_t>() {
  return aicpu::DataType::DT_INT16;
}

template <>
inline aicpu::DataType ToDataType<std::int32_t>() {
  return aicpu::DataType::DT_INT32;
}

template <>
inline aicpu::DataType ToDataType<std::int64_t>() {
  return aicpu::DataType::DT_INT64;
}

template <>
inline aicpu::DataType ToDataType<std::uint8_t>() {
  return aicpu::DataType::DT_UINT8;
}

template <>
inline aicpu::DataType ToDataType<std::uint16_t>() {
  return aicpu::DataType::DT_UINT16;
}

template <>
inline aicpu::DataType ToDataType<std::uint32_t>() {
  return aicpu::DataType::DT_UINT32;
}

template <>
inline aicpu::DataType ToDataType<std::uint64_t>() {
  return aicpu::DataType::DT_UINT64;
}

template <>
inline aicpu::DataType ToDataType<std::complex<std::float_t> >() {
  return aicpu::DataType::DT_COMPLEX64;
}
template <>
inline aicpu::DataType ToDataType<std::complex<std::double_t> >() {
  return aicpu::DataType::DT_COMPLEX128;
}

template <typename T>
inline const char *ToDataName() {
  return typeid(T).name();
}

template <>
inline const char *ToDataName<Eigen::half>() {
  return "float16";
}

template <>
inline const char *ToDataName<std::float_t>() {
  return "float32";
}

template <>
inline const char *ToDataName<std::double_t>() {
  return "float64";
}

template <>
inline const char *ToDataName<std::int8_t>() {
  return "int8";
}

template <>
inline const char *ToDataName<std::int16_t>() {
  return "int16";
}

template <>
inline const char *ToDataName<std::int32_t>() {
  return "int32";
}

template <>
inline const char *ToDataName<std::int64_t>() {
  return "int64";
}

template <>
inline const char *ToDataName<std::uint8_t>() {
  return "uint8";
}

template <>
inline const char *ToDataName<std::uint16_t>() {
  return "uint16";
}

template <>
inline const char *ToDataName<std::uint32_t>() {
  return "uint32";
}

template <>
inline const char *ToDataName<std::uint64_t>() {
  return "uint64";
}

template <>
inline const char *ToDataName<std::complex<std::float_t> >() {
  return "complex64";
}
template <>
inline const char *ToDataName<std::complex<std::double_t> >() {
  return "complex128";
}
inline std::uint64_t SizeOf(std::vector<std::int64_t> &shape) {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

template <std::shared_ptr<aicpu::Device> aicpu::CpuKernelContext::*DEVICE_PTR>
struct Friend {
  friend void SetDeviceNull(aicpu::CpuKernelContext &ctx) {
    ctx.*DEVICE_PTR = nullptr;
  }
};

template struct Friend<&aicpu::CpuKernelContext::device_>;
void SetDeviceNull(aicpu::CpuKernelContext &ctx);

inline void RunKernelSquare(std::shared_ptr<aicpu::NodeDef> node_def,
                            aicpu::DeviceType device_type, uint32_t expect,
                            bool bad_kernel = false) {
  std::string node_def_str;
  node_def->SerializeToString(node_def_str);
  aicpu::CpuKernelContext ctx(device_type);
  EXPECT_EQ(ctx.Init(node_def.get()), aicpu::KernelStatus::KERNEL_STATUS_OK);
  if (bad_kernel) {
    SetDeviceNull(ctx);
  }
  std::uint32_t ret{aicpu::CpuKernelRegister::Instance().RunCpuKernel(ctx)};
  EXPECT_EQ(ret, expect);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelSquare(
    const std::vector<std::int64_t> &dims_in,
    const std::vector<std::int64_t> &dims_out, Tin *input, Tout *output,
    aicpu::KernelStatus status = aicpu::KernelStatus::KERNEL_STATUS_OK,
    bool bad_kernel = false) {
  const auto data_type_in{ToDataType<Tin>()};
  const auto data_type_out{ToDataType<Tout>()};
  EXPECT_NE(data_type_in, aicpu::DataType::DT_UNDEFINED);
  EXPECT_NE(data_type_out, aicpu::DataType::DT_UNDEFINED);
  auto node_def{aicpu::CpuKernelUtils::CreateNodeDef()};
  aicpu::NodeDefBuilder(node_def.get(), "Square", "Square")
      .Input({"x", data_type_in, dims_in, input})
      .Output({"y", data_type_out, dims_out, output});
  RunKernelSquare(node_def, aicpu::DeviceType::HOST, status, bad_kernel);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelSquare(
    const std::vector<std::int64_t> &dims, Tin *input, Tout *output,
    aicpu::KernelStatus status = aicpu::KernelStatus::KERNEL_STATUS_OK,
    bool bad_kernel = false) {
  CreateAndRunKernelSquare(dims, dims, input, output, status, bad_kernel);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelSquareParamInvalid(
    const std::vector<std::int64_t> &dims_in,
    const std::vector<std::int64_t> &dims_out, Tin *input, Tout *output) {
  CreateAndRunKernelSquare(dims_in, dims_out, input, output,
                           aicpu::KernelStatus::KERNEL_STATUS_PARAM_INVALID);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelSquareParamInvalid(const std::vector<std::int64_t> &dims,
                                          Tin *input, Tout *output) {
  CreateAndRunKernelSquareParamInvalid(dims, dims, input, output);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelSquareInnerError(
    const std::vector<std::int64_t> &dims_in,
    const std::vector<std::int64_t> &dims_out, Tin *input, Tout *output) {
  CreateAndRunKernelSquare(dims_in, dims_out, input, output,
                           aicpu::KernelStatus::KERNEL_STATUS_INNER_ERROR,
                           true);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelSquareInnerError(const std::vector<std::int64_t> &dims,
                                        Tin *input, Tout *output) {
  CreateAndRunKernelSquareInnerError(dims, dims, input, output);
}

template <typename T>
bool ReadBinFile(std::string file_name, T buf[], std::size_t size) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }
    in_file.read(reinterpret_cast<char *>(buf), size * sizeof(buf[0]));
    in_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << file_name << " failed, " << e.what()
              << std::endl;
    return false;
  }
  return true;
}

template <typename T>
bool WriteBinFile(std::string file_name, T buf[], std::size_t size) {
  try {
    std::ofstream out_file{file_name};
    if (!out_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }
    out_file.write(reinterpret_cast<char *>(buf), size * sizeof(buf[0]));
    out_file.close();
  } catch (std::exception &e) {
    std::cout << "write file " << file_name << " failed, " << e.what()
              << std::endl;
    return false;
  }
  return true;
}

template <typename T>
bool WriteFile(std::string file_name, T buf[], std::size_t size) {
  try {
    std::ofstream out_file{file_name};
    if (!out_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }
    out_file << std::setprecision(std::numeric_limits<T>::digits10 + 1);
    for (auto index{0}; index < size; index++) {
      out_file << buf[index] << '\n';
    }
    out_file.close();
  } catch (std::exception &e) {
    std::cout << "write file " << file_name << " failed, " << e.what()
              << std::endl;
    return false;
  }
  return true;
}

template <typename T>
void RunTestSquare(int flag) {
  const auto data_name{ToDataName<T>()};
  if (flag == 0) {
    std::uint64_t dim[1];
    std::string data_dim_path{ktestcaseFilePath + "square/data/square_data_" +
                              data_name + "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_path, dim, 1), true);

    std::uint64_t shape[dim[0]];
    std::string data_shape_path{ktestcaseFilePath + "square/data/square_data_" +
                                data_name + "_shape.txt"};
    EXPECT_EQ(ReadFile(data_shape_path, shape, dim[0]), true);

    std::vector<std::int64_t> dims(shape, shape + dim[0]);
    auto input1_size{SizeOf(dims)};

    // T data1[input1_size];
    // T* data1 = new T[input1_size];
    T *data1 = (T *)malloc(input1_size * sizeof(T));
    std::string data_path{ktestcaseFilePath + "square/data/square_data_input_" +
                          data_name + ".bin"};
    EXPECT_EQ(ReadBinFile(data_path, data1, input1_size), true);
    // T output[input1_size];
    // T* output = new T[input1_size];
    T *output = (T *)malloc(input1_size * sizeof(T));
    CreateAndRunKernelSquare(dims, data1, output);
    std::string out_data_actual_path{ktestcaseFilePath +
                                     "square/data/square_data_output_" +
                                     data_name + "_actual.txt"};
    EXPECT_EQ(WriteFile(out_data_actual_path, output, input1_size), true);

    // T expect_out[input1_size];
    // T* expect_out = new T[input1_size];
    T *expect_out = (T *)malloc(input1_size * sizeof(T));
    std::string out_data_path{ktestcaseFilePath +
                              "square/data/square_data_output_" + data_name +
                              ".bin"};
    EXPECT_EQ(ReadBinFile(out_data_path, expect_out, input1_size), true);
    EXPECT_EQ(CompareResult(output, expect_out, input1_size), true);
    free(data1);
    free(output);
    free(expect_out);
  } else if (flag == 1) {
    std::uint64_t dim[1];
    std::string data_dim_path{ktestcaseFilePath +
                              "square/data/square_bigdata_" + data_name +
                              "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_path, dim, 1), true);

    std::uint64_t shape[dim[0]];
    std::string data_shape_path{ktestcaseFilePath +
                                "square/data/square_bigdata_" + data_name +
                                "_shape.txt"};
    EXPECT_EQ(ReadFile(data_shape_path, shape, dim[0]), true);

    std::vector<std::int64_t> dims(shape, shape + dim[0]);
    auto input1_size{SizeOf(dims)};

    // T data1[input1_size];
    // T* data1 = new T[input1_size];
    T *data1 = (T *)malloc(input1_size * sizeof(T));
    std::string data_path{ktestcaseFilePath +
                          "square/data/square_bigdata_input_" + data_name +
                          ".bin"};
    EXPECT_EQ(ReadBinFile(data_path, data1, input1_size), true);

    // T output[input1_size];
    // T* output = new T[input1_size];
    T *output = (T *)malloc(input1_size * sizeof(T));
    CreateAndRunKernelSquare(dims, data1, output);
    std::string out_data_actual_path{ktestcaseFilePath +
                                     "square/data/square_bigdata_output_" +
                                     data_name + "_actual.txt"};
    EXPECT_EQ(WriteFile(out_data_actual_path, output, input1_size), true);

    // T expect_out[input1_size];
    // T* expect_out = new T[input1_size];
    T *expect_out = (T *)malloc(input1_size * sizeof(T));
    std::string out_data_path{ktestcaseFilePath +
                              "square/data/square_bigdata_output_" + data_name +
                              ".bin"};
    EXPECT_EQ(ReadBinFile(out_data_path, expect_out, input1_size), true);
    EXPECT_EQ(CompareResult(output, expect_out, input1_size), true);
    free(data1);
    free(output);
    free(expect_out);
  }
}

#define ADD_CASE(base_type, aicpu_type)            \
  TEST_F(TEST_SQUARE_UT, DATA_TYPE_##aicpu_type) { \
    RunTestSquare<base_type>(0);                   \
  }

#define ADD_BIGDATA_CASE(base_type, aicpu_type)       \
  TEST_F(TEST_SQUARE_UT, BIGDATA_TYPE_##aicpu_type) { \
    RunTestSquare<base_type>(1);                      \
  }
// exception inssquarece
TEST_F(TEST_SQUARE_UT, BAD_KERNEL_EXCEPTION) {
  CreateAndRunKernelSquareInnerError({2, 6}, float_12_, float_12_);
}

TEST_F(TEST_SQUARE_UT, INPUT_SHAPE_EXCEPTION) {
  CreateAndRunKernelSquareParamInvalid({2, 6}, {2, 8}, float_12_, float_16_);
}

TEST_F(TEST_SQUARE_UT, INPUT_DIM_EXCEPTION) {
  CreateAndRunKernelSquareParamInvalid({2, 6}, {6, 2}, float_12_, float_12_);
}

TEST_F(TEST_SQUARE_UT, INPUT_DIMSIZE_EXCEPTION) {
  CreateAndRunKernelSquareParamInvalid({2, 2, 3}, {6, 2}, float_12_, float_12_);
}

TEST_F(TEST_SQUARE_UT, INPUT_DTYPE_EXCEPTION) {
  CreateAndRunKernelSquareParamInvalid({2, 11}, int32_22_, int64_22_);
}

TEST_F(TEST_SQUARE_UT, INPUT_NULL_EXCEPTION) {
  CreateAndRunKernelSquareParamInvalid({2, 11}, float_null_, float_null_);
}

TEST_F(TEST_SQUARE_UT, OUTPUT_NULL_EXCEPTION) {
  CreateAndRunKernelSquareParamInvalid({0, 0}, float_0_, float_null_);
}

TEST_F(TEST_SQUARE_UT, NO_OUTPUT_EXCEPTION) {
  const auto data_type_in{ToDataType<std::float_t>()};
  auto node_def{aicpu::CpuKernelUtils::CreateNodeDef()};
  aicpu::NodeDefBuilder(node_def.get(), "Square", "Square")
      .Input({"x", data_type_in, {2, 6}, float_12_});
  RunKernelSquare(node_def, aicpu::DeviceType::HOST,
                  aicpu::KernelStatus::KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SQUARE_UT, INPUT_BOOL_UNSUPPORT) {
  CreateAndRunKernelSquareParamInvalid({2, 11}, bool_22_, bool_22_);
}

ADD_CASE(std::int32_t, DT_INT32)
ADD_CASE(std::int64_t, DT_INT64)
ADD_CASE(Eigen::half, DT_FLOAT16)
ADD_CASE(std::float_t, DT_FLOAT)
ADD_CASE(std::double_t, DT_DOUBLE)
ADD_CASE(std::complex<std::float_t>, DT_COMPLEX64)
ADD_CASE(std::complex<std::double_t>, DT_COMPLEX128)

ADD_BIGDATA_CASE(std::int32_t, DT_INT32)
ADD_BIGDATA_CASE(Eigen::half, DT_FLOAT16)
