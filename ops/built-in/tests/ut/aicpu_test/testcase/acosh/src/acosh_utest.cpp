/**
 * Copyright 2021 Jilin University
 * Copyright 2020 Huawei Technologies Co., Ltd.
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

#include <Eigen/Core>
#include <iostream>

#include "aicpu_read_file.h"
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"

class TEST_ACOSH_UT : public testing::Test {
 protected:
  std::float_t *float_null_{nullptr};
  std::float_t float_0_[0];
  std::float_t float_12_[12]{1.0f};
  std::float_t float_12_nan_[12]{NAN};
  std::complex<std::float_t> complex_float_0_[0];
  std::complex<std::float_t> complex_float_12_[12]{1.0f};
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
inline aicpu::DataType ToDataType<std::complex<std::float_t>>() {
  return aicpu::DataType::DT_COMPLEX64;
}

template <>
inline aicpu::DataType ToDataType<std::complex<std::double_t>>() {
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
inline const char *ToDataName<std::complex<std::float_t>>() {
  return "complex64";
}

template <>
inline const char *ToDataName<std::complex<std::double_t>>() {
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

inline void RunKernelAcosh(std::shared_ptr<aicpu::NodeDef> node_def,
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
void CreateAndRunKernelAcosh(
    const std::vector<std::int64_t> &dims_in,
    const std::vector<std::int64_t> &dims_out, Tin *input0, Tout *output,
    aicpu::KernelStatus status = aicpu::KernelStatus::KERNEL_STATUS_OK,
    bool bad_kernel = false) {
  const auto data_type_in{ToDataType<Tin>()};
  const auto data_type_out{ToDataType<Tout>()};
  EXPECT_NE(data_type_in, aicpu::DataType::DT_UNDEFINED);
  EXPECT_NE(data_type_out, aicpu::DataType::DT_UNDEFINED);
  auto node_def{aicpu::CpuKernelUtils::CreateNodeDef()};
  aicpu::NodeDefBuilder(node_def.get(), "Acosh", "Acosh")
      .Input({"x", data_type_in, dims_in, input0})
      .Output({"output", data_type_out, dims_out, output});
  RunKernelAcosh(node_def, aicpu::DeviceType::HOST, status, bad_kernel);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelAcosh(
    const std::vector<std::int64_t> &dims, Tin *input0, Tout *output,
    aicpu::KernelStatus status = aicpu::KernelStatus::KERNEL_STATUS_OK,
    bool bad_kernel = false) {
  CreateAndRunKernelAcosh(dims, dims, input0, output, status, bad_kernel);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelAcoshParamInvalid(
    const std::vector<std::int64_t> &dims_in,
    const std::vector<std::int64_t> &dims_out, Tin *input0, Tout *output) {
  CreateAndRunKernelAcosh(dims_in, dims_out, input0, output,
                          aicpu::KernelStatus::KERNEL_STATUS_PARAM_INVALID);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelAcoshParamInvalid(const std::vector<std::int64_t> &dims,
                                         Tin *input0, Tout *output) {
  CreateAndRunKernelAcoshParamInvalid(dims, dims, input0, output);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelAcoshInnerError(
    const std::vector<std::int64_t> &dims_in,
    const std::vector<std::int64_t> &dims_out, Tin *input0, Tout *output) {
  CreateAndRunKernelAcosh(dims_in, dims_out, input0, output,
                          aicpu::KernelStatus::KERNEL_STATUS_INNER_ERROR, true);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelAcoshInnerError(const std::vector<std::int64_t> &dims,
                                       Tin *input0, Tout *output) {
  CreateAndRunKernelAcoshInnerError(dims, dims, input0, output);
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

template <typename T, int N>
bool WriteFile(std::string file_name, T buf[], std::size_t size) {
  try {
    std::ofstream out_file{file_name};
    if (!out_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }
    out_file << std::setprecision(N);
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
bool WriteFile(std::string file_name, T buf[], std::size_t size) {
  return WriteFile<T, std::numeric_limits<T>::digits10 + 1>(file_name, buf,
                                                            size);
}

template <typename T>
bool WriteFile(std::string file_name, std::complex<T> buf[], std::size_t size) {
  return WriteFile<std::complex<T>, std::numeric_limits<T>::digits10 + 1>(
      file_name, buf, size);
}

template <typename Tin, typename Tout>
void RunTestAcosh() {
  const auto data_name{ToDataName<Tin>()};

  std::uint64_t dim[1];
  std::string data_dim_path{ktestcaseFilePath + "acosh/data/acosh_data_" +
                            data_name + "_dim.txt"};
  EXPECT_EQ(ReadFile(data_dim_path, dim, 1), true);

  std::uint64_t shape[dim[0]];
  std::string data_shape_path{ktestcaseFilePath + "acosh/data/acosh_data_" +
                              data_name + "_shape.txt"};
  EXPECT_EQ(ReadFile(data_shape_path, shape, dim[0]), true);

  std::vector<std::int64_t> dims(shape, shape + dim[0]);
  auto output_size{SizeOf(dims)};
  auto input_size{output_size};
  Tin data0[input_size];
  std::string data_path0{ktestcaseFilePath + "acosh/data/acosh_data_input_0_" +
                         data_name + ".bin"};
  EXPECT_EQ(ReadBinFile(data_path0, data0, input_size), true);

  Tout output[output_size];
  CreateAndRunKernelAcosh(dims, data0, output);
  std::string out_data_actual_path{ktestcaseFilePath +
                                   "acosh/data/acosh_data_output_" + data_name +
                                   "_actual.txt"};
  EXPECT_EQ(WriteFile(out_data_actual_path, output, output_size), true);
  std::string out_data_actual_bin_path{ktestcaseFilePath +
                                       "acosh/data/acosh_data_output_" +
                                       data_name + "_actual.bin"};
  EXPECT_EQ(WriteBinFile(out_data_actual_bin_path, output, output_size), true);

  Tout expect_out[output_size];
  std::string out_data_path{ktestcaseFilePath +
                            "acosh/data/acosh_data_output_" + data_name +
                            "_expect.bin"};
  EXPECT_EQ(ReadBinFile(out_data_path, expect_out, output_size), true);
  EXPECT_EQ(CompareResult(output, expect_out, output_size), true);
}

template <typename T>
void RunTestAcosh() {
  RunTestAcosh<T, T>();
}

TEST_F(TEST_ACOSH_UT, DATA_TYPE_DT_FLOAT16) { RunTestAcosh<Eigen::half>(); }

TEST_F(TEST_ACOSH_UT, DATA_TYPE_DT_FLOAT) { RunTestAcosh<std::float_t>(); }

TEST_F(TEST_ACOSH_UT, DATA_TYPE_DT_DOUBLE) { RunTestAcosh<std::double_t>(); }

TEST_F(TEST_ACOSH_UT, DATA_TYPE_DT_COMPLEX64) {
  RunTestAcosh<std::complex<std::float_t>>();
}

TEST_F(TEST_ACOSH_UT, DATA_TYPE_DT_COMPLEX128) {
  RunTestAcosh<std::complex<std::double_t>>();
}
// exception instance
TEST_F(TEST_ACOSH_UT, BAD_KERNEL_EXCEPTION) {
  CreateAndRunKernelAcoshInnerError({2000, 6000}, float_0_, float_0_);
}

TEST_F(TEST_ACOSH_UT, INPUT_SHAPE_EXCEPTION) {
  CreateAndRunKernelAcoshParamInvalid({2, 6}, {2, 8}, float_12_, float_16_);
}

TEST_F(TEST_ACOSH_UT, INPUT_DTYPE_EXCEPTION) {
  CreateAndRunKernelAcoshParamInvalid({2, 11}, int32_22_, int64_22_);
}

TEST_F(TEST_ACOSH_UT, INPUT_NULL_EXCEPTION) {
  CreateAndRunKernelAcoshParamInvalid({2, 11}, float_null_, float_null_);
}

TEST_F(TEST_ACOSH_UT, OUTPUT_NULL_EXCEPTION) {
  CreateAndRunKernelAcoshParamInvalid({0, 0}, float_0_, float_null_);
}

TEST_F(TEST_ACOSH_UT, NO_OUTPUT_EXCEPTION) {
  const auto data_type_in{ToDataType<std::float_t>()};
  auto node_def{aicpu::CpuKernelUtils::CreateNodeDef()};
  aicpu::NodeDefBuilder(node_def.get(), "Acosh", "Acosh")
      .Input({"x", data_type_in, {2, 6}, float_12_});
  RunKernelAcosh(node_def, aicpu::DeviceType::HOST,
                 aicpu::KernelStatus::KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ACOSH_UT, INPUT_BOOL_UNSUPPORT) {
  CreateAndRunKernelAcoshParamInvalid({2, 11}, bool_22_, bool_22_);
}
