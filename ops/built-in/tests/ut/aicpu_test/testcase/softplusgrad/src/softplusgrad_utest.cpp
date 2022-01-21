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

#include <Eigen/Core>
#include <complex>
#include <iostream>
#include "aicpu_read_file.h"
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"

class TEST_SOFTPLUSGRAD_UT : public testing::Test {
protected:
  std::float_t* float_null_{nullptr};
  std::float_t float_0_[0];
  std::float_t float_12_[12]{1.0f};
  std::float_t float_16_[16]{1.0f};
  std::int64_t int64_22_[22]{0L};
  std::double_t double_16_[16]{0.0f};
  std::int32_t int32_16_[16]{1};
  std::int32_t int32_22_[22]{1};
  std::float_t float_empty_[0]{};
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
inline const char* ToDataName() {
  return typeid(T).name();
}

template <>
inline const char* ToDataName<Eigen::half>() {
  return "float16";
}

template <>
inline const char* ToDataName<std::float_t>() {
  return "float32";
}

template <>
inline const char* ToDataName<std::double_t>() {
  return "float64";
}

template <>
inline const char* ToDataName<std::int8_t>() {
  return "int8";
}

template <>
inline const char* ToDataName<std::int16_t>() {
  return "int16";
}

template <>
inline const char* ToDataName<std::int32_t>() {
  return "int32";
}

template <>
inline const char* ToDataName<std::int64_t>() {
  return "int64";
}

template <>
inline const char* ToDataName<std::uint8_t>() {
  return "uint8";
}

template <>
inline const char* ToDataName<std::uint16_t>() {
  return "uint16";
}

template <>
inline const char* ToDataName<std::uint32_t>() {
  return "uint32";
}

template <>
inline const char* ToDataName<std::uint64_t>() {
  return "uint64";
}

template <>
inline const char* ToDataName<std::complex<std::float_t> >() {
  return "complex64";
}
template <>
inline const char* ToDataName<std::complex<std::double_t> >() {
  return "complex128";
}
inline std::uint64_t SizeOf(std::vector<std::int64_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

template <std::shared_ptr<aicpu::Device> aicpu::CpuKernelContext::*DEVICE_PTR>
struct Friend {
  friend void SetDeviceNull(aicpu::CpuKernelContext& ctx) {
    ctx.*DEVICE_PTR = nullptr;
  }
};

template struct Friend<&aicpu::CpuKernelContext::device_>;
void SetDeviceNull(aicpu::CpuKernelContext& ctx);

inline void RunKernelSoftplusGrad(std::shared_ptr<aicpu::NodeDef> node_def,
                                  aicpu::DeviceType device_type,
                                  uint32_t expect, bool bad_kernel = false) {
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
void CreateAndRunKernelSoftplusGrad(
    const std::vector<std::int64_t>& dims_in_gradients,
    const std::vector<std::int64_t>& dims_in_features,
    const std::vector<std::int64_t>& dims_out, Tin* input_gradients,
    Tin* input_features, Tout* output,
    aicpu::KernelStatus status = aicpu::KernelStatus::KERNEL_STATUS_OK,
    bool bad_kernel = false) {
  const auto data_type_in_gradients{ToDataType<Tin>()};
  const auto data_type_in_features{ToDataType<Tin>()};
  const auto data_type_out{ToDataType<Tout>()};
  EXPECT_NE(data_type_in_gradients, aicpu::DataType::DT_UNDEFINED);
  EXPECT_NE(data_type_in_features, aicpu::DataType::DT_UNDEFINED);
  EXPECT_NE(data_type_out, aicpu::DataType::DT_UNDEFINED);
  auto node_def{aicpu::CpuKernelUtils::CreateNodeDef()};
  aicpu::NodeDefBuilder(node_def.get(), "SoftplusGrad", "SoftplusGrad")
      .Input({"gradients", data_type_in_gradients, dims_in_gradients,
              input_gradients})
      .Input(
          {"features", data_type_in_features, dims_in_features, input_features})
      .Output({"backprops", data_type_out, dims_out, output});
  RunKernelSoftplusGrad(node_def, aicpu::DeviceType::HOST, status, bad_kernel);
}

template <typename Tin1, typename Tin2, typename Tout>
void CreateAndRunKernelNullptrSoftplusGrad(
    const std::vector<std::int64_t>& dims_in_gradients,
    const std::vector<std::int64_t>& dims_in_features,
    const std::vector<std::int64_t>& dims_out, Tin1 input_gradients,
    Tin2 input_features, Tout output,
    aicpu::KernelStatus status = aicpu::KernelStatus::KERNEL_STATUS_OK,
    bool bad_kernel = false) {
  const auto data_type_in_gradients{ToDataType<Tin1>()};
  const auto data_type_in_features{ToDataType<Tin2>()};
  const auto data_type_out{ToDataType<Tout>()};
  EXPECT_NE(data_type_in_gradients, aicpu::DataType::DT_UNDEFINED);
  EXPECT_NE(data_type_in_features, aicpu::DataType::DT_UNDEFINED);
  EXPECT_NE(data_type_out, aicpu::DataType::DT_UNDEFINED);
  auto node_def{aicpu::CpuKernelUtils::CreateNodeDef()};
  aicpu::NodeDefBuilder(node_def.get(), "SoftplusGrad", "SoftplusGrad")
      .Input({"gradients", data_type_in_gradients, dims_in_gradients,
              input_gradients})
      .Input(
          {"features", data_type_in_features, dims_in_features, input_features})
      .Output({"backprops", data_type_out, dims_out, output});
  RunKernelSoftplusGrad(node_def, aicpu::DeviceType::HOST, status, bad_kernel);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelSoftplusGrad(
    const std::vector<std::int64_t>& dims, Tin* input_gradients,
    Tin* input_features, Tout* output,
    aicpu::KernelStatus status = aicpu::KernelStatus::KERNEL_STATUS_OK,
    bool bad_kernel = false) {
  CreateAndRunKernelSoftplusGrad(dims, dims, dims, input_gradients,
                                 input_features, output, status, bad_kernel);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelSoftplusGradParamInvalid(
    const std::vector<std::int64_t>& dims_in_gradients,
    const std::vector<std::int64_t>& dims_in_features,
    const std::vector<std::int64_t>& dims_out, Tin* input_gradients,
    Tin* input_features, Tout* output) {
  CreateAndRunKernelSoftplusGrad(
      dims_in_gradients, dims_in_features, dims_out, input_gradients,
      input_features, output, aicpu::KernelStatus::KERNEL_STATUS_PARAM_INVALID);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelSoftplusGradParamInvalid(
    const std::vector<std::int64_t>& dims, Tin* input_gradients,
    Tin* input_features, Tout* output) {
  CreateAndRunKernelSoftplusGradParamInvalid(dims, dims, dims, input_gradients,
                                             input_features, output);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelSoftplusGradInnerError(
    const std::vector<std::int64_t>& dims_in_gradients,
    const std::vector<std::int64_t>& dims_in_features,
    const std::vector<std::int64_t>& dims_out, Tin* input_gradients,
    Tin* input_features, Tout* output) {
  CreateAndRunKernelSoftplusGrad(dims_in_gradients, dims_in_features, dims_out,
                                 input_gradients, input_features, output,
                                 aicpu::KernelStatus::KERNEL_STATUS_INNER_ERROR,
                                 true);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelSoftplusGradInnerError(
    const std::vector<std::int64_t>& dims, Tin* input_gradients,
    Tin* input_features, Tout* output) {
  CreateAndRunKernelSoftplusGradInnerError(dims, dims, dims, input_gradients,
                                           input_features, output);
}

template <typename T>
bool ReadBinFile(std::string file_name, T buf[], std::size_t size) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }
    in_file.read(reinterpret_cast<char*>(buf), size * sizeof(buf[0]));
    in_file.close();
  } catch (std::exception& e) {
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
    out_file.write(reinterpret_cast<char*>(buf), size * sizeof(buf[0]));
    out_file.close();
  } catch (std::exception& e) {
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
  } catch (std::exception& e) {
    std::cout << "write file " << file_name << " failed, " << e.what()
              << std::endl;
    return false;
  }
  return true;
}

template <typename T>
void RunTestSoftplusGrad(std::uint32_t flag) {
  const auto data_name{ToDataName<T>()};
  if (flag == 0) {
    const auto data_name{ToDataName<T>()};
    // read input gradients
    std::uint64_t dim_gradients[1];
    std::string data_dim_gradients_path{
        ktestcaseFilePath +
        "softplusgrad/data/softplusgrad_bigdata_gradients_" + data_name +
        "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_gradients_path, dim_gradients, 1), true);

    std::uint64_t shape_gradients[dim_gradients[0]];
    std::string data_shape_gradients_path{
        ktestcaseFilePath +
        "softplusgrad/data/softplusgrad_bigdata_gradients_" + data_name +
        "_shape.txt"};
    EXPECT_EQ(
        ReadFile(data_shape_gradients_path, shape_gradients, dim_gradients[0]),
        true);

    std::vector<std::int64_t> dims_gradients(
        shape_gradients, shape_gradients + dim_gradients[0]);
    auto input_gradients_size{SizeOf(dims_gradients)};

    T data_gradients[input_gradients_size];
    std::string data_gradients_path{
        ktestcaseFilePath +
        "softplusgrad/data/softplusgrad_bigdata_input_gradients_" + data_name +
        ".bin"};
    EXPECT_EQ(
        ReadBinFile(data_gradients_path, data_gradients, input_gradients_size),
        true);
    // read input features
    std::uint64_t dim_features[1];
    std::string data_dim_features_path{
        ktestcaseFilePath + "softplusgrad/data/softplusgrad_bigdata_features_" +
        data_name + "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_features_path, dim_features, 1), true);

    std::uint64_t shape_features[dim_features[0]];
    std::string data_shape_features_path{
        ktestcaseFilePath + "softplusgrad/data/softplusgrad_bigdata_features_" +
        data_name + "_shape.txt"};
    EXPECT_EQ(
        ReadFile(data_shape_features_path, shape_features, dim_features[0]),
        true);

    std::vector<std::int64_t> dims_features(shape_features,
                                            shape_features + dim_features[0]);
    auto input_features_size{SizeOf(dims_features)};
    T data_features[input_features_size];
    std::string data_features_path{
        ktestcaseFilePath +
        "softplusgrad/data/softplusgrad_bigdata_input_features_" + data_name +
        ".bin"};
    EXPECT_EQ(
        ReadBinFile(data_features_path, data_features, input_features_size),
        true);

    // read output backprops
    std::uint64_t dim_backprops[1];
    std::string data_dim_backprops_path{
        ktestcaseFilePath + "softplusgrad/data/softplusgrad_bigdata_output_" +
        data_name + "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_backprops_path, dim_backprops, 1), true);

    std::uint64_t shape_backprops[dim_backprops[0]];
    std::string data_shape_backprops_path{
        ktestcaseFilePath + "softplusgrad/data/softplusgrad_bigdata_output_" +
        data_name + "_shape.txt"};
    EXPECT_EQ(
        ReadFile(data_shape_backprops_path, shape_backprops, dim_backprops[0]),
        true);

    std::vector<std::int64_t> dims_backprops(
        shape_backprops, shape_backprops + dim_backprops[0]);
    auto input_backprops_size{SizeOf(dims_backprops)};
    T output[input_backprops_size];
    CreateAndRunKernelSoftplusGrad(dims_gradients, dims_features,
                                   dims_backprops, data_gradients,
                                   data_features, output);
    std::string out_data_actual_path{
        ktestcaseFilePath + "softplusgrad/data/softplusgrad_bigdata_output_" +
        data_name + "_actual.txt"};
    EXPECT_EQ(WriteFile(out_data_actual_path, output, input_backprops_size),
              true);

    T expect_out[input_backprops_size];
    std::string out_data_path{ktestcaseFilePath +
                              "softplusgrad/data/softplusgrad_bigdata_output_" +
                              data_name + ".bin"};
    EXPECT_EQ(ReadBinFile(out_data_path, expect_out, input_backprops_size),
              true);
    EXPECT_EQ(CompareResult(output, expect_out, input_backprops_size), true);
  } else if (flag == 1) {
    const auto data_name{ToDataName<T>()};
    // read input gradients
    std::uint64_t dim_gradients[1];
    std::string data_dim_gradients_path{
        ktestcaseFilePath + "softplusgrad/data/softplusgrad_data_gradients_" +
        data_name + "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_gradients_path, dim_gradients, 1), true);

    std::uint64_t shape_gradients[dim_gradients[0]];
    std::string data_shape_gradients_path{
        ktestcaseFilePath + "softplusgrad/data/softplusgrad_data_gradients_" +
        data_name + "_shape.txt"};
    EXPECT_EQ(
        ReadFile(data_shape_gradients_path, shape_gradients, dim_gradients[0]),
        true);

    std::vector<std::int64_t> dims_gradients(
        shape_gradients, shape_gradients + dim_gradients[0]);
    auto input_gradients_size{SizeOf(dims_gradients)};

    T data_gradients[input_gradients_size];
    std::string data_gradients_path{
        ktestcaseFilePath +
        "softplusgrad/data/softplusgrad_data_input_gradients_" + data_name +
        ".bin"};
    EXPECT_EQ(
        ReadBinFile(data_gradients_path, data_gradients, input_gradients_size),
        true);
    // read input features
    std::uint64_t dim_features[1];
    std::string data_dim_features_path{
        ktestcaseFilePath + "softplusgrad/data/softplusgrad_data_features_" +
        data_name + "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_features_path, dim_features, 1), true);

    std::uint64_t shape_features[dim_features[0]];
    std::string data_shape_features_path{
        ktestcaseFilePath + "softplusgrad/data/softplusgrad_data_features_" +
        data_name + "_shape.txt"};
    EXPECT_EQ(
        ReadFile(data_shape_features_path, shape_features, dim_features[0]),
        true);

    std::vector<std::int64_t> dims_features(shape_features,
                                            shape_features + dim_features[0]);
    auto input_features_size{SizeOf(dims_features)};

    T data_features[input_features_size];
    std::string data_features_path{
        ktestcaseFilePath +
        "softplusgrad/data/softplusgrad_data_input_features_" + data_name +
        ".bin"};
    EXPECT_EQ(
        ReadBinFile(data_features_path, data_features, input_features_size),
        true);
    // read output backprops
    std::uint64_t dim_backprops[1];
    std::string data_dim_backprops_path{
        ktestcaseFilePath + "softplusgrad/data/softplusgrad_data_output_" +
        data_name + "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_backprops_path, dim_backprops, 1), true);

    std::uint64_t shape_backprops[dim_backprops[0]];
    std::string data_shape_backprops_path{
        ktestcaseFilePath + "softplusgrad/data/softplusgrad_data_output_" +
        data_name + "_shape.txt"};
    EXPECT_EQ(
        ReadFile(data_shape_backprops_path, shape_backprops, dim_backprops[0]),
        true);

    std::vector<std::int64_t> dims_backprops(
        shape_backprops, shape_backprops + dim_backprops[0]);
    auto input_backprops_size{SizeOf(dims_backprops)};
    T output[input_backprops_size];
    CreateAndRunKernelSoftplusGrad(dims_gradients, dims_features,
                                   dims_backprops, data_gradients,
                                   data_features, output);
    std::string out_data_actual_path{
        ktestcaseFilePath + "softplusgrad/data/softplusgrad_data_output_" +
        data_name + "_actual.txt"};
    EXPECT_EQ(WriteFile(out_data_actual_path, output, input_backprops_size),
              true);

    T expect_out[input_backprops_size];
    std::string out_data_path{ktestcaseFilePath +
                              "softplusgrad/data/softplusgrad_data_output_" +
                              data_name + ".bin"};
    EXPECT_EQ(ReadBinFile(out_data_path, expect_out, input_backprops_size),
              true);
    EXPECT_EQ(CompareResult(output, expect_out, input_backprops_size), true);
  }
}

template <typename T>
void RunTestSoftplusGradDiff(std::uint32_t flag) {
  const auto data_name{ToDataName<T>()};
  if (flag == 0) {
    const auto data_name{ToDataName<T>()};
    // read input gradients
    std::uint64_t dim_gradients[1];
    std::string data_dim_gradients_path{
        ktestcaseFilePath +
        "softplusgrad/data/softplusgrad_diff_bigdata_gradients_" + data_name +
        "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_gradients_path, dim_gradients, 1), true);

    std::uint64_t shape_gradients[dim_gradients[0]];
    std::string data_shape_gradients_path{
        ktestcaseFilePath +
        "softplusgrad/data/softplusgrad_diff_bigdata_gradients_" + data_name +
        "_shape.txt"};
    EXPECT_EQ(
        ReadFile(data_shape_gradients_path, shape_gradients, dim_gradients[0]),
        true);

    std::vector<std::int64_t> dims_gradients(
        shape_gradients, shape_gradients + dim_gradients[0]);
    auto input_gradients_size{SizeOf(dims_gradients)};

    T data_gradients[input_gradients_size];
    std::string data_gradients_path{
        ktestcaseFilePath +
        "softplusgrad/data/softplusgrad_diff_bigdata_input_gradients_" +
        data_name + ".bin"};
    EXPECT_EQ(
        ReadBinFile(data_gradients_path, data_gradients, input_gradients_size),
        true);
    // read input features
    std::uint64_t dim_features[1];
    std::string data_dim_features_path{
        ktestcaseFilePath +
        "softplusgrad/data/softplusgrad_diff_bigdata_features_" + data_name +
        "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_features_path, dim_features, 1), true);

    std::uint64_t shape_features[dim_features[0]];
    std::string data_shape_features_path{
        ktestcaseFilePath +
        "softplusgrad/data/softplusgrad_diff_bigdata_features_" + data_name +
        "_shape.txt"};
    EXPECT_EQ(
        ReadFile(data_shape_features_path, shape_features, dim_features[0]),
        true);

    std::vector<std::int64_t> dims_features(shape_features,
                                            shape_features + dim_features[0]);
    auto input_features_size{SizeOf(dims_features)};
    T data_features[input_features_size];
    std::string data_features_path{
        ktestcaseFilePath +
        "softplusgrad/data/softplusgrad_diff_bigdata_input_features_" +
        data_name + ".bin"};
    EXPECT_EQ(
        ReadBinFile(data_features_path, data_features, input_features_size),
        true);

    // read output backprops
    std::uint64_t dim_backprops[1];
    std::string data_dim_backprops_path{
        ktestcaseFilePath +
        "softplusgrad/data/softplusgrad_diff_bigdata_output_" + data_name +
        "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_backprops_path, dim_backprops, 1), true);

    std::uint64_t shape_backprops[dim_backprops[0]];
    std::string data_shape_backprops_path{
        ktestcaseFilePath +
        "softplusgrad/data/softplusgrad_diff_bigdata_output_" + data_name +
        "_shape.txt"};
    EXPECT_EQ(
        ReadFile(data_shape_backprops_path, shape_backprops, dim_backprops[0]),
        true);

    std::vector<std::int64_t> dims_backprops(
        shape_backprops, shape_backprops + dim_backprops[0]);
    auto input_backprops_size{SizeOf(dims_backprops)};
    T output[input_backprops_size];
    CreateAndRunKernelSoftplusGrad(dims_gradients, dims_features,
                                   dims_backprops, data_gradients,
                                   data_features, output);
    std::string out_data_actual_path{
        ktestcaseFilePath +
        "softplusgrad/data/softplusgrad_diff_bigdata_output_" + data_name +
        "_actual.txt"};
    EXPECT_EQ(WriteFile(out_data_actual_path, output, input_backprops_size),
              true);

    T expect_out[input_backprops_size];
    std::string out_data_path{
        ktestcaseFilePath +
        "softplusgrad/data/softplusgrad_diff_bigdata_output_" + data_name +
        ".bin"};
    EXPECT_EQ(ReadBinFile(out_data_path, expect_out, input_backprops_size),
              true);
    EXPECT_EQ(CompareResult(output, expect_out, input_backprops_size), true);
  } else if (flag == 1) {
    const auto data_name{ToDataName<T>()};
    // read input gradients
    std::uint64_t dim_gradients[1];
    std::string data_dim_gradients_path{
        ktestcaseFilePath +
        "softplusgrad/data/softplusgrad_diff_data_gradients_" + data_name +
        "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_gradients_path, dim_gradients, 1), true);

    std::uint64_t shape_gradients[dim_gradients[0]];
    std::string data_shape_gradients_path{
        ktestcaseFilePath +
        "softplusgrad/data/softplusgrad_diff_data_gradients_" + data_name +
        "_shape.txt"};
    EXPECT_EQ(
        ReadFile(data_shape_gradients_path, shape_gradients, dim_gradients[0]),
        true);

    std::vector<std::int64_t> dims_gradients(
        shape_gradients, shape_gradients + dim_gradients[0]);
    auto input_gradients_size{SizeOf(dims_gradients)};

    T data_gradients[input_gradients_size];
    std::string data_gradients_path{
        ktestcaseFilePath +
        "softplusgrad/data/softplusgrad_diff_data_input_gradients_" +
        data_name + ".bin"};
    EXPECT_EQ(
        ReadBinFile(data_gradients_path, data_gradients, input_gradients_size),
        true);
    // read input features
    std::uint64_t dim_features[1];
    std::string data_dim_features_path{
        ktestcaseFilePath +
        "softplusgrad/data/softplusgrad_diff_data_features_" + data_name +
        "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_features_path, dim_features, 1), true);

    std::uint64_t shape_features[dim_features[0]];
    std::string data_shape_features_path{
        ktestcaseFilePath +
        "softplusgrad/data/softplusgrad_diff_data_features_" + data_name +
        "_shape.txt"};
    EXPECT_EQ(
        ReadFile(data_shape_features_path, shape_features, dim_features[0]),
        true);

    std::vector<std::int64_t> dims_features(shape_features,
                                            shape_features + dim_features[0]);
    auto input_features_size{SizeOf(dims_features)};

    T data_features[input_features_size];
    std::string data_features_path{
        ktestcaseFilePath +
        "softplusgrad/data/softplusgrad_diff_data_input_features_" + data_name +
        ".bin"};
    EXPECT_EQ(
        ReadBinFile(data_features_path, data_features, input_features_size),
        true);
    // read output backprops
    std::uint64_t dim_backprops[1];
    std::string data_dim_backprops_path{
        ktestcaseFilePath + "softplusgrad/data/softplusgrad_diff_data_output_" +
        data_name + "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_backprops_path, dim_backprops, 1), true);

    std::uint64_t shape_backprops[dim_backprops[0]];
    std::string data_shape_backprops_path{
        ktestcaseFilePath + "softplusgrad/data/softplusgrad_diff_data_output_" +
        data_name + "_shape.txt"};
    EXPECT_EQ(
        ReadFile(data_shape_backprops_path, shape_backprops, dim_backprops[0]),
        true);

    std::vector<std::int64_t> dims_backprops(
        shape_backprops, shape_backprops + dim_backprops[0]);
    auto input_backprops_size{SizeOf(dims_backprops)};
    T output[input_backprops_size];
    CreateAndRunKernelSoftplusGrad(dims_gradients, dims_features,
                                   dims_backprops, data_gradients,
                                   data_features, output);
    std::string out_data_actual_path{
        ktestcaseFilePath + "softplusgrad/data/softplusgrad_diff_data_output_" +
        data_name + "_actual.txt"};
    EXPECT_EQ(WriteFile(out_data_actual_path, output, input_backprops_size),
              true);

    T expect_out[input_backprops_size];
    std::string out_data_path{
        ktestcaseFilePath + "softplusgrad/data/softplusgrad_diff_data_output_" +
        data_name + ".bin"};
    EXPECT_EQ(ReadBinFile(out_data_path, expect_out, input_backprops_size),
              true);
    EXPECT_EQ(CompareResult(output, expect_out, input_backprops_size), true);
  }
}

#define ADD_SAME_BIGDATA_CASE(base_type, aicpu_type)                  \
  TEST_F(TEST_SOFTPLUSGRAD_UT, DATA_TYPE_SAME_BIGDATA_##aicpu_type) { \
    RunTestSoftplusGrad<base_type>(0);                                \
  }
#define ADD_SAME_CASE(base_type, aicpu_type)                          \
  TEST_F(TEST_SOFTPLUSGRAD_UT, DATA_TYPE_SAME_##aicpu_type) {         \
    RunTestSoftplusGrad<base_type>(1);                                \
  }
#define ADD_DIFF_BIGDATA_CASE(base_type, aicpu_type)                  \
  TEST_F(TEST_SOFTPLUSGRAD_UT, DATA_TYPE_DIFF_BIGDATA_##aicpu_type) { \
    RunTestSoftplusGradDiff<base_type>(0);                            \
  }

#define ADD_DIFF_CASE(base_type, aicpu_type)                          \
  TEST_F(TEST_SOFTPLUSGRAD_UT, DATA_TYPE_DIFF_##aicpu_type) {         \
    RunTestSoftplusGradDiff<base_type>(1);                            \
  }

// exception instance
TEST_F(TEST_SOFTPLUSGRAD_UT, BAD_KERNEL_EXCEPTION) {
  CreateAndRunKernelSoftplusGrad(
      {2, 8}, {2, 8}, {2, 8}, float_16_, float_16_, float_16_,
      aicpu::KernelStatus::KERNEL_STATUS_INNER_ERROR, true);
}

TEST_F(TEST_SOFTPLUSGRAD_UT, INPUT0_DIM_EXCEPTION) {
  CreateAndRunKernelSoftplusGradParamInvalid({2, 4, 2}, {2, 8}, {2, 8},
                                             float_16_, float_16_, float_16_);
}

TEST_F(TEST_SOFTPLUSGRAD_UT, INPUT1_SHAPE_EXCEPTION) {
  CreateAndRunKernelSoftplusGradParamInvalid({2, 8}, {2, 6}, {2, 8}, float_16_,
                                             float_12_, float_16_);
}

TEST_F(TEST_SOFTPLUSGRAD_UT, OUTPUT_SHAPE_EXCEPTION) {
  CreateAndRunKernelSoftplusGradParamInvalid({2, 8}, {2, 8}, {2, 6}, float_16_,
                                             float_16_, float_12_);
}

TEST_F(TEST_SOFTPLUSGRAD_UT, INPUT1_DTYPE_EXCEPTION) {
  CreateAndRunKernelSoftplusGradParamInvalid({2, 8}, float_16_, float_16_,
                                             double_16_);
}

TEST_F(TEST_SOFTPLUSGRAD_UT, INPUT_NULL_EXCEPTION) {
  CreateAndRunKernelSoftplusGradParamInvalid({2, 11}, float_null_, float_null_,
                                             float_null_);
}

TEST_F(TEST_SOFTPLUSGRAD_UT, OUTPUT_DATA_NULL_EXCEPTION) {
  CreateAndRunKernelSoftplusGradParamInvalid({0, 0}, float_0_, float_0_,
                                             float_null_);
}

TEST_F(TEST_SOFTPLUSGRAD_UT, INPUT1_DATA_NULL_EXCEPTION) {
  CreateAndRunKernelSoftplusGradParamInvalid({0, 0}, float_0_, float_null_,
                                             float_0_);
}

TEST_F(TEST_SOFTPLUSGRAD_UT, INPUT0_DATA_NULL_EXCEPTION) {
  CreateAndRunKernelSoftplusGradParamInvalid({0, 0}, float_null_, float_0_,
                                             float_0_);
}
TEST_F(TEST_SOFTPLUSGRAD_UT, INPUT_EMPTY_EXCEPTION) {
  CreateAndRunKernelSoftplusGrad({0, 0}, float_0_, float_0_, float_0_);
}
TEST_F(TEST_SOFTPLUSGRAD_UT, NO_OUTPUT_EXCEPTION) {
  const auto data_type_in{ToDataType<std::float_t>()};
  auto node_def{aicpu::CpuKernelUtils::CreateNodeDef()};
  aicpu::NodeDefBuilder(node_def.get(), "SoftplusGrad", "SoftplusGrad")
      .Input({"gradients", data_type_in, {2, 6}, float_12_})
      .Input({"features", data_type_in, {2, 6}, float_12_});
  RunKernelSoftplusGrad(node_def, aicpu::DeviceType::HOST,
                        aicpu::KernelStatus::KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SOFTPLUSGRAD_UT, INPUT_BOOL_UNSUPPORT) {
  CreateAndRunKernelSoftplusGradParamInvalid({2, 11}, bool_22_, bool_22_,
                                             bool_22_);
}

ADD_SAME_CASE(Eigen::half, DT_FLOAT16)
ADD_SAME_CASE(std::float_t, DT_FLOAT)
ADD_SAME_CASE(std::double_t, DT_DOUBLE)

ADD_SAME_BIGDATA_CASE(Eigen::half, DT_FLOAT16)
ADD_SAME_BIGDATA_CASE(std::float_t, DT_FLOAT)
ADD_SAME_BIGDATA_CASE(std::double_t, DT_DOUBLE)
