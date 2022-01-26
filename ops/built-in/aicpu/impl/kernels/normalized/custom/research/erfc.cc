/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "erfc.h"

#include <cmath>
#include <typeinfo>
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const int64_t paralled_data_size = 1024 * 64;
const char *kErfc = "Erfc";
#define ERFC_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                      \
    uint32_t result = ErfcCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                  \
      KERNEL_LOG_ERROR("Erfc kernel compute failed."); \
      return result;                                   \
    }                                                  \
    break;                                             \
  }
#define ERFC_COMPUTE_CASE2(DTYPE, TYPE, CTX)           \
  case (DTYPE): {                                      \
    uint32_t result = ErfcCompute2<TYPE>(CTX);         \
    if (result != KERNEL_STATUS_OK) {                  \
      KERNEL_LOG_ERROR("Erfc kernel compute failed."); \
      return result;                                   \
    }                                                  \
    break;                                             \
  }

}  // namespace

namespace internel {
template <typename T, size_t N>
inline T EvaluatePolynomial(T x, const std::array<T, N> &coeffs) {
  T result = 0;
  for (T c : coeffs) {
    result = result * x + c;
  }
  return result;
}
static double ErfImpl64(double x) {
  // Coefficients for by erf(f64), from Cephes.
  // erf(x) = x T(x^2) / U(x^2), 0 < x < 1
  static std::array<double, 5> kErfTCoefficient{
      9.60497373987051638749E0, 9.00260197203842689217E1,
      2.23200534594684319226E3, 7.00332514112805075473E3,
      5.55923013010394962768E4};
  static std::array<double, 6> kErfUCoefficient{
      1.00000000000000000000E0, 3.35617141647503099647E1,
      5.21357949780152679795E2, 4.59432382970980127987E3,
      2.26290000613890934246E4, 4.92673942608635921086E4};
  double squared_x = x * x;
  return x * EvaluatePolynomial<double>(squared_x, kErfTCoefficient) /
         EvaluatePolynomial<double>(squared_x, kErfUCoefficient);
}
static double ErfcImpl64(double x) {
  // Coefficients for erfc(f64), from Cephes.
  const double kMaxlog = 7.09782712893383996843E2;
  // erfc(x) = exp(-x^2) P(|x|) / Q(|x|), 1 < x < 8
  static const std::array<double, 9> kErfcPCoefficient{
      2.46196981473530512524E-10, 5.64189564831068821977E-1,
      7.46321056442269912687E0,   4.86371970985681366614E1,
      1.96520832956077098242E2,   5.26445194995477358631E2,
      9.34528527171957607540E2,   1.02755188689515710272E3,
      5.57535335369399327526E2};
  static const std::array<double, 9> kErfcQCoefficient{
      1.00000000000000000000E0, 1.32281951154744992508E1,
      8.67072140885989742329E1, 3.54937778887819891062E2,
      9.75708501743205489753E2, 1.82390916687909736289E3,
      2.24633760818710981792E3, 1.65666309194161350182E3,
      5.57535340817727675546E2};
  // erfc(x) = exp(-x^2) R(|x|) / S(|x|), 8 <= x < kMaxlog
  static const std::array<double, 6> kErfcRCoefficient{
      5.64189583547755073984E-1, 1.27536670759978104416E0,
      5.01905042251180477414E0,  6.16021097993053585195E0,
      7.40974269950448939160E0,  2.97886665372100240670E0};
  static const std::array<double, 7> kErfcSCoefficient{
      1.00000000000000000000E0, 2.26052863220117276590E0,
      9.39603524938001434673E0, 1.20489539808096656605E1,
      1.70814450747565897222E1, 9.60896809063285878198E0,
      3.36907645100081516050E0};
  const auto constzero = static_cast<double>(0.0);
  const auto consttwo = static_cast<double>(2.0);
  const auto consteight = static_cast<double>(8.0);
  double minus_squared_x = -x * x;
  double abs_x = abs(x);
  double y = (abs_x < consteight)
                 ? (std::exp(minus_squared_x) *
                    EvaluatePolynomial<double>(abs_x, kErfcPCoefficient) /
                    EvaluatePolynomial<double>(abs_x, kErfcQCoefficient))
                 : (std::exp(minus_squared_x) *
                    EvaluatePolynomial<double>(abs_x, kErfcRCoefficient) /
                    EvaluatePolynomial<double>(abs_x, kErfcSCoefficient));
  double y_clamp = (minus_squared_x < -kMaxlog) ? constzero : y;
  return (x < constzero) ? (consttwo - y_clamp) : y_clamp;
}
static float ErfImpl32Cephes(float x) {
  // Coefficients for by erf(f32), from Cephes.
  // erf(x) = x P(x^2), 0 < x < 1
  static const std::array<float, 7> kErfTCoefficient{
      +7.853861353153693E-5, -8.010193625184903E-4, +5.188327685732524E-3,
      -2.685381193529856E-2, +1.128358514861418E-1, -3.761262582423300E-1,
      +1.128379165726710E+0,
  };
  return x * EvaluatePolynomial<float>(x * x, kErfTCoefficient);
}
static float ErfcImpl32(float x) {
  // Coefficients for erfc(f32), from Cephes.
  const double kMaxlog = 88.72283905206835;
  // erfc(x) = exp(-x^2) P(1/x^2), 1 < x < 2
  static const std::array<float, 9> kErfcPCoefficient{
      +2.326819970068386E-2, -1.387039388740657E-1, +3.687424674597105E-1,
      -5.824733027278666E-1, +6.210004621745983E-1, -4.944515323274145E-1,
      +3.404879937665872E-1, -2.741127028184656E-1, +5.638259427386472E-1,
  };
  // erfc(x) = exp(-x^2) R(1/x^2), 2 <= x < kMaxlog
  static const std::array<float, 8> kErfcRCoefficient{
      -1.047766399936249E+1, +1.297719955372516E+1, -7.495518717768503E+0,
      +2.921019019210786E+0, -1.015265279202700E+0, +4.218463358204948E-1,
      -2.820767439740514E-1, +5.641895067754075E-1,
  };
  const auto constone = static_cast<float>(1.0);
  const auto constzero = static_cast<float>(0.0);
  const auto consttwo = static_cast<float>(2.0);
  float abs_x = abs(x);
  float exp_squared_x = std::exp(-x * x);
  float q = constone / abs_x;
  float y = q * q;
  float p = (abs_x < consttwo)
                ? EvaluatePolynomial<float>(y, kErfcPCoefficient)
                : EvaluatePolynomial<float>(y, kErfcRCoefficient);
  y = exp_squared_x * q * p;
  float y_clamp = (exp_squared_x < -float(kMaxlog)) ? constzero : y;
  return (x < constzero) ? (consttwo - y_clamp) : y_clamp;
}
template <typename T>
inline T Erfccal(T x) {
  const auto constone = static_cast<T>(1.0);
  T abs_x = abs(x);
  if (typeid(x) == typeid(double)) {
    return (abs_x > constone) ? ErfcImpl64(x) : (constone - ErfImpl64(x));
  } else if (typeid(x) == typeid(float)) {
    return (abs_x > constone) ? ErfcImpl32(x) : (constone - ErfImpl32Cephes(x));
  }
}
template <typename T>
inline T Erfccalhalf(T x) {
  const auto constone = static_cast<Eigen::half>(1.0);
  const auto constone_half = static_cast<float>(1.0);
  Eigen::half abs_x = abs(x);
  auto to_float_x = static_cast<float>(x);
  float y = (abs_x > constone) ? ErfcImpl32(to_float_x)
                               : (constone_half - ErfImpl32Cephes(to_float_x));
  Eigen::half expect = static_cast<Eigen::half>(y);
  return expect;
}
}  // namespace internel

namespace aicpu {
uint32_t ErfcCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "[%s] check input and output failed.", kErfc);
  KERNEL_HANDLE_ERROR(ErfcCheck(ctx), "[%s] check params failed.", kErfc);
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    ERFC_COMPUTE_CASE2(DT_FLOAT16, Eigen::half, ctx)
    ERFC_COMPUTE_CASE(DT_FLOAT, float, ctx)
    ERFC_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("Erfc kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
uint32_t ErfcCpuKernel::ErfcCheck(CpuKernelContext &ctx) {
  KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input data failed.")
  KERNEL_CHECK_NULLPTR(ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID,
                       "Get output data failed.")
  std::vector<int64_t> shape_input =
      ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_output =
      ctx.Output(0)->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((shape_input.size() != 0), KERNEL_STATUS_PARAM_INVALID,
                     "Input must be at least rank 1, got [%zu].",
                     shape_input.size())
  KERNEL_CHECK_FALSE(
      (shape_input.size() == shape_output.size()), KERNEL_STATUS_PARAM_INVALID,
      "The output shape size should be same as the output shape size")
  return KERNEL_STATUS_OK;
}
template <typename T>
uint32_t ErfcCpuKernel::ErfcCompute(CpuKernelContext &ctx) {
  auto input_data = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  const auto Erfccal = internel::Erfccal<T>;
  int64_t data_num = ctx.Input(0)->NumElements();
  int64_t data_size = data_num * sizeof(T);
  if (data_size <= paralled_data_size) {
    std::transform(input_data, input_data + data_num, output_data, Erfccal);
  } else {
    auto shard_erfc = [&](size_t start, size_t end) {
      std::transform(input_data + start, input_data + end, output_data + start,
                     Erfccal);
    };
    uint32_t min_core_num = 1;
    uint32_t max_core_num =
        std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    max_core_num = static_cast<int64_t>(max_core_num);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, 
                                                    data_num / max_core_num, 
                                                    shard_erfc),
                        "Erfc Compute failed.")
  }
  return KERNEL_STATUS_OK;
}
template <typename T>
uint32_t ErfcCpuKernel::ErfcCompute2(CpuKernelContext &ctx) {
  auto input_data = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  const auto Erfccalhalf = internel::Erfccalhalf<T>;
  int64_t data_num = ctx.Input(0)->NumElements();
  int64_t data_size = data_num * sizeof(T);
  if (data_size <= paralled_data_size / 4) {
    std::transform(input_data, input_data + data_num, output_data, Erfccalhalf);
  } else {
    auto shard_erfc = [&](size_t start, size_t end) {
      std::transform(input_data + start, input_data + end, output_data + start,
                     Erfccalhalf);
    };
    uint32_t min_core_num = 1;
    uint32_t max_core_num =
        std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    max_core_num = static_cast<int64_t>(max_core_num);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, 
                                                    data_num / max_core_num, 
                                                    shard_erfc),
                        "Erfc Compute failed.")
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kErfc, ErfcCpuKernel);
}  // namespace aicpu