/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
#include "stateless_drop_out_gen_mask.h"

#include <cfloat>

#include "cpu_kernel_utils.h"
#include "securec.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kStatelessDropOutGenMask = "StatelessDropOutGenMask";
const int32_t kInputProbIndex = 1;
const int32_t kInputSeedIndex = 2;
const int32_t kInputSeed1Index = 3;
#if ((defined __ARM_ARCH) || (defined PLANTFORM_AARCH64)) && \
    (!defined RUN_ANDROID)
const int32_t kTransfer128Bit = 7;
const int32_t kSkipBytesIndex = 4;
#endif
}  // namespace

namespace aicpu {
#if ((defined __ARM_ARCH) || (defined PLANTFORM_AARCH64)) && \
    (!defined RUN_ANDROID)
#define CONFIG_ENABLE_PERIOD_64BIT

void StatelessDropOutGenMaskCpuKernel::StatelessDropOutGenMaskKernel(
    const uint64_t count, const uint8_t *offset, const uint8_t *key,
    uint8_t *out) {
  const uint16_t threshold = static_cast<uint16_t>(UINT16_MAX * prob_);
  const uint8_t in_offset[16] = {0x01, 0, 0, 0, 0, 0, 0, 0, 0x01};
  const uint8_t inc_step[16] = {0x02};

  // a const key. reference paper: https://dl.acm.org/citaion.cfm?id=206340
  const uint8_t key_const[16] = {0xBB, 0x67, 0xAE, 0x85, 0x84, 0xCA,
                                 0xA7, 0x3B, 0x9E, 0X37, 0x79, 0xB9,
                                 0X7F, 0X4A, 0x7c, 0x15};

  const uint8_t *key_const_ptr = &(key_const[0]);
  const uint8_t *inc_step_ptr = &(inc_step[0]);

  // Each iteration generates 4-bit * 8 elements (in vector reg) * 4 (repeated
  // code blocks)
  const uint64_t loop_time = count / 4 / 8 / 4;
  __asm volatile(
      ".arch armv8-a+crypto \n"
      "ldr x0, %[loop_time] \n"

      "ldr x16, %[key_const_ptr]\n"
      "ld1 {v2.16b}, [x16] \n"

      // generate in1
      "ldr x1, %[offset] \n"
      "ld1 {v0.16b}, [x1] \n"  // tmp input

      "ldr x2, %[key] \n"
      "ld1 {v1.16b}, [x2] \n"       // first round key
      "add v5.2d, v1.2d, v2.2d \n"  // second round key

      // generate in2
      "ldp x10, x11, %[in_offset] \n"
      "ldp x12, x13, [x1] \n"
      "adds x14, x12, x10 \n"
      "adc x15, x13, x11 \n"
      "mov v10.d[0], x14 \n"
      "mov v10.d[1], x15 \n"

      // generate in3 = in1
      "mov v3.16b, v0.16b \n"
      // generate in4 = in2
      "mov v13.16b, v10.16b \n"

  // load input inc step
#ifdef CONFIG_ENABLE_PERIOD_64BIT
      "ldr x17, %[inc_step_ptr] \n"
      "ld1 {v4.16b}, [x17] \n"
#else
      "ldp x10, x11, %[inc_step] \n"
#endif

      "ldr w7, %[threshold] \n"
      "dup v20.8h, w7 \n"

      // Generate 16 bitmasks to 16 regs
      "mov w7, #0x8000 \n"
      "dup v21.8h, w7 \n"
      "mov w7, #0x4000 \n"
      "dup v12.8h, w7 \n"
      "mov w7, #0x2000 \n"
      "dup v2.8h, w7 \n"
      "mov w7, #0x1000 \n"
      "dup v6.8h, w7 \n"

      "mov w7, #0x800 \n"
      "dup v7.8h, w7 \n"
      "mov w7, #0x400 \n"
      "dup v8.8h, w7 \n"
      "mov w7, #0x200 \n"
      "dup v14.8h, w7 \n"
      "mov w7, #0x100 \n"
      "dup v15.8h, w7 \n"

      "mov w7, #0x80 \n"
      "dup v26.8h, w7 \n"
      "mov w7, #0x40 \n"
      "dup v27.8h, w7 \n"
      "mov w7, #0x20 \n"
      "dup v9.8h, w7 \n"
      "mov w7, #0x10 \n"
      "dup v11.8h, w7 \n"

      "mov w7, #0x8 \n"
      "dup v16.8h, w7 \n"
      "mov w7, #0x4 \n"
      "dup v17.8h, w7 \n"
      "mov w7, #0x2 \n"
      "dup v18.8h, w7 \n"
      "mov w7, #0x1 \n"
      "dup v19.8h, w7 \n"

      // load out pointer addr to register
      "ldr x5, %[out] \n"

      // Iteration begins
    "1: \n" 

      /* Mix v0 with v1 */
      "aese   v0.16b, v1.16b \n"
      "aesmc  v0.16b, v0.16b \n"

      /* Mix v10 with v5 */
      "aese   v10.16b, v5.16b \n"
      "aesmc  v10.16b, v10.16b \n"

      /* Compare the random number v0 against threshold */
      "cmhs v22.8h, v20.8h, v0.8h \n"
      /* Update the output register with v0 */
      "bit v29.16b, v22.16b, v21.16b \n"

      /* Mix v13 with v1 */
      "aese   v13.16b, v1.16b \n"
      "aesmc  v13.16b, v13.16b \n"

      /* Compare the random number v10 against threshold */
      "cmhs v23.8h, v20.8h, v10.8h \n"
      /* Update the output register with v10 */
      "bit v29.16b, v23.16b, v12.16b \n"

      /* Mix v3 with v5 */
      "aese   v3.16b, v5.16b \n"
      "aesmc  v3.16b, v3.16b \n"

      /* Compare the random number v13 against threshold */
      "cmhs v25.8h, v20.8h, v13.8h \n"
      /* Update the output register with v13 */
      "bit v29.16b, v25.16b, v6.16b \n"

      /* mix v0 with v1 */
      "aese   v0.16b, v1.16b \n"
      "aesmc  v0.16b, v0.16b \n"

      /* Compare the randowm number v3 against threshold */
      "cmhs v24.8h, v20.8h, v3.8h \n"
      /* Update the output register with v3 */
      "bit v29.16b, v24.16b, v2.16b \n"

      /* Mix v10 with v5 */
      "aese   v10.16b, v5.16b \n"
      "aesmc  v10.16b, v10.16b \n"

      /* Compare the random number v0 aganist threshold */
      "cmhs v22.8h, v20.8h, v0.8h \n"
      /* Update the output register with v0 */
      "bit v29.16b, v22.16b, v7.16b \n"

      /* Mix v13 with v1 */
      "aese   v13.16b, v1.16b \n"
      "aesmc  v13.16b, v13.16b \n"

      /* Compare the random number v10 aganist threshold */
      "cmhs v23.8h, v20.8h, v10.8h \n"
      /* update the output register with v10 */
      "bit v29.16b, v23.16b, v8.16b \n"

      /* Mix v3 with v5 */
      "aese   v3.16b, v5.16b \n"
      "aesmc  v3.16b, v3.16b \n"

      /* Compare the random number v13 aganist threshold */
      "cmhs v25.8h, v20.8h, v13.8h \n"
      /* Update the output register with v13 */
      "bit v29.16b, v25.16b, v14.16b \n"

      /* Mix v0 with v1 */
      "aese   v0.16b, v1.16b \n"
      "aesmc  v0.16b, v0.16b \n"

      /* Compare the random number v3 against threshold */
      "cmhs v24.8h, v20.8h, v3.8h \n"
      /* Update the output register with v3 */
      "bit v29.16b, v24.16b, v15.16b \n"

      /* Mix v10 with v5 */
      "aese   v10.16b, v5.16b \n"
      "aesmc  v10.16b, v10.16b \n"

      /* Compare the random number v0 against threshold */
      "cmhs v22.8h, v20.8h, v0.8h \n"
      /* Update the output register with v0 */
      "bit v29.16b, v22.16b, v26.16b \n"

      /* Mix v13 with v1 */
      "aese   v13.16b, v1.16b \n"
      "aesmc  v13.16b, v13.16b \n"

      /* Compare the random number v10 against threshold */
      "cmhs v23.8h, v20.8h, v10.8h \n"
      /* Update the output register with v10 */
      "bit v29.16b, v23.16b, v27.16b \n"

      /* Mix v3 with v5 */
      "aese   v3.16b, v5.16b \n"
      "aesmc  v3.16b, v3.16b \n"

      /* Compare the random number v13 against threshold */
      "cmhs v25.8h, v20.8h, v13.8h \n"
      /* Update the output register with v13 */
      "bit v29.16b, v25.16b, v9.16b \n"

      /* Mix v0 with v1 */
      "aese   v0.16b, v1.16b \n"
      "aesmc  v0.16b, v0.16b \n"

      /* Compare the random number v3 against threshold */
      "cmhs v24.8h, v20.8h, v3.8h \n"
      /* update the output register with v3 */
      "bit v29.16b, v24.16b, v11.16b\n"

      /* Mix v10 with v5 */
      "aese   v10.16b, v5.16b \n"
      "aesmc  v10.16b, v10.16b \n"

      /* Compare the random number v0 against threshold */
      "cmhs v22.8h, v20.8h, v0.8h \n"
      /* Update the output register with v0 */
      "bit v29.16b, v22.16b, v16.16b \n"

      /* Mix v13 with v1 */
      "aese   v13.16b, v1.16b \n"

      "aesmc  v13.16b, v13.16b \n"

      /* Compare the random number v10 against threshold */
      "cmhs v23.8h, v20.8h, v10.8h \n"
      /* Update the output register with v10 */
      "bit v29.16b, v23.16b, v17.16b \n"

      /* Mix v3 with v5 */
      "aese   v3.16b, v5.16b \n"
      "aesmc  v3.16b, v3.16b \n"

      /* Compare the random number v13 against threshold */
      "cmhs v25.8h, v20.8h, v13.8h \n"
      /* Update the output register with v13 */
      "bit v29.16b, v25.16b, v18.16b \n"

  // Update the key
#ifdef CONFIG_ENABLE_PERIOD_64BIT
      "add v1.2d, v1.2d, v4.2d \n"
      "add v5.2d, v5.2d, v4.2d \n"
#else
      "mov x12, v1.d[0] \n"
      "mov x13, v1.d[1] \n"
      "adds x14, x12, x10 \n"
      "adc x15, x13, x11 \n"
      "mov v1.d[0], x14 \n"
      "mov v1.d[1], x15 \n"

      "mov x12, v5.d[0] \n"
      "mov x13, v5.d[1] \n"
      "adds x14, x12, x10 \n"
      "adc x15, x13, x11 \n"
      "mov v5.d[0], x14 \n"
      "mov v5.d[1], x15 \n"
#endif

      /* Compare the random number v3 against threshold */
      "cmhs v24.8h, v20.8h, v3.8h \n"
      /* update the output register with v3 */
      "bit v29.16b, v24.16b, v19.16b \n"

      // Store the output register to memory
      "st1 {v29.16b}, [x5] \n"
      "add x5, x5, 16 \n"

      // next iteration
      "subs   x0, x0, 1 \n"
      "bne   1b \n"
      :
      : [offset] "m"(offset), [out] "m"(out), [in_offset] "m"(in_offset),
        [key] "m"(key), [key_const] "m"(key_const), [inc_step] "m"(inc_step),
        [loop_time] "m"(loop_time), [threshold] "m"(threshold),
        [key_const_ptr] "m"(key_const_ptr), [inc_step_ptr] "m"(inc_step_ptr)
      : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "w7", "x7", "x10", "x11",
        "x12", "x13", "x14", "x15", "x16", "x17", "x18", "v0", "v1", "v2", "v3",
        "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
        "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
        "v25", "v26", "v27", "v29");
}

#else  // compiled on x86 arch
void StatelessDropOutGenMaskCpuKernel::StatelessDropOutGenMaskKernel(
    const uint64_t count, const uint8_t *offset, const uint8_t *key,
    uint8_t *out) {
  std::default_random_engine e(*key);
  std::bernoulli_distribution b(prob_);
  const uint8_t mask[8] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
  for (uint64_t i = 0; i < count; i++) {
    out[i] = 0x0;
    for (const auto &m : mask) {
      if (b(e)) {
        out[i] = out[i] | m;
      }
    }
  }
  return;
}
#endif

uint32_t StatelessDropOutGenMaskCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *probTensor = ctx.Input(kInputProbIndex);
  Tensor *seedTensor = ctx.Input(kInputSeedIndex);
  Tensor *seed1Tensor = ctx.Input(kInputSeed1Index);
  Tensor *yTensor = ctx.Output(0);
  auto probDataType = probTensor->GetDataType();
  auto seedDataType = seedTensor->GetDataType();
  auto seed1DataType = seed1Tensor->GetDataType();

  switch (probDataType) {
    case DT_FLOAT16:
      prob_ = static_cast<float>(
          *(static_cast<Eigen::half *>(probTensor->GetData())));
      break;
    case DT_FLOAT:
      prob_ = *(static_cast<float *>(probTensor->GetData()));
      break;
    default:
      KERNEL_LOG_ERROR("Data type not support[%s].",
                       DTypeStr(probDataType).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  switch (seedDataType) {
    case DT_INT32:
      seed_ = *(static_cast<int32_t *>(seedTensor->GetData()));
      break;
    case DT_INT64:
      seed_ = *(static_cast<int64_t *>(seedTensor->GetData()));
      break;
    default:
      KERNEL_LOG_ERROR("Data type not support[%s].",
                       DTypeStr(seedDataType).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  switch (seed1DataType) {
    case DT_INT32:
      seed1_ = *(static_cast<int32_t *>(seed1Tensor->GetData()));
      break;
    case DT_INT64:
      seed1_ = *(static_cast<int64_t *>(seed1Tensor->GetData()));
      break;
    default:
      KERNEL_LOG_ERROR("Data type not support[%s].",
                       DTypeStr(seed1DataType).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  uint8_t *out_buff = static_cast<uint8_t *>(yTensor->GetData());
  uint64_t outputSize = yTensor->GetDataSize();

  KERNEL_LOG_INFO("prob[%f] seed[%ld] seed1[%ld] outputSize[%lu].", prob_,
                  seed_, seed1_, outputSize);

  return DoCompute(ctx, out_buff, outputSize);
}

uint32_t StatelessDropOutGenMaskCpuKernel::DoCompute(CpuKernelContext &ctx,
                                                     uint8_t *out_buff,
                                                     uint64_t outputSize) {
  if (prob_ <= FLT_EPSILON) {
    (void)memset_s(out_buff, outputSize, 0x00, outputSize);
    return KERNEL_STATUS_OK;
  }
  // if prob_ is 1, set all bits to 1
  if (fabs(prob_ - 1.0f) <= FLT_EPSILON) {
    (void)memset_s(out_buff, outputSize, 0xff, outputSize);
    return KERNEL_STATUS_OK;
  }

  uint64_t key[2] = {static_cast<uint64_t>(seed1_),
                     static_cast<uint64_t>(seed_)};
  uint64_t offset[2] = {0, 0};
#if ((defined __ARM_ARCH) || (defined PLANTFORM_AARCH64)) && \
    (!defined RUN_ANDROID)
  KERNEL_LOG_INFO("DropOutGenMaskCpuKernel::PLATFORM_AARCH64.");
  auto shards = [&](int64_t start, int64_t limit) {
    // transfer 128bits to bit
    uint64_t count = (static_cast<uint64_t>(limit - start)) << kTransfer128Bit;
    uint8_t *out_ptr = out_buff;
    // cacluate skip bytes
    out_ptr += ((static_cast<uint64_t>(start)) << kSkipBytesIndex);
    StatelessDropOutGenMaskCpuKernel::StatelessDropOutGenMaskKernel(
        count, reinterpret_cast<const uint8_t *>(&offset),
        reinterpret_cast<const uint8_t *>(&key), out_ptr);
  };
  // shared unit size
  const int64_t total_unit =
      static_cast<int64_t>(outputSize >> kSkipBytesIndex);
  const int64_t per_unit_size = 1;
  CpuKernelUtils::ParallelFor(ctx, total_unit, per_unit_size, shards);
#else
  KERNEL_LOG_INFO("StatelessDropOutGenMaskKernel::X86.");
  StatelessDropOutGenMaskKernel(outputSize,
                                reinterpret_cast<uint8_t *>(&offset),
                                reinterpret_cast<uint8_t *>(&key), out_buff);
#endif

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kStatelessDropOutGenMask, StatelessDropOutGenMaskCpuKernel);
}  // namespace aicpu