/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef AICPU_KERNELS_NORMALIZED_TOP_K_PQ_DISTANCE_H_
#define AICPU_KERNELS_NORMALIZED_TOP_K_PQ_DISTANCE_H_

#include <vector>

#include "cpu_kernel.h"
#include "utils/eigen_tensor.h"
namespace aicpu {
#if (defined __ARM_ARCH) || (defined PLANTFORM_AARCH64)
#include "arm_fp16.h"
#define FP16 float16_t
#else
#define FP16 Eigen::half
#endif
class TopKPQDistanceCpuKernel : public CpuKernel {
 public:
  TopKPQDistanceCpuKernel() = default;

  ~TopKPQDistanceCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  struct Item {
    T val;
    int32_t grp;
    int32_t grpi;
  };

  template <typename T>
  class InputGroup {
   private:
    std::vector<T *> data_;
    T **data_ptr_ = nullptr;

   public:
    std::vector<int32_t> data_count;

    void add(T *data, int32_t count) {
      this->data_.push_back(data);
      this->data_count.push_back(count);
      data_ptr_ = this->data_.data();
    }
    T **getPtr() const { return data_ptr_; }

    T getv(const int32_t grp, const int32_t grpi) const {
      return data_ptr_[grp][grpi];
    }
  };

  template <typename T>
  struct InputsData {
    int32_t actual_count = 0;
    InputGroup<T> pq_distances;
    InputGroup<T> grouped_extreme_distances;
    InputGroup<int32_t> pq_ivfs;
    InputGroup<int32_t> pq_indexs;
  };

  std::string order_;
  int32_t k_ = 0;
  int32_t group_size_ = 0;
  uint32_t data_batch_ = 0;
  bool is_min_heap_ = true;

  template <typename T>
  uint32_t GetInputAndCheck(CpuKernelContext &ctx, InputsData<T> &input_data);

  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);

  template <typename T>
  uint32_t ProcessResult(CpuKernelContext &ctx, const InputsData<T> &input_data,
                         Item<T> topk_ptr[]);

  template <typename T>
  void InitTopKHeap(int &cnt, int &cntk, Item<T> topk_ptr[],
                    const Item<T> grp_extreme_ptr[],
                    const InputsData<T> &inputs_data);

  template <typename T>
  uint32_t GetDistanceTopKHeap(Item<T> topk_ptr[],
                               const Item<T> grp_extreme_ptr[],
                               const InputsData<T> &inputs_data);

  template <typename T>
  uint32_t GetGroupedDistanceTopKHeap(Item<T> grp_extreme_ptr[],
                                      const InputsData<T> &input_data);

  template <typename T>
  void MakeHeap(Item<T> arr_ptr[], const int32_t n);

  template <typename T>
  void PopHeap(Item<T> arr_ptr[], const int32_t n, Item<T> *const res);

  template <typename T>
  inline void HeapFixdown(Item<T> a[], const int32_t index, const int32_t n);

  template <typename T>
  void SortHeap(Item<T> arr_ptr[], const int32_t n);

  template <typename T>
  void InitGrpExtreme(Item<T> grp_extreme_ptr[],
                      const InputsData<T> &input_data, int32_t &grp,
                      int32_t &grpi);
};
}  // namespace aicpu
#endif
