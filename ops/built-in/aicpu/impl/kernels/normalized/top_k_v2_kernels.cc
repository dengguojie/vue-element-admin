/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of reshape
 */

#include "top_k_v2_kernels.h"

#include <securec.h>
#include <string>

#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

using std::string;

namespace {
const char *TOPK = "TopK";

template <class T, class Cmp = std::greater<T>> class TopN {
public:
    explicit TopN(size_t limit) : TopN(limit, Cmp()) {}
    TopN(size_t limit, const Cmp &cmp) : limit_(limit), cmp_(cmp) {}
    ~TopN() = default;

    size_t size() const
    {
        return std::min(values_.size(), limit_);
    }
    void reserve(size_t n)
    {
        values_.reserve(std::min(n, limit_ + 1));
    }

    void push(const T &v)
    {
        push(v, nullptr);
    }
    void push(const T &v, T *dropped)
    {
        PushInternal(v, dropped);
    }

    void push(T &&v)
    {
        push(std::move(v), nullptr);
    }
    void push(T &&v, T *dropped)
    {
        PushInternal(std::move(v), dropped);
    }

    std::vector<T> *Extract(bool extract_sorted = true);

    Cmp *comparator()
    {
        return &cmp_;
    }


private:
    template <typename U> void PushInternal(U &&v, T *dropped);

    std::vector<T> values_;
    size_t limit_;
    Cmp cmp_;
    bool heap_sorted_ = false;
};

template <class T, class Cmp> template <typename U> void TopN<T, Cmp>::PushInternal(U &&v, T *dropped)
{
    if (limit_ == 0) {
        if (dropped)
            *dropped = std::forward<U>(v);
        return;
    }
    if (!heap_sorted_) {
        values_.push_back(std::forward<U>(v));
        if (!heap_sorted_ || cmp_(values_.back(), values_.front())) {
        } else {
            std::swap(values_.front(), values_.back());
        }
        if (values_.size() == limit_ + 1) {
            std::make_heap(values_.begin(), values_.end(), cmp_);
            if (dropped)
                *dropped = std::move(values_.front());
            std::pop_heap(values_.begin(), values_.end(), cmp_);
            heap_sorted_ = true;
        }
    } else {
        if (cmp_(v, values_.front())) {
            values_.back() = std::forward<U>(v);

            std::pop_heap(values_.begin(), values_.end(), cmp_);
            if (dropped)
                *dropped = std::move(values_.back());
        } else {
            if (dropped)
                *dropped = std::forward<U>(v);
        }
    }
}

template <class T, class Cmp> std::vector<T> *TopN<T, Cmp>::Extract(bool extract_sorted)
{
    auto out = new std::vector<T>;
    if (extract_sorted) {
        out->swap(values_);
        if (!heap_sorted_) {
            std::sort(out->begin(), out->end(), cmp_);
        } else {
            out->pop_back();
            std::sort_heap(out->begin(), out->end(), cmp_);
        }
    } else {
        values_.resize(size());
        out->swap(values_);
    }

    return out;
}
}

namespace aicpu {
uint32_t TopKV2CpuKernel::Compute(CpuKernelContext &ctx)
{
    KERNEL_LOG_INFO("TopKV2CpuKernel::Compute start!! ");

    uint32_t res = GetInputAndCheck(ctx);
    if (res != KERNEL_STATUS_OK) {
        return res;
    }

    size_t type_size = GetSizeByDataType(static_cast<DataType>(matrix_info_.matrix_type));
    if (type_size < 1) {
        KERNEL_LOG_ERROR("don't support input tensor types");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    switch (matrix_info_.matrix_type) {
        case DT_FLOAT: {
            float data_type = 0;
            DoCompute(data_type);
            break;
        }
        case DT_FLOAT16: {
            Eigen::half data_type = Eigen::half(0);
            DoCompute(data_type);
            break;
        }
        case DT_INT32: {
            int data_type = 0;
            DoCompute(data_type);
            break;
        }
        default: {
            KERNEL_LOG_ERROR("TopKV2 op don't support input tensor types: %d", matrix_info_.matrix_type);
            return KERNEL_STATUS_PARAM_INVALID;
        }
    }

    KERNEL_LOG_INFO("TopKV2CpuKernel::Compute end!! ");
    return KERNEL_STATUS_OK;
}

template <typename T> uint32_t TopKV2CpuKernel::DoCompute(T data_type)
{
    KERNEL_LOG_INFO("TopKV2CpuKernel::DoCompute start!! ");
    auto input_data = reinterpret_cast<T *>(input_tensor_->GetData());
    auto values_data = reinterpret_cast<T *>(output_values_->GetData());
    auto indices_data = reinterpret_cast<T *>(output_indices_->GetData());

    KERNEL_LOG_INFO("TopKV2 input data type: %s", typeid(data_type).name());
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> input_map((T *)input_data, row_, col_);
    const auto &input = Eigen::Tensor<T, 2, Eigen::RowMajor>(input_map);

    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> values((T *)values_data, row_, k_);
    Eigen::TensorMap<Eigen::Tensor<int, 2, Eigen::RowMajor>> indices((int *)indices_data, row_, k_);
    for (int32_t r = 0; r < row_; ++r) { // 0~row_
        const T *input_data = &input(r, 0);
        const auto comp = [input_data](const size_t a, const size_t b) {
            if (input_data[b] == input_data[a]) {
                return a < b;
            }
            return input_data[b] < input_data[a];
        };
        if (k_ == col_) {
            std::iota(&indices(r, 0), &indices(r, k_), 0);
            std::sort(&indices(r, 0), &indices(r, k_), comp);
        } else {
            TopN<size_t, decltype(comp)> filter(k_, comp);
            filter.reserve(col_);
            for (int32_t c = 0; c < col_; ++c) {
                filter.push(c);
            }
            size_t i = 0;
            std::unique_ptr<std::vector<size_t>> top_k(filter.Extract(sorted_));
            for (auto top_k_it = top_k->begin(); top_k_it != top_k->end(); ++top_k_it, ++i) {
                indices(r, i) = *top_k_it;
            }
        }
        for (int32_t j = 0; j < k_; ++j) {
            values(r, j) = input(r, indices(r, j));
        }
    }
    KERNEL_LOG_INFO("TopKV2CpuKernel::DoCompute end!! ");
    return KERNEL_STATUS_OK;
}

uint32_t TopKV2CpuKernel::GetInputAndCheck(CpuKernelContext &ctx)
{
    KERNEL_LOG_INFO("TopKV2CpuKernel::GetInputAndCheck start!! ");

    // get x
    input_tensor_ = ctx.Input(0);
    if (input_tensor_ == nullptr) {
        KERNEL_LOG_ERROR("get input:0 failed");
        KERNEL_LOG_INFO("TopKV2CpuKernel::GetInputAndCheck end!! ");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    // get col_
    std::shared_ptr<TensorShape> input_shape = input_tensor_->GetTensorShape();
    int32_t input_rank = input_shape->GetDims();

    KERNEL_LOG_INFO("input dim size is %d(input_rank) ", input_rank);

    if (input_rank < 1) {
        KERNEL_LOG_ERROR("input must be >= 1-D");
        KERNEL_LOG_INFO("TopKV2CpuKernel::GetInputAndCheck end!! ");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    col_ = input_shape->GetDimSize(input_rank - 1);
    if (col_ <= 0) {
        KERNEL_LOG_ERROR("col_:%lld must be > 0", col_);
        return KERNEL_STATUS_PARAM_INVALID;
    }

    // cal row_
    size_t input_size = 1;
    matrix_info_.matrix_type = static_cast<DataType>(input_tensor_->GetDataType());
    for (int32_t i = 0; i < input_rank; ++i) {
        matrix_info_.matrix_shape.push_back(input_shape->GetDimSize(i));
        input_size *= input_shape->GetDimSize(i);
    }
    row_ = input_size / col_;

    // get k
    Tensor *k_tensor = ctx.Input(1);
    if (k_tensor == nullptr) {
        KERNEL_LOG_ERROR("get input:1 failed");
        KERNEL_LOG_INFO("TopKV2CpuKernel::GetInputAndCheck end!! ");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    k_ = *static_cast<int32_t *>(k_tensor->GetData());
    if (k_ <= 0) {
        KERNEL_LOG_ERROR("k must be greater than 0, but got %d", k_);
        KERNEL_LOG_INFO("TopKV2CpuKernel::GetInputAndCheck end!! ");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    KERNEL_LOG_INFO("input row_ is %d, col_ is %d and k_ is %d", row_, col_, k_);

    if (col_ < k_) {
        KERNEL_LOG_ERROR("input must have at least %d(k) columns, but got %d", k_, col_);
        KERNEL_LOG_INFO("TopKV2CpuKernel::GetInputAndCheck end!! ");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    // get attr: sorted
    AttrValue *sorted = ctx.GetAttr("sorted");
    KERNEL_CHECK_NULLPTR(sorted, KERNEL_STATUS_PARAM_INVALID, "get attr:sorted failed.");
    sorted_ = sorted->GetBool();

    output_values_ = ctx.Output(0);
    if (output_values_ == nullptr) {
        KERNEL_LOG_ERROR("get output:0 failed");
        KERNEL_LOG_INFO("TopKV2CpuKernel::GetInputAndCheck end!! ");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    output_indices_ = ctx.Output(1);
    if (output_indices_ == nullptr) {
        KERNEL_LOG_ERROR("get output:1 failed");
        KERNEL_LOG_INFO("TopKV2CpuKernel::GetInputAndCheck end!! ");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    KERNEL_LOG_INFO("TopKV2CpuKernel::GetInputAndCheck end!! ");
    return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(TOPK, TopKV2CpuKernel);
} // namespace aicpu
