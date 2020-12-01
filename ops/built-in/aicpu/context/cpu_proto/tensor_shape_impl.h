/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of tensor shape
 */

#ifndef CPU_KERNEL_TENSOR_SHAPE_IMPL_H
#define CPU_KERNEL_TENSOR_SHAPE_IMPL_H
#include <vector>
#include <memory>
#include <functional>
#include "cpu_types.h"
#include "proto/cpu_tensor_shape.pb.h"

namespace aicpu {
class TensorShapeImpl {
    friend class CpuKernelUtils;

public:
    TensorShapeImpl(
        aicpuops::TensorShape *shape,
        std::function<void(aicpuops::TensorShape *)> delFunc = [](aicpuops::TensorShape *p) {})
        : tensorShape_(shape, delFunc)
    {}

    ~TensorShapeImpl() = default;
    TensorShapeImpl(const TensorShapeImpl &) = delete;
    TensorShapeImpl(TensorShapeImpl &&) = delete;
    TensorShapeImpl &operator=(const TensorShapeImpl &) = delete;
    TensorShapeImpl &operator=(TensorShapeImpl &&) = delete;

    /*
     * set format value to tensor shape.
     * @param format: format value need to set to tensor shape
     */
    void SetFormat(Format format);

    /*
     * get format value of tensor shape.
     * @return Format: format value of tensor shape
     */
    Format GetFormat() const;

    /*
     * get unknown rank value of tensor shape.
     * @return bool: unknown rank value of tensor shape
     */
    bool GetUnknownRank() const;

    /*
     * set unknown rank value to tensor shape.
     * @param unknownRank: unknown rank value need to set to tensor shape
     */
    void SetUnknownRank(bool unknownRank);

    /*
     * set dims value to tensor shape.
     * @param dims: dims value need to set to tensor shape
     */
    void SetDimSizes(const std::vector<int64_t> &dims);

    /*
     * get dims value of tensor shape.
     * @return int32_t: dims value of tensor shape
     */
    std::vector<int64_t> GetDimSizes() const;

    /*
     * get dim value of tensor shape index dim.
     * @param index: index dim of tensor shape
     * @return int64_t: dim value of tensor shape index dim
     */
    int64_t GetDimSize(int32_t index) const;

    /*
     * get dims size of tensor shape.
     * @return int32_t: dims size of tensor shape
     */
    int32_t GetDims() const;

    /*
     * get data elements number.
     * @return success->not less than 0, unknown->less than 0
     */
    int64_t NumElements() const;

    /*
     * get tensor shape proto.
     */
    aicpuops::TensorShape *GetProto() const;

private:
    std::shared_ptr<aicpuops::TensorShape> tensorShape_ { nullptr };
};
} // namespace aicpu
#endif // CPU_KERNEL_TENSOR_SHAPE_IMPL_H
