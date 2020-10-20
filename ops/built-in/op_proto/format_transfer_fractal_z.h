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

/*!
 * \file format_transfer_fractal_z.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_FORMAT_TRANSFER_FRACTAL_Z_H_
#define OPS_BUILT_IN_OP_PROTO_FORMAT_TRANSFER_FRACTAL_Z_H_

#include <vector>

#include "register/register_format_transfer.h"

namespace ge {
namespace formats {
class FormatTransferFractalZ_TBE : public FormatTransfer {
 public:
  Status TransFormat(const TransArgs& args, TransResult& result) override;
  Status TransShape(Format src_format, const std::vector<int64_t>& src_shape, DataType data_type, Format dst_format,
                    std::vector<int64_t>& dst_shape) override;
};
}  // namespace formats
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_FORMAT_TRANSFER_FRACTAL_Z_H_
