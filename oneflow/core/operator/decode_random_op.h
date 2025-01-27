/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_OPERATOR_DECODE_RANDOM_OP_H_
#define ONEFLOW_CORE_OPERATOR_DECODE_RANDOM_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class DecodeRandomOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeRandomOp);
  DecodeRandomOp() = default;
  ~DecodeRandomOp() = default;

  void InitFromOpConf() override;
  LogicalNode* NewProperLogicalNode() const override { return new DecodeRandomLogicalNode; }

  Maybe<void> InferOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_DECODE_RANDOM_OP_H_
