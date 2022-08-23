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
#ifndef ONEFLOW_USER_KERNELS_COLLECTIVE_COMMUNICATION_CUDA_CUDA_SEND_RECV_UTIL_H_
#define ONEFLOW_USER_KERNELS_COLLECTIVE_COMMUNICATION_CUDA_CUDA_SEND_RECV_UTIL_H_

#ifdef WITH_CUDA
#include "oneflow/core/device/nccl_util.h"

namespace oneflow {

namespace ccl {

extern std::pair<ncclComm_t, int64_t> (*GetNcclCommAndPeerNcclRank)(int64_t peer_process_i);

}  // namespace ccl

}  // namespace oneflow

#endif  // WITH_CUDA

#endif  // ONEFLOW_USER_KERNELS_COLLECTIVE_COMMUNICATION_CUDA_CUDA_SEND_RECV_UTIL_H_
