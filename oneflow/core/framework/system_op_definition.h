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
#ifndef ONEFLOW_CORE_FRAMEWORK_SYSTEM_OP_DEFINITION_H_
#define ONEFLOW_CORE_FRAMEWORK_SYSTEM_OP_DEFINITION_H_

#include "oneflow/core/framework/op_definition.h"

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {
namespace schema {

class CastToGlobalOp : public OpDefinition<CastToGlobalOp> {
 public:
  Maybe<AttrVal> Attr(const std::string& attr_name) const override;
  static const Set<std::string>& AttrNames();

 public:
  Shape shape;
  DataType dtype;
};

class SelectTopNOp : public OpDefinition<SelectTopNOp> {
 public:
  Maybe<AttrVal> Attr(const std::string& attr_name) const override;
  static const Set<std::string>& AttrNames();

 public:
  int32_t top_n;
};

class FeedInputOp : public OpDefinition<FeedInputOp> {
 public:
  Maybe<AttrVal> Attr(const std::string& attr_name) const override;
  static const Set<std::string>& AttrNames();
};

class FetchOutputOp : public OpDefinition<FetchOutputOp> {
 public:
  Maybe<AttrVal> Attr(const std::string& attr_name) const override;
  static const Set<std::string>& AttrNames();
};

class FeedVariableOp : public OpDefinition<FeedVariableOp> {
 public:
  Maybe<AttrVal> Attr(const std::string& attr_name) const override;
  static const Set<std::string>& AttrNames();

 public:
  double l2;
};

class ImageDecoderRandomCropResizeOp : public OpDefinition<ImageDecoderRandomCropResizeOp> {
 public:
  Maybe<AttrVal> Attr(const std::string& attr_name) const override;
  static const Set<std::string>& AttrNames();

 public:
  int64_t target_width;
  int64_t target_height;
  int64_t num_workers;
  int64_t max_num_pixels;
  int64_t warmup_size;
  int64_t seed;
  int64_t num_attempts;
  float random_area_min;
  float random_area_max;
  float random_aspect_ratio_min;
  float random_aspect_ratio_max;
};

}  // namespace schema
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_SYSTEM_OP_DEFINITION_H_
