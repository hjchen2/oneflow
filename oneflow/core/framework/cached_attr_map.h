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
#ifndef ONEFLOW_CORE_FRAMEWORK_CACHED_ATTR_MAP_H_
#define ONEFLOW_CORE_FRAMEWORK_CACHED_ATTR_MAP_H_

#include "llvm/ADT/StringRef.h"

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/attr_value.h"
#include "oneflow/core/framework/attr_value_accessor.h"
#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {

template<int N>
class CachedMutableAttrMap {
 public:
  CachedMutableAttrMap() : element_size_(0), hash_value_(0) {}
  ~CachedMutableAttrMap() = default;

  int size() const { return element_size_; }
  size_t hash_value() const { return hash_value_; }
  const std::string* attr_names() const { return attr_names_; }
  const bool* valid_masks() const { return valid_masks_; }
  const std::shared_ptr<user_op::AttrVal>* attrs() const { return attrs_; }

  void reset() {
    hash_value_ = 0;
    memset(valid_masks_, 0, element_size_);
  }

  template<typename T>
  void SetAttr(const char* attr_name, const T& attr_val) {
    auto it = name_indices_.find(attr_name);
    if (it == name_indices_.end()) {
      CHECK_LT_OR_THROW(element_size_, N)
          << "cached attribute map space is not enough and current size is " << N
          << ", please enlarge it";
      attrs_[element_size_] = std::make_shared<user_op::TypedAttrVal<T>>(attr_val);
      valid_masks_[element_size_] = true;
      attr_names_[element_size_] = attr_name;
      it = name_indices_.emplace(attr_names_[element_size_], element_size_).first;
      ++element_size_;
    } else {
      if (/*attrs_[i].v->value_type() != user_op::GetAttrType<T>::value ||*/
          *static_cast<const T*>(attrs_[it->second]->Ptr()) != attr_val) {
        attrs_[it->second] = std::make_shared<user_op::TypedAttrVal<T>>(attr_val);
      }
      valid_masks_[it->second] = true;
    }
    HashCombine(&hash_value_, attr_names_[it->second].size());
    HashCombine(&hash_value_, std::hash<T>()(attr_val));
  }

 private:
  // the actually element size
  int element_size_;
  size_t hash_value_;

  std::map<llvm::StringRef, int> name_indices_;
  std::string attr_names_[N];

  bool valid_masks_[N];
  std::shared_ptr<user_op::AttrVal> attrs_[N];
};

class AttrMap2 final {
 public:
  template<int N>
  AttrMap2(const CachedMutableAttrMap<N>& other)
      : max_size_(other.size()),
        valid_size_(0),
        hash_value_(other.hash_value()),
        data_(std::make_shared<small_vector<SharedAttr, 10>>()) {
    data_->resize(max_size_);
    for (int i = 0; i < max_size_; ++i) {
      (*data_)[i].valid_mask = other.valid_masks()[i];
      if ((*data_)[i].valid_mask) {
        ++valid_size_;
        (*data_)[i].attr = other.attrs()[i];
      }
    }
    attr_names_ = other.attr_names();
  }

  AttrMap2(const AttrMap2&) = default;
  AttrMap2(AttrMap2&&) = default;
  ~AttrMap2() = default;

  struct SharedAttr {
    // bool valid_masks[20];
    // std::shared_ptr<const user_op::AttrVal> attrs[20];
    bool valid_mask;
    std::shared_ptr<const user_op::AttrVal> attr;
  };

 public:
  int max_size_;
  int valid_size_;
  size_t hash_value_;
  std::shared_ptr<small_vector<SharedAttr, 10>> data_;

  const std::string* attr_names_;
  small_vector<std::string, 4> allocated_attr_names_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_CACHED_ATTR_MAP_H_
