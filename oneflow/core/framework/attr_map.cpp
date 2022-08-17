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

#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/attr_value.h"
#include "oneflow/core/framework/attr_value_accessor.h"
#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/framework/cached_attr_map.h"

namespace oneflow {

AttrMap::AttrMap()
    : max_size_(0),
      valid_size_(0),
      hash_value_(0),
      data_(std::make_shared<AttrMap::SharedAttr>()) {}

AttrMap::AttrMap(const MutableAttrMap& other)
    : max_size_(other.size()),
      valid_size_(other.size()),
      hash_value_(0),
      data_(std::make_shared<AttrMap::SharedAttr>()) {
  data_->allocated_attr_names.reserve(max_size_);
  data_->attrs.reserve(max_size_);
  for (const auto& it : other) {
    data_->attrs.emplace_back(it.second, true);
    data_->allocated_attr_names.emplace_back(it.first);

    HashCombine(&hash_value_, it.first.size());
    HashCombine(&hash_value_, it.second->hash_value());
  }
  data_->attr_names = data_->allocated_attr_names.data();
}

template<int N>
AttrMap::AttrMap(const CachedMutableAttrMap<N>& other)
    : max_size_(other.size()),
      valid_size_(0),
      hash_value_(other.hash_value()),
      data_(std::make_shared<AttrMap::SharedAttr>()) {
  data_->attrs.resize(max_size_);
  for (int i = 0; i < max_size_; ++i) {
    data_->attrs[i].second = other.valid_masks()[i];
    if (data_->attrs[i].second) {
      ++valid_size_;
      data_->attrs[i].first = other.attrs()[i];
    }
  }
  data_->attr_names = other.attr_names();
}

#define INSTANCE_CACHED_MUTABLE_ATTR_MAP_CONSTRUCTOR(N) \
  template AttrMap::AttrMap(const CachedMutableAttrMap<N>&);

INSTANCE_CACHED_MUTABLE_ATTR_MAP_CONSTRUCTOR(5)

AttrMap::AttrMap(const UserOpConf& user_op_conf)
    : max_size_(0), valid_size_(0), hash_value_(0), data_(std::make_shared<AttrMap::SharedAttr>()) {
  for (const auto& kv : user_op_conf.attr()) {
    auto cpp_attr_value = user_op::AttrValueUtil::ToCppAttrValue(kv.second);
    if (cpp_attr_value.IsOk()) {
      ++max_size_;
      data_->attrs.emplace_back(CHECK_JUST(cpp_attr_value), true);
      data_->allocated_attr_names.emplace_back(kv.first);

      HashCombine(&hash_value_, kv.first.size());
      HashCombine(&hash_value_, data_->attrs.back().first->hash_value());
    } else {
      LOG(ERROR) << user_op_conf.DebugString()
                 << " failed to convert to cpp attr value, key: " << kv.first;
    }
  }
  valid_size_ = max_size_;
  data_->attr_names = data_->allocated_attr_names.data();
}

AttrMap& AttrMap::operator=(const AttrMap& other) {
  max_size_ = other.max_size_;
  valid_size_ = other.valid_size_;
  hash_value_ = other.hash_value_;
  data_ = other.data_;
  return *this;
}

bool AttrMap::operator==(const AttrMap& other) const {
  if (valid_size_ != other.valid_size_ || hash_value_ != other.hash_value_) { return false; }
  for (int i = 0; i < std::min(max_size_, other.max_size_); ++i) {
    if (data_->attrs[i].second != other.data_->attrs[i].second) { return false; }
    if (data_->attr_names[i] != other.data_->attr_names[i]) { return false; }
    if (*(data_->attrs[i].first) != *(other.data_->attrs[i].first)) { return false; }
  }
  return true;
}

template<typename T>
Maybe<const T&> AttrMap::GetAttr(const std::string& attr_name) const {
  const auto& attr = Attr4Name(attr_name);
  CHECK_OR_RETURN(attr) << Error::InvalidValueError()
                        << "no attribute found. attribute name: " << attr_name;
  const auto* ptr = dynamic_cast<const user_op::TypedAttrVal<T>*>(attr.get());
  CHECK_NOTNULL_OR_RETURN(ptr);
  return ptr->val();
}

const std::shared_ptr<const user_op::AttrVal>& AttrMap::Attr4Name(
    const std::string& attr_name) const {
  for (int i = 0; i < max_size_; ++i) {
    if (data_->attrs[i].second && attr_name.data() == data_->attr_names[i]) {
      return data_->attrs[i].first;
    }
  }
  static const std::shared_ptr<const user_op::AttrVal> none;
  return none;
}

bool AttrMap::HasAttr4Name(const std::string& attr_name) const {
  return Attr4Name(attr_name) != nullptr;
}

AttrMap MakeAttrMapFromUserOpConf(const UserOpConf& user_op_conf) { return AttrMap(user_op_conf); }

template<typename T>
Maybe<const T&> ComposedAttrMap::GetAttr(const std::string& attr_name) const {
  const auto& attr = Attr4Name(attr_name);
  CHECK_OR_RETURN(attr) << Error::InvalidValueError()
                        << "no attribute found. attribute name: " << attr_name;
  return dynamic_cast<const user_op::TypedAttrVal<T>*>(attr.get())->val();
}

const std::shared_ptr<const user_op::AttrVal>& ComposedAttrMap::Attr4Name(
    const std::string& attr_name) const {
  const auto& prior_attr = prior_.Attr4Name(attr_name);
  if (prior_attr) { return prior_attr; }
  return base_.Attr4Name(attr_name);
}

bool ComposedAttrMap::HasAttr4Name(const std::string& attr_name) const {
  return Attr4Name(attr_name) != nullptr;
}

#define DEFINE_ATTR_VALUE_MAP_GET_ATTR(field, T, attr_type)                         \
  template Maybe<const T&> AttrMap::GetAttr<T>(const std::string& attr_name) const; \
  template Maybe<const T&> ComposedAttrMap::GetAttr<T>(const std::string& attr_name) const;

OF_PP_FOR_EACH_TUPLE(DEFINE_ATTR_VALUE_MAP_GET_ATTR, ATTR_SEQ);
#undef DEFINE_ATTR_VALUE_MAP_GET_ATTR

template<>
Maybe<void> MutableAttrMap::SetAttr(const std::string& attr_name,
                                    const std::shared_ptr<user_op::AttrVal>& attr_val) {
  (*this)[attr_name] = attr_val;
  return Maybe<void>::Ok();
}

template<typename T>
Maybe<void> MutableAttrMap::SetAttr(const std::string& attr_name, const T& attr_val) {
  (*this)[attr_name] = std::make_shared<user_op::TypedAttrVal<T>>(attr_val);
  return Maybe<void>::Ok();
}

#define DEFINE_ATTR_VALUE_MAP_SET_ATTR(field, T, attr_type) \
  template Maybe<void> MutableAttrMap::SetAttr<T>(const std::string& attr_name, const T& attr_val);

OF_PP_FOR_EACH_TUPLE(DEFINE_ATTR_VALUE_MAP_SET_ATTR, ATTR_SEQ);
#undef DEFINE_ATTR_VALUE_MAP_SET_ATTR

}  // namespace oneflow
