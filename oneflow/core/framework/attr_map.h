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
#ifndef ONEFLOW_CORE_FRAMEWORK_ATTR_MAP_H_
#define ONEFLOW_CORE_FRAMEWORK_ATTR_MAP_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/common/small_vector.h"

namespace oneflow {

namespace user_op {
class AttrVal;
}
class AttrValue;
class MutableAttrMap;
template<int N>
class CachedMutableAttrMap;
class UserOpConf;

class AttrMap final {
 public:
  AttrMap();
  AttrMap(const MutableAttrMap& other);

  template<int N>
  AttrMap(const CachedMutableAttrMap<N>& other);

  AttrMap(const UserOpConf& user_op_conf);

  AttrMap(const AttrMap&) = default;
  AttrMap(AttrMap&&) = default;
  ~AttrMap() = default;

  AttrMap& operator=(const AttrMap& other);

  bool operator==(const AttrMap& other) const;

  template<typename T>
  Maybe<const T&> GetAttr(const std::string& attr_name) const;

  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(const std::string& attr_name) const;
  bool HasAttr4Name(const std::string& attr_name) const;

  size_t size() const { return valid_size_; }
  bool empty() const { return valid_size_ > 0; }

  size_t hash_value() const { return hash_value_; }

  struct SharedAttr {
    const std::string* attr_names;
    small_vector<std::string, 4> allocated_attr_names;
    small_vector<std::pair<std::shared_ptr<const user_op::AttrVal>, bool>, 10> attrs;
  };

  class const_iterator {
   public:
    using reference = const std::pair<std::string, std::shared_ptr<const user_op::AttrVal>>&;
    using pointer = const std::pair<std::string, std::shared_ptr<const user_op::AttrVal>>*;

    const_iterator(int pos, int limit, const SharedAttr* data)
        : pos_(pos), limit_(limit), data_(data) {
      while (pos_ < limit_) {
        if (!data_->attrs[pos_].second) {
          ++pos_;
          continue;
        }
        kv_.first = data_->attr_names[pos_];
        kv_.second = data_->attrs[pos_].first;
      }
    }
    ~const_iterator() = default;

    reference operator*() const { return kv_; }
    pointer operator->() const { return &kv_; }

    const_iterator& operator++() {
      while (pos_ < limit_ - 1) {
        ++pos_;
        if (!data_->attrs[pos_].second) {
          ++pos_;
          continue;
        }
        kv_.first = data_->attr_names[pos_];
        kv_.second = data_->attrs[pos_].first;
      }
      return *this;
    }
    bool operator==(const const_iterator& x) const { return pos_ == x.pos_ && data_ == x.data_; }
    bool operator!=(const const_iterator& x) const { return !(*this == x); }

   private:
    int pos_;
    int limit_;
    const SharedAttr* data_;
    std::pair<std::string, std::shared_ptr<const user_op::AttrVal>> kv_;
  };

  const_iterator begin() const { return const_iterator(0, max_size_, data_.get()); }
  const_iterator end() const { return const_iterator(max_size_, max_size_, data_.get()); }

 private:
  int max_size_;
  int valid_size_;
  size_t hash_value_;
  std::shared_ptr<SharedAttr> data_;
};

AttrMap MakeAttrMapFromUserOpConf(const UserOpConf& user_op_conf);

class ComposedAttrMap final {
 public:
  ComposedAttrMap(const ComposedAttrMap&) = default;
  ComposedAttrMap(ComposedAttrMap&&) = default;
  ComposedAttrMap(const AttrMap& base) : base_(base) {}
  ComposedAttrMap(const AttrMap& prior, const AttrMap& base) : prior_(prior), base_(base) {}

  template<typename T>
  Maybe<const T&> GetAttr(const std::string& attr_name) const;

  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(const std::string& attr_name) const;

  bool HasAttr4Name(const std::string& attr_name) const;

  void ResetPrior(const AttrMap& prior) { prior_ = prior; }
  void ResetBase(const AttrMap& base) { base_ = base; }

 private:
  AttrMap prior_;
  AttrMap base_;
};

class MutableAttrMap : public std::map<std::string, std::shared_ptr<user_op::AttrVal>> {
 public:
  using std::map<std::string, std::shared_ptr<user_op::AttrVal>>::map;

  template<typename T>
  Maybe<void> SetAttr(const std::string& attr_name, const T& attr_val);
};

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::AttrMap> final {
  size_t operator()(const oneflow::AttrMap& attr_map) const { return attr_map.hash_value(); }
};

}  // namespace std

#endif  // ONEFLOW_CORE_FRAMEWORK_ATTR_MAP_H_
