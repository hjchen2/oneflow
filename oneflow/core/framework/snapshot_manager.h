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
#ifndef ONEFLOW_CORE_FRAMEWORK_SNAPSHOT_MANAGER_H_
#define ONEFLOW_CORE_FRAMEWORK_SNAPSHOT_MANAGER_H_

#include <string>
#include "oneflow/core/common/util.h"

namespace oneflow {

class SnapshotManager {
 public:
  SnapshotManager() = default;
  virtual ~SnapshotManager() = default;

  Maybe<void> InitVariableSnapshotPath(const std::string& root_dir, bool refresh = true);

  Maybe<const std::string&> GetSnapshotPath(const std::string& variable_name) const;

 private:
  std::string default_path_ = "";
  HashMap<std::string, std::string> variable_name2path_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_SNAPSHOT_MANAGER_H_