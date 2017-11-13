# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Description:
# Tools and models for using TensorFlow with music and art.

licenses(["notice"])  # Apache 2.0

py_library(
  name = "data",
  srcs = ["lib/data.py"],
  deps = [
    ":mask",
    ":pianoroll",
    ":util",
  ]
)

py_library(
  name = "mask",
  srcs = ["lib/mask.py"],
  deps = [
    ":util",
  ]
)

py_library(
  name = "util",
  srcs = ["lib/util.py"],
)

py_library(
  name = "tfutil",
  srcs = ["lib/tfutil.py"],
)

py_library(
  name = "pianoroll",
  srcs = ["lib/pianoroll.py"],
)

py_library(
  name = "graph",
  srcs = ["lib/graph.py"],
  deps = [
    ":util",
    ":tfutil",
  ]
)

py_library(
  name = "hparams",
  srcs = ["lib/hparams.py"],
  deps = [
    ":util",
  ]
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    data = [
        "testdata/Jsb16thSeparated.npz",
    ],
    deps = [
      ":data",
      ":util",
      ":graph",
      ":hparams",
    ],
)

py_binary(
    name = "evaluate",
    srcs = ["evaluate.py"],
    deps = [
        ":data",
        ":evaluation",
        ":graph",
        ":util",
    ],
)

py_binary(
    name = "sample",
    srcs = ["sample.py"],
    deps = [
        ":data",
        ":graph",
        ":logging",
        ":mask",
        ":pianoroll",
        ":sampling",
        ":util",
    ],
)
