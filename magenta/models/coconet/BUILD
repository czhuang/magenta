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
  srcs = ["lib_data.py"],
  deps = [
    ":mask",
    ":pianoroll",
    ":util",
    # tensorflow dep
  ]
)

py_library(
  name = "mask",
  srcs = ["lib_mask.py"],
  deps = [
    ":util",
    # tensorflow dep
  ]
)

py_library(
  name = "util",
  srcs = ["lib_util.py"],
)

py_library(
  name = "tfutil",
  srcs = ["lib_tfutil.py"],
  deps = [
      # tensorflow dep
  ]
)

py_library(
  name = "pianoroll",
  srcs = ["lib_pianoroll.py"],
  deps = [
      # tensorflow dep
  ]
)

py_library(
  name = "graph",
  srcs = ["lib_graph.py"],
  deps = [
    ":util",
    ":tfutil",
    # tensorflow dep
  ]
)

py_library(
  name = "hparams",
  srcs = ["lib_hparams.py"],
  deps = [
    ":util",
    # tensorflow dep
  ]
)


py_library(
    name = "logging",
    srcs = ["lib_logging.py"],
    # tensroflow dep
)


py_library(
    name = "sampling",
    srcs = ["lib_sampling.py"],
    deps = [
        ":mask",
        ":data",
        ":util",
        ":tfutil",
        ":logging",
        # tensorflow dep
    ]
)


py_library(
    name = "evaluation",
    srcs = ["lib_evaluation.py"],
    deps = [
        ":mask",
        ":util",
        ":tfutil",
        # tensorflow dep
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
        # tensorflow dep
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
        # tensorflow dep
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
        # tensorflow dep
    ],
)

