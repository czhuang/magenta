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
  name = "lib_data",
  srcs = ["lib_data.py"],
  deps = [
    ":lib_mask",
    ":lib_pianoroll",
    ":lib_util",
    # tensorflow dep
  ]
)

py_library(
  name = "lib_mask",
  srcs = ["lib_mask.py"],
  deps = [
    ":lib_util",
    # tensorflow dep
  ]
)

py_library(
  name = "lib_util",
  srcs = ["lib_util.py"],
)

py_library(
  name = "lib_tfutil",
  srcs = ["lib_tfutil.py"],
  deps = [
      # tensorflow dep
  ]
)

py_library(
  name = "lib_pianoroll",
  srcs = ["lib_pianoroll.py"],
  deps = [
      # tensorflow dep
  ]
)

py_library(
  name = "lib_graph",
  srcs = ["lib_graph.py"],
  deps = [
    ":lib_util",
    ":lib_tfutil",
    # tensorflow dep
  ]
)

py_library(
  name = "lib_hparams",
  srcs = ["lib_hparams.py"],
  deps = [
    ":lib_util",
    # tensorflow dep
  ]
)


py_library(
    name = "lib_logging",
    srcs = ["lib_logging.py"],
    # tensroflow dep
)


py_library(
    name = "lib_sampling",
    srcs = ["lib_sampling.py"],
    deps = [
        ":lib_mask",
        ":lib_data",
        ":lib_util",
        ":lib_tfutil",
        ":lib_logging",
        # tensorflow dep
    ]
)


py_library(
    name = "lib_evaluation",
    srcs = ["lib_evaluation.py"],
    deps = [
        ":lib_mask",
        ":lib_util",
        ":lib_tfutil",
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
      ":lib_data",
      ":lib_util",
      ":lib_graph",
      ":lib_hparams",
        # tensorflow dep
    ],
)

py_binary(
    name = "evaluate",
    srcs = ["evaluate.py"],
    deps = [
        ":lib_data",
        ":lib_evaluation",
        ":lib_graph",
        ":lib_util",
        # tensorflow dep
    ],
)

py_binary(
    name = "sample",
    srcs = ["sample.py"],
    deps = [
        ":lib_data",
        ":lib_graph",
        ":lib_logging",
        ":lib_mask",
        ":lib_pianoroll",
        ":lib_sampling",
        ":lib_util",
        # tensorflow dep
    ],
)

