

py_binary(
    name = "evaluate_gibbs_tools",
    srcs = ["evaluate_gibbs_tools.py"],
    deps = [
        ":postprocess",
	":evaluation_tools",
        ":retrieve_model_tools",
        ":pianorolls_lib", 
    ],
)


py_binary(
    name = "create_dataset_and_generate",
    srcs = ["create_dataset_and_generate.py"],
    deps = [
        ":create_dataset",
        ":generate_tools",
        "//magenta/lib:note_sequence_io",
    ],
)

py_binary(
    name = 'dataset_statistics_tools',
    srcs = ["dataset_statistics_tools.py"],
    deps = [
        ":data_tools",
        ":test_tools",
        ":seed_tools",
        "//magenta/lib:note_sequence_io",
        "//magenta/lib:midi_io",
    ],
)

py_binary(
    name = "postprocess",
    srcs = ["postprocess.py"],
    deps = [
        ":generate_tools",
        "//magenta/protobuf:music_py_pb2",
        "//magenta/lib:midi_io",
    ],
)

py_binary(
    name = "music21_corpus_to_note_sequences",
    srcs = ["music21_corpus_to_note_sequences.py"],
    deps = [
        ":pianorolls_lib",
        ":test_tools",
        "//magenta/lib:midi_io",
	"//magenta/lib:music_xml_io",
        "//magenta/lib:note_sequence_io",
    ],
)

py_binary(
    name = "seed_tools",
    srcs = ["seed_tools.py"],
    data = [
        "testdata/jsb/bach-one_phrase-note_sequence.tfrecord",
    ],
    deps = [
        ":config_tools",
        ":data_tools",
        ":mask_tools",
        ":pianorolls_lib",
        ":retrieve_model_tools",
        ":test_tools",
        "//magenta/lib:midi_io",
        "//magenta/lib:music_xml_io",
        "//magenta/lib:note_sequence_io",
    ],
)

py_binary(
    name = "evaluation_tools",
    srcs = ["evaluation_tools.py"],
    data = [
    ],
    deps = [
        ":config_tools",
        ":mask_tools",
        ":data_tools",
        ":pianorolls_lib",
        ":seed_tools",
        ":retrieve_model_tools",
        "//magenta/lib:midi_io",
    ],
)

py_binary(
    name = "generate_tools",
    srcs = ["generate_tools.py"],
    data = [
    ],
    deps = [
        ":config_tools",
        ":mask_tools",
        ":pianorolls_lib",
        ":seed_tools",
	":plot_tools",
        ":retrieve_model_tools",
        "//magenta/lib:midi_io",
    ],
)

py_binary(
    name = "gibbs",
    srcs = ["gibbs.py"],
    deps = [
        ":config_tools",
        ":mask_tools",
        ":retrieve_model_tools",
	":generate_tools",
    ],
)

py_binary(
    name = "pianorolls_lib",
    srcs = ["pianorolls_lib.py"],
    deps = [
        ":test_tools",
        "//magenta/lib:midi_io",
    ],
)

py_binary(
    name = "hparams_tools",
    srcs = ["hparams_tools.py"],
    deps = [
    ],
)

py_binary(
    name = "config_tools",
    srcs = ["config_tools.py"],
    deps = [
        ":hparams_tools",
    ],
)

py_binary(
    name = "data_tools",
    srcs = ["data_tools.py"],
    data = [
        "testdata/jsb/test.tfrecord",
        "testdata/jsb/train.tfrecord",
        "testdata/jsb/valid.tfrecord",
    ],
    deps = [
        ":mask_tools",
        ":pianorolls_lib",
        "//magenta/lib:note_sequence_io",
        "//magenta/protobuf:music_py_pb2",
    ],
)

py_binary(
    name = "mask_tools",
    srcs = ["mask_tools.py"],
    deps = [
    ],
)

py_binary(
    name = "summary_tools",
    srcs = ["summary_tools.py"],
    deps = [],
)


py_binary(
    name = "basic_autofill_cnn_graph",
    srcs = ["basic_autofill_cnn_graph.py"],
    deps = [
    ],
)

py_binary(
    name = "basic_autofill_cnn_train",
    srcs = ["basic_autofill_cnn_train.py"],
    deps = [
        ":config_tools",
        ":data_tools",
        ":pianorolls_lib",
        ":summary_tools",
        ":basic_autofill_cnn_graph",
    ],
)

py_binary(
    name = "plot_tools",
    srcs = ["plot_tools.py"],
    data = [
        "testdata/generated/sample1.midi",
    ],
    deps = [
	":pianorolls_lib",
	":mask_tools",
	":config_tools",
        "//magenta/protobuf:music_py_pb2",
        "//magenta/lib:midi_io",
    ],
)

py_binary(
    name = "create_dataset",
    srcs = ["create_dataset.py"],
    data = [
        "testdata/jsb/note_sequences.tfrecord",
    ],
    deps = [
        ":pianorolls_lib",
        "//magenta/lib:note_sequence_io",
        "//magenta/pipelines:dag_pipeline",
        "//magenta/pipelines:pipeline",
        "//magenta/pipelines:pipelines_common",
        "//magenta/pipelines:statistics",
        "//magenta/protobuf:music_py_pb2",
    ],
)

py_binary(
    name = "retrieve_model_tools",
    srcs = ["retrieve_model_tools.py"],
    data = [
        "checkpoints/balanced_by_scaling_64-128.ckpt",
        "checkpoints/random_64-128_0.5.ckpt",
        "checkpoints/balanced_64-128_fc_mask_only.ckpt",
        "checkpoints/denoising_d-64_f-128.ckpt",
        "checkpoints/DeepStraightConvSpecs_d-32_f-256_best_model.ckpt",
        "checkpoints/DeepStraightConvSpecs_d-64_f-128_best_model.ckpt",
        "checkpoints/DeepStraightConvSpecs-multi_instr_depth-28_filter-256-256-best_model.ckpt",
        "checkpoints/DeepStraightConvSpecs-with_res-multi_instr_depth-28_filter-256-256-best_model.ckpt",
        "checkpoints/DeepStraightConvSpecs-with_res-with_aug-multi_instr_depth-28_filter-256-256-best_model.ckpt",
        "checkpoints/PitchFullyConnected-multi_instr_depth-29_filter-256-256-best_model.ckpt",
    ],
    deps = [
        ":basic_autofill_cnn_graph",
        ":config_tools",
    ],
)

py_test(
    name = "retrieve_model_tools_test",
    size = "large",
    srcs = ["retrieve_model_tools_test.py"],
    deps = [
        ":basic_autofill_cnn_graph",
        ":config_tools",
        ":retrieve_model_tools",
        ":test_tools",
    ],
)

py_binary(
    name = "test_tools",
    srcs = ["test_tools.py"],
    data = [
        "testdata/jsb/6_note_sequences_with_only_4_voices.tfrecord",
        "testdata/jsb/note_sequences.tfrecord",
    ],
    deps = [
        ":basic_autofill_cnn_graph",
        "//magenta/protobuf:music_py_pb2",
    ],
)

py_test(
    name = "pianorolls_lib_test",
    size = "small",
    srcs = ["pianorolls_lib_test.py"],
    data = [
        "testdata/jsb/note_sequences.tfrecord",
    ],
    deps = [
        ":mask_tools",
        ":pianorolls_lib",
        ":test_tools",
    ],
)

py_test(
    name = "create_dataset_test",
    size = "small",
    srcs = ["create_dataset_test.py"],
    data = [
        "testdata/jsb/note_sequences.tfrecord",
    ],
    deps = [
        ":create_dataset",
    ],
)

py_test(
    name = "hparams_tools_test",
    size = "small",
    srcs = ["hparams_tools_test.py"],
    deps = [
        ":hparams_tools",
    ],
)

py_test(
    name = "basic_autofill_cnn_graph_test",
    size = "medium",
    srcs = ["basic_autofill_cnn_graph_test.py"],
    deps = [
        ":basic_autofill_cnn_graph",
        ":config_tools",
        ":test_tools",
    ],
)

py_test(
    name = "mask_tools_test",
    size = "small",
    srcs = ["mask_tools_test.py"],
    deps = [
        ":mask_tools",
    ],
)

py_test(
    name = "data_tools_test",
    size = "small",
    srcs = ["data_tools_test.py"],
    deps = [
        ":config_tools",
        ":data_tools",
        ":pianorolls_lib",
    ],
)

py_test(
    name = "basic_autofill_cnn_train_test",
    size = "medium",
    srcs = ["basic_autofill_cnn_train_test.py"],
    deps = [
        ":basic_autofill_cnn_train",
    ],
)



py_test(
    name = "summary_tools_test",
    size = "small",
    srcs = ["summary_tools_test.py"],
    deps = [
        ":summary_tools",
    ],
)
