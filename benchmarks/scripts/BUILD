load("@rules_cc//cc:defs.bzl", "cc_binary")
COPTS = ["-O3", "-fno-fast-math", "-march=native", "-std=c++20", "-I.", "-Isrc"]
cc_binary(name="original_worker", srcs=["worker.cc"], copts=COPTS,
    deps=["//benchmarks/baseline:original_lib", "@nlohmann_json//:json"])
cc_binary(name="optimized_worker", srcs=["worker.cc"], copts=COPTS+["-DFINAL_PRODUCTION"],
    deps=["//src:libtesseract", "//src:libutils", "@nlohmann_json//:json"])
cc_binary(name="input_export", srcs=["input_export.cc"], copts=COPTS,
    deps=["//src:libtesseract", "//src:libutils", "@nlohmann_json//:json", "@stim//:stim_lib"])
