load("@rules_cc//cc:defs.bzl", "cc_library")
cc_library(name="original_lib", srcs=["original_tesseract.cc"], hdrs=["tesseract.h"],
    copts=["-O3","-fno-fast-math","-march=native","-std=c++20","-Isrc"],
    deps=["//src:libcommon","//src:libdecoder","//src:libutils","//src:libviz","@boost//:dynamic_bitset","@nlohmann_json//:json","@stim//:stim_lib"],
    visibility=["//benchmarks/scripts:__pkg__"])
