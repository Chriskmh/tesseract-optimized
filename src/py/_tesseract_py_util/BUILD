load("@rules_python//python:py_test.bzl", "py_test")
load("@rules_python//python:py_library.bzl", "py_library")

py_library(
    name = "_tesseract_py_util",
    srcs = glob(["*.py"], exclude=["*_test.py"]),
    visibility = ["//:__subpackages__"],
    deps = [
        "@pypi//stim",
        "@pypi//numpy",
        "@pypi//scipy",
    ],
)






