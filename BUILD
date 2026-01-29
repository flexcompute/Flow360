load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")

# =============================================================================
# Flow360 Python Client Library
# =============================================================================

py_library(
    name = "flow360",
    srcs = glob(["flow360/**/*.py"]),
    data = glob([
        "flow360/plugins/report/fonts/**",
        "flow360/plugins/report/img/**",
        "flow360/examples/**/*.json",
        "flow360/examples/**/*.csm",
        "flow360/examples/**/*.egads",
        "flow360/examples/**/*.yaml",
        "flow360/examples/**/*.xrotor",
        "flow360/examples/**/*.x_t",
    ]),
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "@pip//boto3",
        "@pip//click",
        "@pip//h5py",
        "@pip//matplotlib",
        "@pip//numexpr",
        "@pip//numpy",
        "@pip//pandas",
        "@pip//prettyprinttree",
        "@pip//pydantic",
        "@pip//pylatex",
        "@pip//pyyaml",
        "@pip//requests",
        "@pip//rich",
        "@pip//toml",
        "@pip//unyt",
        "@pip//wcmatch",
        "@pip//zstandard",
    ],
)

# CLI binary
py_binary(
    name = "flow360_cli",
    srcs = ["flow360/cli/app.py"],
    main = "flow360/cli/app.py",
    visibility = ["//visibility:public"],
    deps = [":flow360"],
)
