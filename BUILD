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
        "@pip//pillow",
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

# CLI binary - uses generated wrapper to invoke the click entry point
genrule(
    name = "flow360_cli_main",
    outs = ["flow360_cli_main.py"],
    cmd = "echo 'from flow360.cli import flow360; flow360()' > $@",
)

py_binary(
    name = "flow360_cli",
    srcs = [":flow360_cli_main"],
    main = "flow360_cli_main.py",
    visibility = ["//visibility:public"],
    deps = [":flow360"],
)
