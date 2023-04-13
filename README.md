# Flow360

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/flexcompute/Flow360/pypi-publish.yml)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/flexcompute/flow360/test.yml?label=tests)
[![PyPI version shields.io](https://img.shields.io/pypi/v/Flow360)](https://pypi.python.org/pypi/flow360/)

![](https://raw.githubusercontent.com/flexcompute/Flow360/main/img/Flow360-logo.svg)


# How to use Flow360 python client

## install
### Using pip (recommended)
``pip install flow360``

### install pre-release version
``pip install -U flow360 --pre``



## client config api-key
Get your *api-key* from [flow360.simulation.cloud](https://flow360.simulation.cloud)

<img src="https://user-images.githubusercontent.com/83596707/231739277-0f863e5a-b8b7-4f45-bd9b-6bfa32b8bdcb.gif" width="60%">

You can set your *api-key* by ONE of the following methods:
1. Set globaly for your acount: ``flow360 configure`` will store *api-key* in ~/.flow360
2. In shell: ``export FLOW360_APIKEY="my api-key"``
3. In python script: ``os.environ["FLOW360_APIKEY"] = "my api-key"`` before `import flow360`

## run examples:
1. Get examples from this repository:
    1. ``git clone https://github.com/flexcompute/Flow360.git``
    2. ``cd Flow360/examples``
2. run ``python case_results.py``


# Development
## setup
0. clone repo
1. Install poetry ``pip install poetry``
2. Install dependencies: ``poetry install``

## run examples
``poetry run python examples/case_results.py``

## check in
1. ``poetry run pytest -rA``
2. ``black .`` - performs auto-formatting
3. ``isort .`` - sorts imports
4. ``pylint flow360 --rcfile .pylintrc`` - checks code style
