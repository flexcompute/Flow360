# Development Setup
0. clone repo
1. Install poetry
2. Install dependencies: ``poetry install``

## check in
1. ``poetry run pytest``
2. ``black .``
3. ``pylint flow360 --rcfile .pylintrc``


## install
``pip install flow360``

## client config api-key
You can set your *api-key* by the following methods:
1. Set globaly for your acount: ``flow360 configure`` will store *api-key* in ~/.flow360
2. In shell: ``export FLOW360_APIKEY="my api-key"``
3. In python script: ``os.environ["FLOW360_APIKEY"] = "my api-key"`` before `import flow360`


