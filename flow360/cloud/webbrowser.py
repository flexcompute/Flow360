import webbrowser

from ..environment import Env


def open_browser(path):
    webbrowser.open(Env.current.get_web_real_url(path=path))
