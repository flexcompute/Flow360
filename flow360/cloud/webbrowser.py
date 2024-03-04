"""
Web browser handling module
"""

import webbrowser

from ..environment import Env


def open_browser(path):
    """
    Open selected path in browser
    """
    webbrowser.open(Env.current.get_web_real_url(path=path))
