from ._version import __version__

def ensure_tmp_dir():
    """ Ensure tmp dir exists. """
    from os import path, mkdir

    if not path.exists('~/tmp/'):
        import os
        mkdir('~/tmp/')

ensure_tmp_dir()
