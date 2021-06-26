"""
    Module for protein global vars
"""

# Module for recording name and sequence mapping
_GLOBAL_EMBED = list()

def set_name2seq(name, value):
    _GLOBAL_EMBED[name] = value

def get_global_embed():
    return _GLOBAL_EMBED
