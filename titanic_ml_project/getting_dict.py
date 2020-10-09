#!/usr/bin/env python
"""
This module has the get_dict(file) method
"""

import yaml

# ******************************************* EDA ***********************************************
def get_dict(file="setup_models.yml"):
    """
    * file - file to get dict

    Return the dict
    """
    with open(file, "r") as f:
        stream = f.read()
    return yaml.safe_load(stream)
