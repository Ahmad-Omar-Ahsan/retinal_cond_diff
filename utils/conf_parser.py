import yaml
import os
import torch
import sys


def get_config(config_file: str) -> dict:
    """Reads settings from config file.
    Args:
        config_file (str): YAML config file.
    Returns:
        dict: Dict containing settings.
    """

    with open(config_file, "r") as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)


    return base_config


if __name__ == "__main__":
    config = get_config(sys.argv[1])
    print("Using settings:\n", yaml.dump(config))