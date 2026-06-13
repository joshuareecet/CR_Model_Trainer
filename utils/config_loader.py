import configparser
from pathlib import Path

root_dir = Path(__file__).parent.parent
cfg_path = root_dir / "utils" / "config.cfg"

def write_config():
    with open(cfg_path,"w") as cfg_file:
        cfg_file.write(
        "[train.settings]"
        "\nMODEL = your-model-here"
        "\nDATASET = your-dataset-here"
        )

if not cfg_path.exists():
    write_config()

config = configparser.ConfigParser()
config.read(cfg_path)
config["train.settings"]["dataset"] = (Path(root_dir) / "datasets" / config["train.settings"]["dataset"]).as_posix()
TRAIN_CONFIG = config["train.settings"]