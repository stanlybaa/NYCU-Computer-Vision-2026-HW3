import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["MPLBACKEND"] = "Agg"

from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules


CONFIG_PATH = "cascade_mask_rcnn_cell.py"


def main():
    register_all_modules()

    cfg = Config.fromfile(CONFIG_PATH)
    cfg.work_dir = "work_dirs/cascade_cell"

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
