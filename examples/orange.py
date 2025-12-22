from usd_rerun_logger.usd_logger import UsdRerunLogger
import rerun as rr
from pathlib import Path
from pxr import Usd

rr.init("orange_example", spawn=True)
stage = Usd.Stage.Open(str(Path(__file__).parent / "assets/Orange001/Orange001.usda"))
logger = UsdRerunLogger(stage)
logger.log_stage()
