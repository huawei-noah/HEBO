from pathlib import Path

# Try to locate the GP training/opt scripts
try:
    GP_TRAIN_FILE = str(Path(__path__[0]) / "gp_train.py")
    GP_OPT_FILE = str(Path(__path__[0]) / "gp_opt.py")
except:
    pass
