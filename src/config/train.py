from config.var import BASE_DIR

DATA_DIR = BASE_DIR / 'data'
TRAIN_DATA = DATA_DIR / 'train'
VAL_DATA = DATA_DIR / 'validation'
VIS_DATA = DATA_DIR / 'visualization'

DATA_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DATA.mkdir(parents=True, exist_ok=True)
VAL_DATA.mkdir(parents=True, exist_ok=True)
VIS_DATA.mkdir(parents=True, exist_ok=True)
