import os
import json
from pathlib import Path
CONFIG_DIR = Path(__file__).resolve().parent
CODE_DIR = CONFIG_DIR.parent

class Config:
    def read_cfg(self, config_path):
        with open(os.path.join(CONFIG_DIR, config_path), 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_cfg(self):
        return self.cfg
    

class Dataset_Config(Config):
    def __init__(self):
        self.cfg = self.read_cfg('dataset_config.json')
        
    def get_dataset_root(self, name):
        return self.cfg[name]["dataroot"] if name in self.cfg.keys() else name