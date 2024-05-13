from pathlib import Path
from typing import Dict, List, Optional, Union
import pickle

import numpy as np
import pandas as pd

from ..dataset import KinodataDocked

class KinodataChemblKey:
    
    def __init__(
        self,
        dataset: KinodataDocked,
        df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.dataset = dataset
        self.df = df
        self.mapping = self.load()
        if self.mapping is None:
           self.mapping = self.generate_and_save() 
   
    @property
    def cache_file(self) -> Path:
        processed_dir = Path(self.dataset.processed_dir)
        return processed_dir / "chembl_key.pkl"
    
    def load(self) -> Optional[Dict]:
        if self.cache_file.exists():
            with open(self.cache_file, "rb") as f:
                return pickle.load(f)
        return None
    
    def generate_and_save(self):
        print("Generating kinodata chembl key, this might take a while...")
        if self.df is None:
            self.df = self.dataset.df
        get_ident = lambda d: d["ident"].item()
        get_aid = lambda ident: int(self.df[self.df["ident"] == ident]["activities.activity_id"].values[0])
        mapping = {
            get_aid(get_ident(data)): idx
            for idx, data in enumerate(self.dataset)
        }
        with open(self.cache_file, "wb") as f:
            pickle.dump(mapping, f)
        return mapping
        
            
    
    def __getitem__(self, activity_id: Union[int, np.int64, np.ndarray, List]):
        if isinstance(activity_id, int) or isinstance(activity_id, np.int64):
            return self.mapping[activity_id]
        elif isinstance(activity_id, np.ndarray) or isinstance(activity_id, list):
            index = np.empty_like(activity_id)
            for j, aid in enumerate(activity_id):
                index[j] = self.mapping[aid]
            return index