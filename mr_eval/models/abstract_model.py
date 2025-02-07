import abc
import hashlib
import json
import os
from typing import List, Optional, Tuple, Type, TypeVar, Union

from loguru import logger as eval_logger
from tqdm import tqdm


T = TypeVar("T", bound="prm")


class prm(abc.ABC):
    def __init__(
        self,
        redundancy_threshold = 0.15,
        validity_threshold = 0.5,
        generation_config = {},
    ) -> None:
        self.redundancy_threshold = float(redundancy_threshold)
        self.validity_threshold = float(validity_threshold)
        self.set_generation_config(generation_config)
    
    def to(self, device: str) -> T:
        self.model = self.model.to(device)
        return self

    @abc.abstractmethod
    def respond(self, dataloader) -> None:
        pass
    
    def set_generation_config(self, generation_configs: dict) -> None:
        self.generation_config = generation_configs
        self.generation_config["max_length"] = generation_configs.get("max_length", 512)
        self.generation_config["temperature"] = generation_configs.get("temperature", 0.0)
        self.generation_config["top_k"] = generation_configs.get("top_k", 1)
        self.generation_config["top_p"] = generation_configs.get("top_p", 1.0)
        
 

    def get_generation_config(self) -> dict:
        try: 
            return self.generation_config
        except:
            return {}

