##
# @file   engine.py
# @author Shuo Yin
#
import torch
import numpy as np

from abc import ABC, abstractmethod
from openFuILT.src.utils import DeviceHandler
from openFuILT.src import Layout
from openFuILT.src.database import MaskDB
from openFuILT.src.database import PatchDB
from openFuILT import JITFunction
from openFuILT.src.partition import MaskStitcher

from typing import Callable, List

class Engine(ABC):
    def __init__(self,
                 num_worker : int = None,
                 device_ids = None) -> None:
        
        '''
        Engine for FuILT
        Args:
            num_worker (int, optional): number of workers. Defaults to None.
            device_ids (Union[List[int], None], optional): list of device ids. Defaults to
            None, which means all available devices.
        Raises:
            RuntimeError: if CUDA is not available.
        '''
        
         #device 
        self.deviceHandler = DeviceHandler(
            num_worker=num_worker,
            device_ids=device_ids
        )
        
        self.verbose = False
    
    @staticmethod
    def writePickle(mask : np.ndarray, file_path : str):
        '''
        write mask to pickle file
        Args:
            mask (np.ndarray): mask to be written
            file_path (str): path to the pickle file
        '''
        import pickle
        with open(f"{file_path}.pkl", 'wb') as f:
            pickle.dump(mask, f)
            

    def set_layout(self,
                 layout_path : str,
                 layer : int):
        '''
        set layout from gds file
        Args:
            layout_path (str): path to gds file
            layer (int): layer number
        Raises:
            FileNotFoundError: if gds file is not found
        '''
        # layout
        self.layout : Layout = Layout(
            layer=layer,
            filePath=layout_path
        )
        return self
        
    def set_epoch(self, epoch : int):
        '''
        set number of epochs
        Args:
            epoch (int): number of epochs
        '''
        self.epoch = epoch
        return self
        
    def set_lr(self, lr : float):
        '''
        set learning rate
        Args:
            lr (float): learning rate
        '''
        self.lr = lr
        return self
        
    def set_optimizer(self, optimizer : Callable):
        '''
        set optimizer
        Args:
            optimizer (Callable): optimizer function, e.g., torch.optim.SGD
        '''
        self.optimizer = optimizer
        return self

    def set_verbose(self):
        '''
        set verbose mode
        '''
        self.verbose = True
        return self
        
    def set_pixel(self, pixel : int):
        '''
        set pixel size
        Args:
            pixel (int): pixel size
        '''
        self.pixel = pixel
        return self

    def set_macro_params(self,
                 macro_size : List[int],
                 macro_olrate : float):
        '''
        set macro partitioner parameters
        Args:
            macro_size (List[int]): macro partitioner size [w, h]
            macro_olrate (float): macro partitioner overlap rate
        '''
        self.macro_size = macro_size
        self.macro_olrate = macro_olrate
        return self
        
    
    def set_micro_params(self,
                        level : int,
                        micro_olrate : float):
        '''
        set micro ILT parameters
        Args:
            level (int): micro ILT level
            micro_olrate (float): micro ILT overlap rate
        '''
        
        self.level = level
        self.micro_olrate = micro_olrate
        return self
    
    # register variables
    def _initialize(self, algo : JITFunction):
        self.maskdb : MaskDB = MaskDB()
        self.maskdb.set_biniarizer(algo.binarizer)
        self.patchdb : PatchDB = PatchDB(
            pixel = self.pixel,
            size = self.macro_size,
            olrate = self.macro_olrate,
            level = self.level,
            layout = self.layout
        )
        self.stitcher = MaskStitcher(self.patchdb, self.maskdb)
        
    def getFullMask(self) -> np.ndarray:
        '''
        get full chip mask
        Returns:
            np.ndarray: full chip mask
        '''
        return self.stitcher.stitch()
    
    def getOriginFullMask(self) -> np.ndarray:
        '''
        get full chip origin mask
        Returns:
            np.ndarray: full chip origin mask
        '''
        return self.stitcher.stitch(origin=True)
    
    def writeTargetGDS(self, file_path : str):
        '''
        write target gds file
        Args:
            file_path (str): path to save gds file
        '''
        from openFuILT.src.utils import GDSConverter
        gds_converter = GDSConverter(
            scaled_bbox = self.patchdb.getLayoutAdapter().bbox,
            polygons = self.patchdb.getLayoutAdapter().polygons,
            file_path = file_path
        )
        gds_converter.writeGDS()
        
    @abstractmethod
    def solve(self,  algo : JITFunction):
        raise NotImplementedError("solve is not implemented")
    