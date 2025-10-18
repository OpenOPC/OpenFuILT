##
# @file   FuILT.py
# @author Shuo Yin
# @brief  Full-chip ILT engine
#
from .engine import Engine
from typing import Union, List
from openFuILT import JITFunction

from openFuILT.src.distribute import SimpleMaster
from openFuILT.src.distribute import SimpleWorker
from openFuILT.src.utils import get_logger

import traceback
logger = get_logger("FuILT")

class FuILT(Engine):
    def __init__(self, 
                 num_worker : int = None,
                 device_ids : Union[List[int], None] = None):
        '''
        FuILT main engine
        Args:
            num_worker (int, optional): number of workers. Defaults to None.
            device_ids (Union[List[int], None], optional): list of device ids. Defaults to
            None, which means all available devices.
        Raises:
            RuntimeError: if CUDA is not available.
        '''
        super().__init__(
            num_worker = num_worker,
            device_ids = device_ids
        )
        
    
    def solve(self, algo : JITFunction):
        '''
        Full-chip ILT solver
        Args:
            algo (JITFunction): JIT compiled ILT function
        Raises:
            RuntimeError: if CUDA is not available.
        '''
        self._initialize(algo)
        
        try:
        
            master : SimpleMaster = SimpleMaster(
                name="Master-Thread",
                patchDB = self.patchdb
            ) 
            
            master.start()
            master.join()
            
            for i in range(self.deviceHandler.getNumWorkers()):
                worker = SimpleWorker(
                    name = f"Worker-Thread-{i}",
                    pixel = self.pixel,
                    level = self.level,
                    micro_olrate=self.micro_olrate,
                    lr = self.lr,
                    epoch = self.epoch,
                    device = self.deviceHandler.getWorker(i),
                    task_queue = master.getTaskQueue(),
                    resultDB = self.maskdb,
                    ILT = algo.fn,
                    initializer=algo.initializer,
                    optimizer = self.optimizer,
                    verbose = self.verbose,
                )
                worker.start()
            
            master.getTaskQueue().join()
            
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Exception occurred: {e}")
            
    
            