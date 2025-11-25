import logging
import os

local_rank = os.environ.get("LOCAL_RANK", -1)

logging.basicConfig(
    level=logging.INFO,
    format=f'[rank {local_rank}]' + '[\033[34m%(asctime)s\033[0m][%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class WrappedLogger():
    def __init__(self, name: str):
        """A warpped logger that allows rank 0 print control

        Args:
            name (str): name of the logger
        """
        self.logger = logging.getLogger(name)
        
    @staticmethod
    def maybe_not_rank0(kwargs):
        if kwargs.get("on_rank0", None) is not None:
            is_rank0_log = kwargs.pop("on_rank0")
            if is_rank0_log:
                return int(local_rank) in [0, -1]
            
        return True
        
    def log(self, *args, **kwargs):
        if self.maybe_not_rank0(kwargs):
            self.logger.log(*args, **kwargs)
                    
    def info(self, *args, **kwargs):
        if self.maybe_not_rank0(kwargs):
            self.logger.info(*args, **kwargs)
                    
    def warning(self, *args, **kwargs):
        if self.maybe_not_rank0(kwargs):
            self.logger.warning(*args, **kwargs)
                    
    def error(self, *args, **kwargs):
        if self.maybe_not_rank0(kwargs):
            self.logger.error(*args, **kwargs)                
