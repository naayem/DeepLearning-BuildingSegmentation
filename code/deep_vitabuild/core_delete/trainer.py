from torch import device as torch_device
from ..utils import utils, config_utils
from ..models import models_common

class Trainer():
    '''
    Class, which runs the training procedure.
    Also keep all necessary components together:
            models, 
            optimizers, 
            schedulers, 
            datasets,
            dataloaders,
            losses,
            procedures.
    This universal module is developed to run deep learning experiments of any kind.
    '''
    def __init__(self, cfg): # initialized by <configname>.yaml file
        self.cfg = config_utils.get_config(cfg) #Parses the cfg.yaml file in dict of dict fashion
        self.final_output_dir = utils.init_exp_folder(self.cfg)
        self.logger, self.log_filename = utils.create_logger(self.final_output_dir, self.cfg)

        self.gpus = [int(i) for i in self.cfg.GPUS.split(',')]
        self.device0 = torch_device(f'cuda:{self.gpus[0]}')
        self.models = models_common.get_models_by_config(self.cfg, self.gpus, self.device0)
    
    def run(self):
        '''Runs the training procedure specified in "self.proc". 
        '''
        for model in self.models:
            model.run()
        self.logger.info(f'\nTraining of "{self.cfg.EXP_NAME}" experiment is finished. \n')