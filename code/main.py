import pprint

from torch._C import device
from deep_vitabuild import core, utils, models
from torch import device as torch_device

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
        self.cfg = utils.config_utils.get_config(cfg) #Parses the cfg.yaml file in dict of dict fashion
        self.final_output_dir = utils.utils.init_exp_folder(self.cfg)
        self.logger, self.log_filename = utils.utils.create_logger(self.final_output_dir, self.cfg)

        self.gpus = [int(i) for i in self.cfg.GPUS.split(',')]
        self.device0 = torch_device(f'cuda:{self.gpus[0]}')
        self.models = models.models_common.get_models_by_config(self.cfg, self.gpus, self.device0)
    
    def run(self):
        '''Runs the training procedure specified in "self.proc". 
        '''
        for model in self.models:
            model.run()
        self.logger.info(f'\nTraining?? of "{self.cfg.EXP_NAME}" experiment is finished. \n')


def main():

    ### initialize Trainer
    # simply get the main.py arguments as namespace ( like a dict)
    # The .cfg retrieve the corresponding path to the exp yaml
    cfg = utils.utils.parse_args().cfg
    print("ooooo", cfg)
 
    trainer = core.trainer.Trainer(cfg)

    ### copy yaml description file to the save folder
    utils.utils.copy_exp_file(trainer)

    ### copy proc.py file to the save folder
    utils.utils.copy_proc_file(trainer)

    trainer.logger.info(pprint.pformat(trainer.cfg))
    trainer.logger.info('#'*100)

    print(trainer.cfg.MODELS.model1.PARAMS.WEIGHT_PATH)

    ### run the training procedure
    #trainer.run()


if __name__ == '__main__':

    main()