import pprint
from deep_vitabuild import utils

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

def main():

    ### initialize Trainer
    # simply get the main.py arguments as namespace ( like a dict)
    # The .cfg retrieve the corresponding path to the exp yaml
    cfg = utils.utils.parse_args().cfg
    print("ooooo", cfg)
 
    trainer = Trainer(cfg)

    ### copy yaml description file to the save folder
    utils.utils.copy_exp_file(trainer)

    ### copy proc.py file to the save folder
    utils.utils.copy_proc_file(trainer)

    trainer.logger.info(pprint.pformat(trainer.cfg))
    trainer.logger.info('#'*100)

    ### run the training procedure
    #trainer.run()


if __name__ == '__main__':

    main()