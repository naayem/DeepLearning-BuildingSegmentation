import pprint

from deep_vitabuild import utils
from deep_vitabuild import core_delete

def main():

    ### initialize Trainer
    # simply get the main.py arguments as namespace ( like a dict)
    # The .cfg retrieve the corresponding path to the exp yaml
    cfg = utils.utils.parse_args().cfg
    print("ooooo", cfg)
 
    trainer = core_delete.trainer.Trainer(cfg)

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