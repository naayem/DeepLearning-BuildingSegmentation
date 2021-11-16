from deep_vitabuild.utils import utils, config_utils
from deep_vitabuild import core
from deep_vitabuild.procedures import inferences_detectron, train_detectron, valid_detectron

def main():
    TRAIN = 0
    VALID = 0
    INFERENCES = 1

    gen_cfg = utils.parse_args().cfg
    gen_cfg = config_utils.get_config(gen_cfg) #Parses the cfg.yaml file in dict of dict fashion
    final_output_dir = utils.init_exp_folder(gen_cfg)
    logger, log_filename = utils.create_logger(final_output_dir, gen_cfg)

    print(tuple(gen_cfg.DETECTRON.CATALOG))
    print(gen_cfg.TRAINING)

    building_metadata, building_val_metadata = train_detectron.add_to_catalog(gen_cfg)
    dataset_dicts = train_detectron.get_building_dicts(gen_cfg.TRAINING.DATASET_DIR+'/train')
    detec_cfg = train_detectron.cfg_detectron(gen_cfg)

    print(detec_cfg.OUTPUT_DIR)
    
    if TRAIN:
        train_detectron.train_detectron(detec_cfg)
    
    if VALID:
        detec_cfg, trainer = valid_detectron.add_val_loss(detec_cfg)
        detec_cfg = valid_detectron.inference_val(detec_cfg, building_metadata)
        valid_detectron.evaluate_AP(detec_cfg, trainer)

    if INFERENCES:
        if gen_cfg.INFERENCE.STRUCTURE == 'folder':
            inferences_detectron.inference_detectron_folder(detec_cfg, gen_cfg, building_metadata)
        if gen_cfg.INFERENCE.STRUCTURE == 'full':
            inferences_detectron.inference_detectron_full(detec_cfg, gen_cfg, building_metadata)

    #utils.count_dataset(DATASET_DIR)

if __name__ == '__main__':
    main()