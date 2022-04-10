from experiments.detectron2.meta_arch.build import build_model
def load_experiment(cfg):
    return build_model(cfg)