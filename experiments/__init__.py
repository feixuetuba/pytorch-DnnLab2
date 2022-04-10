from utils import load_obj


def load_experiment(cfg):
    experiment = cfg['experiment']
    cls = "GeneralizedModel"
    pkg = f"experiments.{experiment}.{cls}"
    return load_obj(pkg, cls)(cfg)
