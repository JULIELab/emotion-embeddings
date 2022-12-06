from emocoder.experiments.mapping import multitask
from emocoder.src.utils import get_project_root

def test_mapping_multitask():
    exp = multitask.EXPERIMENTS["full"]
    exp.parent_dir = get_project_root() / "emocoder" / "tests" / "target"
    exp.epochs = 1
    exp.run()