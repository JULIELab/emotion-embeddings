from emocoder.experiments.image import baseline
from emocoder.src import utils



def run_baseline(key):
    exp = baseline.EXPERIMENTS[key]
    exp.epochs = 1
    exp.parent_dir = utils.get_project_root() / "emocoder" / "tests" / "target"
    exp.run()

def test_iaps_baseline():
    run_baseline("iaps")

def test_you_baseline():
    run_baseline("flickr")

def test_fer_baseline():
    run_baseline("fer_be")


def test_fer_vad_baseline():
    run_baseline("fer_vad")

# Testing on Affectnet would take too long because the dataset is huge