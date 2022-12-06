from emocoder.experiments.text import baseline
from emocoder.src.utils import get_project_root


def run_text_baseline(exp_name):
    exp = baseline.experiments[exp_name]
    exp.parent_dir = get_project_root() / "emocoder" / "tests" / "target"
    exp.epochs = 1
    exp.run()





def test_text_baseline_emobank():
    run_text_baseline("emobank")

def test_text_baseline_cvat():
    run_text_baseline("cvat")

def test_text_baseline_afftext():
    run_text_baseline("afftext_be5")

def test_text_baseline_afftext():
    run_text_baseline("afftext_be6")

def test_text_baseline_sst():
   run_text_baseline("sst2cls")

def test_text_baseline_isear():
    run_text_baseline("isear")


def test_text_baseline_ssec():
    run_text_baseline("ssec")


