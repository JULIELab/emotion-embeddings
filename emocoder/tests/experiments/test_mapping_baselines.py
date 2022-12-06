from emocoder.experiments.mapping import baseline
from emocoder.src.utils import  get_project_root


def run_mapping_baseline(exp_name):
    exp = baseline.experiments[exp_name]
    exp.parent_dir = get_project_root() / "emocoder" / "tests" / "target"
    exp.epochs = 1
    exp.run()


def test_mapping_baseline_anew_stevenson_baseline_vad_to_be5():
    run_mapping_baseline("anew_stevenson_baseline_vad_to_be5")

def test_mapping_baseline_anew_stevenson_baseline_be5_to_vad():
    run_mapping_baseline("anew_stevenson_baseline_be5_to_vad")

def test_mapping_baseline_anew_stevenson_baseline_va_to_be5():
    run_mapping_baseline("anew_stevenson_baseline_va_to_be5")

def test_mapping_baseline_anew_stevenson_baseline_be5_to_va():
    run_mapping_baseline("anew_stevenson_baseline_be5_to_va")

def test_mapping_baseline_xanew_nrc_baseline_vad_to_nrc():
    run_mapping_baseline("xanew_nrc_baseline_vad_to_nrc")

def test_mapping_baseline_xanew_nrc_baseline_nrc_to_vad():
    run_mapping_baseline("xanew_nrc_baseline_nrc_to_vad")

def test_mapping_baseline_FER_BE_VAD():
    run_mapping_baseline("FER_BE_VAD")

def test_mapping_baseline_FER_VAD_BE():
    run_mapping_baseline("FER_VAD_BE")

def test_mapping_baseline_AffectNet_BE_VA():
    run_mapping_baseline("AffectNet_BE_VA")

def test_mapping_baseline_AffectNet_VA_BE():
    run_mapping_baseline("AffectNet_VA_BE")