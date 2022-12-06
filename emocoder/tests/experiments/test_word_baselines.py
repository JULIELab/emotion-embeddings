from emocoder.experiments.word import baseline
from emocoder.src.utils import get_project_root


def run_word_baseline(exp_name):
    exp = baseline.experiments[exp_name]
    exp.embedding_limit = 10 * 1000  # reducing embedding limits
    exp.parent_dir = get_project_root() / "emocoder" / "tests" / "target"   # set dir path to trial
    exp.epochs = 1
    exp.run()




def test_word_baseline_anew1999():
    run_word_baseline("anew1999")

def test_word_baseline_stevenson2007():
    run_word_baseline("stevenson2007")

def test_word_baseline_xanew():
    run_word_baseline("xanew")

def test_word_baseline_nrc():
    run_word_baseline("nrc")

def test_word_baseline_stadthagen_va():
    run_word_baseline("stadthagen_va")

def test_word_baseline_stadthagen_be():
    run_word_baseline("stadthagen_be")

def test_word_baseline_moors():
    run_word_baseline("moors")

def test_word_baseline_vo():
    run_word_baseline("vo")

def test_word_baseline_briesemeister():
    run_word_baseline("briesemeister")

def test_word_baseline_imbir():
    run_word_baseline("imbir")

def test_word_baseline_riegel():
    run_word_baseline("riegel")

def test_word_baseline_wierzba():
    run_word_baseline("wierzba")

def test_word_baseline_kapucu_va():
    run_word_baseline("kapucu_va")

def test_word_baseline_kapucu_be():
    run_word_baseline("kapucu_be")


