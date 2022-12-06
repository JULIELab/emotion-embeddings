from emocoder.experiments import utils as xutils

def test_get_experiments_config():
    config = xutils.get_experiments_config()
    assert isinstance(config["EMOTION_CODEC_PATH"], str)

def test_get_pretrained_emotion_codec():
    model, state_dict = xutils.get_pretrained_emotion_codec()
    pass