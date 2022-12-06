from emocoder.src import models, utils
import copy

def test_compare_state_dicts():

    model1 = models.EmotionCodec(100, None, 0.)
    model1.enc["enc1"] = models.WordFFN(num_outputs=100, scaling="logits")
    model1.enc["enc2"] = models.WordFFN(num_outputs=100, scaling="logits")
    model1.dec["dec1"] = models.LinearDecoder(100, 1)

    model2 = models.EmotionCodec(100, None, 0.)
    model2.enc["enc1"] = models.WordFFN(num_outputs=100, scaling="logits")
    model2.enc["enc2"] = models.WordFFN(num_outputs=100, scaling="logits")
    model2.dec["dec1"] = models.LinearDecoder(100, 1)

    model3 = models.EmotionCodec(100, None, 0.)
    model3.enc["enc1"] = copy.deepcopy(model1.enc["enc1"])
    model3.enc["enc2"] = models.WordFFN(num_outputs=100, scaling="logits")
    model3.dec["dec1"] = copy.deepcopy(model1.dec["dec1"])
    model3.dec["dec2"] = models.LinearDecoder(100, 1)

    # Scenario 1: exactly matching keys, no matching parameters
    rt = utils.compare_state_dicts(model1.state_dict(), model2.state_dict())
    assert len(rt["matching"]) == 0
    assert len(rt["only sd1"]) == 0
    assert len(rt["only sd2"]) == 0
    assert len(rt["not matching"]) == len(model1.state_dict())

    # Scenario 2: everything matches exactly
    rt = utils.compare_state_dicts(model1.state_dict(), copy.deepcopy(model1).state_dict())
    assert len(rt["matching"]) == len(model1.state_dict())
    assert len(rt["only sd1"]) == 0
    assert len(rt["only sd2"]) == 0
    assert len(rt["not matching"]) == 0

    # Scenario 3: some things match others don't
    rt = utils.compare_state_dicts(model1.state_dict(), model3.state_dict())
    assert len(rt["matching"]) == 7
    assert len(rt["not matching"]) == 6
    assert len(rt["only sd1"]) == 0
    assert len(rt["only sd2"]) == 1


