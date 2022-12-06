import argparse

def main(gpu):
    from emocoder.experiments.mapping import baseline, multitask
    baseline.run_all_test_exp(gpu)
    multitask.run_all_test_exp(gpu) #proposed procedure in paper, singletask is omitted here


    # # have to import one after the other, because experiments are dependent form each others output
    from emocoder.experiments.word import baseline, proposed, augmented, multitask
    baseline.run_all_test_exp(gpu)
    proposed.run_all_test_exp(gpu)
    augmented.run_all_test_exp(gpu)
    multitask.run_all_test_exp(gpu)
    from emocoder.experiments.word import zeroshotbaseline, zeroshot, augzero
    zeroshotbaseline.run_all_test_exp(gpu)
    zeroshot.run_all_test_exp(gpu)
    augzero.run_all_test_exp(gpu)


    from emocoder.experiments.text import baseline, proposed, augmented, multitask
    baseline.run_all_test_exp(gpu)
    proposed.run_all_test_exp(gpu)
    augmented.run_all_test_exp(gpu)
    multitask.run_all_test_exp(gpu)
    from emocoder.experiments.text import zeroshotbaseline, zeroshot, augzero
    zeroshotbaseline.run_all_test_exp(gpu)
    zeroshot.run_all_test_exp(gpu)
    augzero.run_all_test_exp(gpu)

    from emocoder.experiments.image import baseline, proposed, augmented, multitask
    baseline.run_all_test_exp(gpu)
    proposed.run_all_test_exp(gpu)
    augmented.run_all_test_exp(gpu)
    multitask.run_all_test_exp(gpu)
    from emocoder.experiments.image import zeroshotbaseline, zeroshot, augzero
    zeroshotbaseline.run_all_test_exp(gpu)
    zeroshot.run_all_test_exp(gpu)
    augzero.run_all_test_exp(gpu)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Which gpu to run on", default=0, type=int)
    args = parser.parse_args()

    main(args.gpu)

