import argparse

def main(gpu=0):
    # have to import one after the other, because experiments are dependent form each others output

    from emocoder.experiments.word import baseline
    baseline.run_all_dev_exp(gpu)
    from emocoder.experiments.word  import zeroshotbaseline
    zeroshotbaseline.run_all_dev_exp(gpu)

    from emocoder.experiments.text import baseline
    baseline.run_all_dev_exp(gpu)
    from emocoder.experiments.text import zeroshotbaseline
    zeroshotbaseline.run_all_dev_exp(gpu)


    from emocoder.experiments.image import baseline
    baseline.run_all_dev_exp(gpu)
    from emocoder.experiments.image import zeroshotbaseline
    zeroshotbaseline.run_all_dev_exp(gpu)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Which GPU to run the experiments on.", type=int, default=0)
    args = parser.parse_args()
    main(gpu=args.gpu)