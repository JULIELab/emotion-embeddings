import argparse

def main(gpu):
    from emocoder.experiments.mapping import baseline, multitask
    baseline.run_all_dev_exp(gpu)
    multitask.run_all_dev_exp(gpu) #proposed procedure in paper, singletask is omitted here

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Which gpu to run on", default=0, type=int)
    args = parser.parse_args()

    main(args.gpu)
