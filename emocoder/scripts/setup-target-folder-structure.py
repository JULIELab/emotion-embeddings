from emocoder.experiments import  constants

def main():
    for d in constants.EXPERIMENT_DIRS:
        for s in ["dev", "test"]:
            p = d / s
            if not p.exists():
                p.mkdir(parents=True)



if __name__ == "__main__":
    main()