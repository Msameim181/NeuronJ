from pip import main

from src.neuronj import NeuronJ


def main():
    NeuronJ(
        data_addr = "./data/rawData/X_DIV5",
        output_dir = "./data/dataSet/",
        colorize = True
    )


if __name__ == '__main__':
    main()
