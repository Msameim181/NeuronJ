from pip import main

from src.neuronj import NeuronJ


def main():
    NeuronJ(
        data_addr = "./data/rawData/X_DIV5",
        output_dir = "./data/dataSet/",
        colorize = True,
        mask_builder='matplotlib',
        resize_mask_to_image_size = True,
        resize_lib='pillow',
        save_data_dpi=400
    )


if __name__ == '__main__':
    main()
