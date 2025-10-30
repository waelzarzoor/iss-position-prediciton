import argparse
from src.modules.dataset_generator import GenerateDataset

def generate_datasets(datasets):
    for dataset in datasets:
        type, samples = dataset
        GenerateDataset(type=type, samples=samples).save_as_csv()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate datasets for training, validation, and testing.')

    parser.add_argument('-d', '--datasets', nargs='+', type=str,
                        help='List of datasets to generate, each dataset should be in the format type:samples. '
                             'Example: -d user_train:700 user_val:200 user_test:100')

    args = parser.parse_args()

    if args.datasets is None:
        datasets = [['user_train', '700'], ['user_val', '200'], ['user_test', '100']]
    else:
        datasets = [dataset.split(':') for dataset in args.datasets]

    datasets = [(dataset[0], int(dataset[1])) for dataset in datasets]

    generate_datasets(datasets)