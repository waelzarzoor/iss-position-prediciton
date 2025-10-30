### - generate_dataset.py

This script utilizes the `GenerateDataset` class from the `modules.dataset_generator` module to generate new datasets for training, validation, and testing.

#### Usage

1. Run the script using the following command:

    ```bash
    python src/scripts/generate_dataset.py
    ```

2. Upon execution, the program collects data in real-time to generate new datasets for training, validation, and testing. Please note that the process may take some time due to real-time data collection.

3. The generated datasets can be utilized for model training in the `train_pipeline.py` file.

The script allows users to specify datasets to be generated using command-line arguments. Use the `-d` or `--datasets` flag followed by a list of datasets to be generated, each in the format `type:samples`. For example:

```bash
python src/scripts/generate_dataset.py -d user_train:700 user_val:200 user_test:100
```
If no datasets are specified, default datasets are generated: `user_train`  (700 samples), `user_val` (200 samples), and `user_test` (100 samples).