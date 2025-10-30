### - train_pipeline.py

This script defines a PyTorch Lightning training pipeline for a machine learning model using the `LightningLatLongPredictor` model, `LightningLatLongDatamodule` data module, and various PyTorch Lightning functionalities.

#### Usage

1. Run the script:

    ```bash
    python src/scripts/train_pipeline.py
    ```

#### Training Pipeline

The `train_pipeline` function performs the following steps:

1. Initializes a Lightning data module (`LightningLatLongDatamodule`) with training, validation, and testing datasets from specified CSV files (`TRAIN_DATASET_PATH`, `VAL_DATASET_PATH`, `TEST_DATASET_PATH`) and a specified batch size (`BATCH_SIZE`).

2. Initializes a Lightning model (`LightningLatLongPredictor`).

3. Configures a PyTorch Lightning Trainer with the following parameters:
   - `max_epochs`: Maximum number of training epochs, default is 50.
   - `accelerator`: The accelerator is set to 'auto', allowing PyTorch Lightning to automatically choose the appropriate accelerator device based on the available hardware (CPU or GPU).
   - `callbacks`: Utilizes EarlyStopping callback to monitor the 'val_loss' and stop training if it does not improve within a patience of 5 epochs. Additionally, it uses the ModelCheckpoint callback to save the best model based on the 'val_loss' during training.

4. Fits the Lightning model to the training data using the provided data module.

5. Evaluates the trained model on the test data and prints the Mean Squared Error (MSE).

Users can modify the behavior of the training pipeline by specifying the following command-line arguments:

- `--use_user_datasets`: Use datasets provided by the user instead of default datasets.
- `--hidden_units`: Number of hidden units in the neural network, default is 16.
- `--learning_rate`: Learning rate for training, default is 0.001.
- `--batch_size`: Batch size for training, default is 128.
- `--sequence_length`: Sequence length for training, default is 1.
- `--max_epochs`: Maximum number of training epochs, default is 50.

Example usage:

```bash
python src/scripts/train_pipeline.py --use_user_datasets --hidden_units 32 --learning_rate 0.01 --batch_size 256
```
This command will train the model using user-provided datasets, with 32 hidden units in the neural network, a learning rate of 0.01, and a batch size of 256.