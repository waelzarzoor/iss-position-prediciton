import argparse
import torch
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from src.modules.dataset_module import LightningLatLongDatamodule
from src.modules.predictor_module import LightningLatLongPredictor

torch.manual_seed(10)

TRAIN_DATASET_PATH = 'datasets/test_dataset.csv'
VAL_DATASET_PATH = 'datasets/val_dataset.csv'
TEST_DATASET_PATH = 'datasets/test_dataset.csv'

USER_TRAIN_DATASET_PATH = 'datasets/user_train_dataset.csv'
USER_VAL_DATASET_PATH = 'datasets/user_val_dataset.csv'
USER_TEST_DATASET_PATH = 'datasets/user_test_dataset.csv'

BATCH_SIZE = 128
SEQUENCE_LENGTH = 1
MAX_EPOCHS = 50
HIDDEN_UNITS = 16
LEARNING_RATE = 0.001


def train_pipeline(
        train_dataset_path: str,
        val_dataset_path: str,
        test_dataset_path: str,
        batch_size: int,
        sequence_length: int, 
        max_epochs: int,
        hidden_units: int,
        learning_rate: float
    ):
    
    '''
#     Train a PyTorch Lightning model using a custom training pipeline.

#     This function performs the following steps:
#     1. Creates a datamodule with training, validation, and testing datasets.
#     2. Initializes a Lightning model (`LightningLatLongPredictor`).
#     3. Configures a PyTorch Lightning Trainer with the following parameters:
#        - `max_epochs`: Maximum number of training epochs (50 in this case).
#        - `accelerator`: The accelerator is set to 'auto', allowing PyTorch Lightning to automatically choose the appropriate accelerator device based on the available hardware (CPU or GPU).
#        - `callbacks`: Utilizes EarlyStopping callback to monitor the 'val_loss' and stop training if it does not improve within a patience of 5 epochs. 
                        Additionally, it uses the ModelCheckpoint callback to save the best model based on the 'val_loss' during training.
#        - `logger`: The logger is set to False, indicating that no logging will be performed during training.

#     4. Fits the Lightning model to the training data using the provided datamodule.
#     5. Evaluates the trained model on the test data and prints the Mean Squared Error (MSE).
#     '''

    datamodule = LightningLatLongDatamodule(
        train_csv=train_dataset_path,
        val_csv=val_dataset_path, 
        test_csv=test_dataset_path, 
        batch_size=batch_size,
        sequence_length=sequence_length
    )
    
    datamodule.setup(stage='train')

    lit_model = LightningLatLongPredictor(
        hidden_units=hidden_units, 
        learning_rate=learning_rate
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        min_delta=0.001,
        patience=5
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val_loss',
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs, 
        accelerator='auto', 
        callbacks=[early_stopping, checkpoint_callback],
        logger=True
    )

    trainer.fit(lit_model, datamodule=datamodule)

    datamodule.setup(stage='test')

    test_mse = trainer.test(datamodule=datamodule, ckpt_path='best')[0]


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a PyTorch Lightning model using a custom training pipeline.')

    parser.add_argument('--use_user_datasets', action='store_true',
                        help='Use datasets provided by the user instead of default datasets')
    parser.add_argument('--hidden_units', type=int, default=HIDDEN_UNITS,
                        help=f'Number of hidden units in the neural network, default: {HIDDEN_UNITS}')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help=f'Learning rate for training, default: {LEARNING_RATE}')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size for training, default: {BATCH_SIZE}')
    parser.add_argument('--sequence_length', type=int, default=SEQUENCE_LENGTH,
                        help=f'Sequence length for training, default: {SEQUENCE_LENGTH}')
    parser.add_argument('--max_epochs', type=int, default=MAX_EPOCHS,
                        help=f'Maximum number of training epochs, default: {MAX_EPOCHS}')

    args = parser.parse_args()

    if args.use_user_datasets:
        train_dataset_path = USER_TRAIN_DATASET_PATH
        val_dataset_path = USER_VAL_DATASET_PATH
        test_dataset_path = USER_TEST_DATASET_PATH
    else:
        train_dataset_path = TRAIN_DATASET_PATH
        val_dataset_path = VAL_DATASET_PATH
        test_dataset_path = TEST_DATASET_PATH

    train_pipeline(
        train_dataset_path, 
        val_dataset_path, 
        test_dataset_path,
        args.batch_size, 
        args.sequence_length, 
        args.max_epochs,
        args.hidden_units,
        args.learning_rate
    )