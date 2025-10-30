import pytest
from itertools import islice
from torch.utils.data import DataLoader
from src.modules.dataset_module import LatLongDataset, LightningLatLongDatamodule

@pytest.fixture
def datamodule():
    return LightningLatLongDatamodule(
        train_csv='datasets/train_dataset.csv',
        val_csv='datasets/val_dataset.csv',
        test_csv='datasets/test_dataset.csv',
        batch_size=128,
        sequence_length=4
        )

def test_setup(datamodule):
    datamodule.setup('fit')
    assert isinstance(datamodule.train_dataset, LatLongDataset)
    assert isinstance(datamodule.val_dataset, LatLongDataset)

def test_train_val_dataloader(datamodule):
    datamodule.setup('fit')
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(val_dataloader, DataLoader)

def test_test_dataloader(datamodule):
    datamodule.setup('test')
    test_dataloader = datamodule.test_dataloader()
    assert isinstance(test_dataloader, DataLoader)

def test_batching(datamodule):
    batch_size = datamodule.batch_size
    sequence_length = datamodule.sequence_length

    datamodule.setup('fit')
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    for batch in islice(train_dataloader, len(train_dataloader) - 1):
        assert batch[0].shape[0] == batch_size
        assert batch[0].shape[1] == sequence_length

    for batch in islice(val_dataloader, len(val_dataloader) - 1):
        assert batch[0].shape[0] == batch_size
        assert batch[0].shape[1] == sequence_length

    datamodule.setup('test')
    test_dataloader = datamodule.test_dataloader()

    for batch in islice(test_dataloader, len(test_dataloader) - 1):
        assert batch[0].shape[0] == batch_size
        assert batch[0].shape[1] == sequence_length