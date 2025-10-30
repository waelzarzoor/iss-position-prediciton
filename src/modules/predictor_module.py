import torch
from torch import nn
import lightning as pl

    
class LightningLatLongPredictor(pl.LightningModule):
    def __init__(self, hidden_units: int, learning_rate: float, features: int=2):
        super().__init__()

        self.save_hyperparameters('features', 'hidden_units', 'learning_rate')

        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()

        layers = [nn.Linear(features, hidden_units),
                  nn.Linear(hidden_units, hidden_units),
                  nn.Linear(hidden_units, features)]
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)
    
    def _shared_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)

        return loss, y, outputs
    
    def training_step(self, batch, batch_idx):
        loss, _, _= self._shared_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx)
        self.log('test_loss', loss)
        
        return {"test_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer