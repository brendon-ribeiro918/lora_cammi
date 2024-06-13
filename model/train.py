import torch
import torch.optim as optim
from tqdm import tqdm
import os
from torch import nn

class Trainer:
    def __init__(self, model, train_loader, val_loader=None, num_epochs=10, learning_rate=1e-4, checkpoint_dir='checkpoints'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.checkpoint_dir = checkpoint_dir

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for inputs, targets in tqdm(self.train_loader):
                inputs, targets = inputs.to("cuda"), targets.to("cuda")
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}")

            self.save_checkpoint(epoch, avg_loss)
            self.log_metrics(epoch, avg_loss)
    
    # Save model checkpoints
    def save_checkpoint(self, epoch, loss):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)

    # Log metrics to text file
    def log_metrics(self, epoch, loss):
        with open(os.path.join(self.checkpoint_dir, 'training_log.txt'), 'a') as f:
            f.write(f'Epoch {epoch+1}, Loss: {loss:.4f}\n')
