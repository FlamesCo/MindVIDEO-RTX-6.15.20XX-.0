# PyTorch imports
import torch
from torch import nn
from torch.utils.data import DataLoader

# Model classes
class TransformerEncoder(nn.Module):
    # PyTorch implementation of encoder
    
class TransformerDecoder(nn.Module):
    # PyTorch implementation of decoder
    
# Dataset and dataloader
train_dataset = MindVideoDataset() 
train_dataloader = DataLoader(train_dataset, batch_size=64)

# Initialize model, loss, optimizer 
model = Transformer(
    encoder=TransformerEncoder(), 
    decoder=TransformerDecoder()
).to("cpu")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for epoch in range(30):

  for batch in train_dataloader:
    optimizer.zero_grad()
    
    # PyTorch training steps
    outputs = model(batch)
    loss = criterion(outputs, batch[labels])
    
    loss.backward()
    optimizer.step()
    
  print(f"Epoch {epoch} loss: {loss.item()}")

# Save trained model
torch.save(model.state_dict(), "mindvideo_pytorch.pt")
