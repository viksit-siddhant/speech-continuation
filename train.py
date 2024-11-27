from model import Spectron
from dataset import LibriSpeechDataset
import torch
import torch.nn as nn
import os
from tqdm import tqdm

epochs = 20
val_split = 0.1
learning_rate = 1e-4
data_dir = "./experiment"
device = torch.device('cuda')

def create_dirs():
    os.makedirs(data_dir, exist_ok=True)
    checkpoint_dir = os.path.join(data_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    result_dir = os.path.join(data_dir, "results")
    os.makedirs(result_dir, exist_ok=True)
    return checkpoint_dir, result_dir

if __name__ == '__main__':
    dataset = LibriSpeechDataset()
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [1- val_split, val_split])
    model = Spectron(80)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ckpt_dir, res_dir = create_dirs()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        for batch in tqdm(train_dataset):
            prompt, continuation, text_labels = batch
            prompt, continuation, text_labels = prompt.to(device), continuation.to(device), text_labels.to(device)
            with torch.amp.autocast():
                optimizer.zero_grad()
                loss = model(prompt, continuation, text_labels)
                loss.backward()
                optimizer.step()
    