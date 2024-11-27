from model import Spectron
from dataset import LibriSpeechDataset, LibriSpeechDataLoader
import torch
import torch.nn as nn
import os
from tqdm import tqdm

epochs = 20
val_split = 0.1
learning_rate = 1e-4
data_dir = "./experiment"
val_size = 1
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
    dataset_length = len(dataset)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [dataset_length-val_size, val_size])
    train_loader = LibriSpeechDataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = LibriSpeechDataLoader(val_dataset, batch_size=1, shuffle=False)
    model = Spectron(80)
    model = torch.compile(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ckpt_dir, res_dir = create_dirs()
    result_file = os.path.join(res_dir, "results.txt")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        for batch in tqdm(train_loader):
            prompt, continuation, text_labels = batch
            prompt, continuation, text_labels = prompt.to(device), continuation.to(device), text_labels.to(device)
            total_loss = 0
            with torch.amp.autocast():
                optimizer.zero_grad()
                transcript, continuation_pred = model(prompt, continuation, text_labels)
                loss = model.compute_loss(transcript, continuation_pred, text_labels, continuation)
                loss.backward()
                with torch.no_grad():
                    total_loss += loss.item()
            optimizer.step()
        print(f"Training loss: {total_loss}")
        with open(result_file, "a") as f:
            f.write(f"Epoch {epoch + 1}\n")
        for batch in tqdm(val_loader):
            prompt, _, _ = batch
            prompt = prompt.to(device)
            model.eval()
            with torch.amp.autocast():
                transcript,_ = model.generate(prompt)
            with open(result_file, "a") as f:
                f.write(transcript)


