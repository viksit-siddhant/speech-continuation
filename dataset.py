import os
import torch
import torchaudio
import numpy as np
import transformers


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        os.makedirs("./dataset", exist_ok=True)
        self.dataset = torchaudio.datasets.LIBRISPEECH("./dataset", url="train-clean-100", download=True)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M")
        self.tokenizer.add_tokens(["[SOS]"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        waveform, sample_rate, text, *_ = self.dataset[idx]
        spectrogram = self._generate_spectrogram(waveform, sample_rate)
        first_half, second_half = self._split_spectrogram(spectrogram)
        text_labels = self._encode_text(text)
        return first_half, second_half, text_labels

    def _generate_spectrogram(self, waveform: np.ndarray, sample_rate: int) -> torch.Tensor:
        transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=128, n_fft=1024, hop_length=512)
        return transform(waveform.float())

    def _split_spectrogram(self, spectrogram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        half_length = spectrogram.shape[-1] // 2
        first_half = spectrogram[:, :, :half_length]
        second_half = spectrogram[:, :, half_length:]
        return first_half, second_half

    def _encode_text(self, text: str) -> torch.Tensor:
        sentence = f"[SOS] {text} {self.tokenizer.eos_token}"
        return torch.tensor(self.tokenizer.encode(sentence))