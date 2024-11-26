import torch
import torchaudio
import numpy as np
import transformers

class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.tokenizer.add_tokens(["[SOS]"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        sample = self.dataset[idx]
        waveform, sample_rate, text = self._parse_sample(sample)
        spectrogram = self._generate_spectrogram(waveform, sample_rate)
        first_half, second_half = self._split_spectrogram(spectrogram)
        text_labels = self._encode_text(text)
        return first_half, second_half, text_labels

    def _parse_sample(self, sample) -> tuple[np.ndarray, int, str]:
        audio_data = sample["audio"]
        waveform = np.array(audio_data["array"])
        sample_rate = audio_data["sampling_rate"]
        text = sample["text"]
        return waveform, sample_rate, text

    def _generate_spectrogram(self, waveform: np.ndarray, sample_rate: int) -> torch.Tensor:
        transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=128, n_fft=1024, hop_length=512)
        waveform_tensor = torch.tensor(waveform).unsqueeze(0).float()
        return transform(waveform_tensor)

    def _split_spectrogram(self, spectrogram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        half_length = spectrogram.shape[-1] // 2
        first_half = spectrogram[:, :, :half_length]
        second_half = spectrogram[:, :, half_length:]
        return first_half, second_half

    def _encode_text(self, text: str) -> torch.Tensor:
        sentence = f"[SOS] {text} {self.tokenizer.eos_token}"
        return torch.tensor(self.tokenizer.encode(sentence))