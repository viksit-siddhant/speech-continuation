import os
import torch
import aiohttp
import datasets
import numpy as np
import transformers
from typing import Iterator


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, test: bool = False):
        super().__init__()
        os.makedirs("./dataset", exist_ok=True)
        if test:
            self.dataset = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        else:
            self.dataset = datasets.load_dataset(
                "librispeech_asr",
                split="train.clean.100", trust_remote_code=True,
                storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=360000)}}
            )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        sample = self.dataset[idx]
        waveform, sample_rate, text = self._parse_sample(sample)
        spectrogram = self._generate_spectrogram(waveform, sample_rate)
        first_half, second_half = self._split_spectrogram(spectrogram)
        text_labels = self._encode_text(text)
        return first_half, second_half, text_labels

    def _parse_sample(self, sample: dict) -> tuple[np.ndarray, int, str]:
        audio_data = sample["audio"]
        waveform = np.array(audio_data["array"]).squeeze().astype(np.float32)
        sample_rate = audio_data["sampling_rate"]
        text = sample["text"]
        return waveform, sample_rate, text

    def _generate_spectrogram(self, waveform: np.ndarray, sample_rate: int) -> torch.Tensor:
        transform = transformers.WhisperFeatureExtractor(sampling_rate=sample_rate, n_fft=1024, hop_length=512)
        result = transform(waveform, sampling_rate=sample_rate, return_tensors="pt")
        return result.get("input_features")

    def _split_spectrogram(self, spectrogram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        half_length = spectrogram.shape[-1] // 2
        first_half = spectrogram[:, :, :half_length]
        second_half = spectrogram[:, :, half_length:]
        return first_half, second_half

    def _encode_text(self, text: str) -> torch.Tensor:
        sentence = f"{self.tokenizer.bos_token} {text} {self.tokenizer.eos_token}"
        return self.tokenizer.encode(sentence, add_special_tokens=False, return_tensors="pt").squeeze()


class LibriSpeechDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: LibriSpeechDataset, batch_size: int = 8, shuffle: bool = True):
        super().__init__(dataset, batch_size=batch_size)
        self.shuffle = shuffle
        self.indices = torch.arange(len(self.dataset))

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor]]:
        if self.shuffle:
            self.indices = torch.randperm(len(self.dataset))

        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i : i + self.batch_size]
            batch = [self.dataset[idx.item()] for idx in batch_indices]
            yield self._collate_batch(batch)

    def _collate_batch(self, batch: list[tuple[torch.Tensor]]) -> tuple[torch.Tensor]:
        first_halves, second_halves, text_labels = zip(*batch)
        first_halves = torch.cat(first_halves, dim=0)
        second_halves = torch.cat(second_halves, dim=0)
        text_labels = torch.nn.utils.rnn.pad_sequence(text_labels, batch_first=True, padding_value=0)
        return first_halves, second_halves, text_labels