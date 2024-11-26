import torch
import torch.nn as nn
import numpy as np
import librosa
import matplotlib.pyplot as plt

class Spectron(nn.Module):
    def __init__(self, mel_dims):
        speech_emb_dim = 495 #Placeholder
        lm_emb_dim = 495 #Placeholder
        self.speech_encoder = None
        self.prenet = nn.Sequential(
            nn.Linear(mel_dims, 360),
            nn.Linear(360,lm_emb_dim)
        )
        self.postnet = nn.Sequential(
            nn.Linear(lm_emb_dim,360),
            nn.Linear()
        )
        self.connector = nn.Linear(speech_emb_dim,lm_emb_dim)
        self.lm = None
        self.text_head = None

    def process_prompt(self, prompt):
        enc_speech = self.speech_encoder(prompt)
        return self.connector(enc_speech)


    def forward(self, prompt, transcript, continuation):
        prompt = self.process_prompt(prompt)
        continuation = self.prenet(continuation)
        #Assuming prompt, transcript, continuation : [b,l,c]
        lm_input = torch.concatenate([prompt,transcript,continuation],dim=1)
        lm_output = self.lm(lm_input)
        bos_token = prompt.size()[1]
        eos_token = bos_token+transcript.size()[1]-1
        transcript = lm_output[:,bos_token:eos_token+1,:]
        continuation = lm_output[:,eos_token+1:,:]
        transcript = self.text_head(transcript)
        continuation = self.postnet(continuation)
        return transcript, continuation

    def compute_loss(self, transcript, continuation, target_transcript, target_continuation):
        pass

    def generate(self, prompt, bos):
        prompt = self.process_prompt(prompt)
        prompt = torch.concatenate([prompt,bos],dim=1)
        while True:
            lm_input = self.prenet