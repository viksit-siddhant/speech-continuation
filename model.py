import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoConfig, AutoTokenizer, WhisperModel
import transformers
from einops import rearrange
import torch
import torch.nn as nn

class SpeechContinuationLoss(nn.Module):
    def __init__(self, max_k):
        """
        Initialize the SpeechContinuationLoss.

        Args:
        max_k (int): The maximum order for the discrete derivative loss (K).
        """
        super(SpeechContinuationLoss, self).__init__()
        self.max_k = max_k

    def compute_deltas(self, z, k, dim):
        """
        Compute the discrete deltas (derivatives) along a given dimension.

        Args:
        z (torch.Tensor): Input tensor of shape (batch, freq, time).
        k (int): Order of the delta.
        dim (int): Dimension along which to compute the delta.

        Returns:
        torch.Tensor: The computed delta.
        """
        if dim == 2:  # Time-deltas
            return z[:, :, k:] - z[:, :, :-k]
        elif dim == 1:  # Feature-deltas
            return z[:, k:, :] - z[:, :-k, :]

    def l1_l2_loss(self, z, z_pred):
        """
        Compute the combined L1 and L2 loss.

        Args:
        z (torch.Tensor): Ground truth tensor of shape (batch, freq, time).
        z_pred (torch.Tensor): Predicted tensor of the same shape.

        Returns:
        torch.Tensor: The combined L1 and L2 loss.
        """
        l1_loss = torch.sum(torch.abs(z - z_pred))
        l2_loss = torch.sum((z - z_pred) ** 2)
        return l1_loss + l2_loss

    def forward(self, x_c, x_c_hat):
        """
        Compute the speech continuation loss.

        Args:
        x_c (torch.Tensor): Ground truth spectrogram of shape (batch, freq, time).
        x_c_hat (torch.Tensor): Predicted spectrogram of the same shape.

        Returns:
        torch.Tensor: The total reconstruction loss.
        """
        # Ls: L1+2 loss on the spectrogram
        ls_loss = self.l1_l2_loss(x_c, x_c_hat)

        # Lf: L1+2 loss on feature deltas up to order 1
        lf_loss = 0
        for k in range(1, 2):  # Assuming order K = 1
            delta_feat_gt = self.compute_deltas(x_c, k, dim=1)
            delta_feat_pred = self.compute_deltas(x_c_hat, k, dim=1)
            lf_loss += self.l1_l2_loss(delta_feat_gt, delta_feat_pred)

        # Lt: L1+2 loss on time deltas up to order K
        lt_loss = 0
        for k in range(1, self.max_k + 1):
            delta_time_gt = self.compute_deltas(x_c, k, dim=2)
            delta_time_pred = self.compute_deltas(x_c_hat, k, dim=2)
            lt_loss += self.l1_l2_loss(delta_time_gt, delta_time_pred)

        # Total loss
        total_loss = ls_loss + lf_loss + lt_loss
        return total_loss


class Spectron(nn.Module):
    def __init__(self, mel_dims):
        self.lm = AutoModel.from_pretrained("HuggingFaceTB/SmolLM-360M")
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M")
        self.lm.config.is_decoder = True
        speech_emb_dim = 512
        lm_emb_dim = self.lm.config.hidden_size
        self.speech_encoder = WhisperModel.from_pretrained('openai/whisper-base')
        self.prenet = nn.Sequential(
            nn.Linear(mel_dims, 360),
            nn.Linear(360,lm_emb_dim)
        )
        self.postnet = nn.Sequential(
            nn.Linear(lm_emb_dim,360),
            nn.Linear(360,mel_dims)
        )
        self.connector = nn.Linear(speech_emb_dim,lm_emb_dim)
        self.text_head = nn.Linear(lm_emb_dim,self.lm.config.vocab_size)
        self.recon_loss = SpeechContinuationLoss(2)

    def process_prompt(self, prompt):
        enc_speech = self.speech_encoder(prompt)
        return self.connector(enc_speech)


    def forward(self, prompt, transcript, continuation):
        prompt = self.process_prompt(prompt)
        continuation = self.prenet(continuation)
        #Assuming prompt, transcript, continuation : [b,l,c]
        transcript = self.lm.embed_tokens(transcript)
        eos_embed = self.lm.embed_tokens(torch.tensor(self.tokenizer.eos_token_id).unsqueeze(0))
        lm_input = torch.concatenate([prompt,transcript,continuation,eos_embed],dim=1)
        lm_output = self.lm(input_embeds=lm_input)
        bos_token = prompt.size()[1]
        eos_token = bos_token+transcript.size()[1]-1
        transcript = lm_output[:,bos_token:eos_token+1,:]
        continuation = lm_output[:,eos_token+1:,:]
        transcript = self.text_head(transcript)
        continuation = self.postnet(continuation)
        return transcript, continuation

    def compute_loss(self, transcript, continuation, target_transcript, target_continuation):
        lm_loss = nn.functional.cross_entropy(transcript,target_transcript)
        recon_loss = self.recon_loss(continuation,target_continuation)
        return lm_loss + recon_loss


    def generate(self, prompt):
        prompt = self.process_prompt(prompt)
        bos = self.tokenizer.bos_token_id
        bos = torch.tensor(bos).unsqueeze(0)
        bos = self.lm.embed_tokens(bos)
        prompt = torch.concatenate([prompt,bos],dim=1)
        start = prompt.size()[1]
        transcript = ""
        eos_count = 0
        continuation = []
        limit = 1000
        for i in range(limit):
            lm_output = self.lm(input_embeds=prompt)
            token = lm_output[0,-1, :]
            token = self.text_head(token).argmax()
            token = token.unsqueeze(0)
            if token.item() == self.tokenizer.eos_token_id:
                eos_count += 1
                if eos_count == 2:
                    break
                prompt = torch.concatenate([prompt,self.lm.embed_tokens(token)],dim=1)
                continue
            if eos_count == 0:
                prompt = torch.concatenate([prompt,self.lm.embed_tokens(token)],dim=1)
                transcript += self.tokenizer.decode(token.squeeze())
            else:
                frame = lm_output[0,-1, :].unsqueeze(0)
                continuation.append(self.postnet(frame))
                prompt = torch.concatenate([prompt,frame],dim=1)
        return transcript, continuation

