import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from utils.rev_in import RevIn

class GPT4MTS(nn.Module):
    
    def __init__(self, configs, device):
        super(GPT4MTS, self).__init__()
        self.is_gpt = configs.is_gpt
        self.revin = configs.revin
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1

        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True, cache_dir='/drive2/florajia/huggingface_cache/')  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())

            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))

        self.relu = nn.ReLU()
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.prompt_layer = nn.Linear(configs.d_model, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * (self.patch_num), configs.pred_len)
        
        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.in_layer, self.out_layer, self.prompt_layer):
            layer.to(device=device)
            layer.train()
        
        self.device = device
        self.rev_in = RevIn(num_features=1) # channel independent

    def get_emb(self, x, tokens=None):
        if tokens is None:
            x = self.gpt2(inputs_embeds=x).last_hidden_state
            return x
        else:
            [a, b, c] = x.shape
            prompt_x = self.relu(self.prompt_layer(tokens))
            x_all = torch.cat((prompt_x, x), dim=1)
            x = self.gpt2(inputs_embeds=x_all).last_hidden_state
            return x[: , -b:, :]

    def get_patch(self, x):
        x = rearrange(x, 'b l m -> b m l')
        x = self.padding_patch_layer(x) # b, 1, seq_len
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) #b, 1, patch_num, patch_size
        x = rearrange(x, 'b m n p -> (b m) n p') # b, patch_num, patch_size

        return x

    def forward(self, x, itr, summary):
        summary = rearrange(summary, 'b l m -> b m l') # [b, 768, 15]
        summary = self.padding_patch_layer(summary) # [b, 768, 19] 
        summary = summary.unfold(dimension=-1, size=self.patch_size, step=self.stride) # [b, 768, 3, 8]
        summary = summary.mean(dim=-1).squeeze() # [b, 768, 3]
        summary = rearrange(summary, 'b l m -> b m l') # [b, 3, 768]

        B, L, M = x.shape # 4, 512, 1

        if self.revin:   
            x = self.rev_in(x, 'norm').to(self.device)
        else:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
            x /= stdev
       
        x = self.get_patch(x)
        x = self.in_layer(x)

        outputs = self.get_emb(x, summary)
        outputs = self.out_layer(outputs.reshape(B*M, -1)) 
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
        
        if self.revin:
            outputs = self.rev_in(outputs, 'denorm').to(self.device)
        else:
            outputs = outputs * stdev
            outputs = outputs + means

        return outputs
