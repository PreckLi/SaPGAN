import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from utils import similar_words
import math


class PrivacyInjector(nn.Module):
    def __init__(self, hidden_dim, noise_dim):
        super(PrivacyInjector, self).__init__()
        self.noise_dim = noise_dim
        self.fc = nn.Linear(hidden_dim + noise_dim, hidden_dim)

    def forward(self, x):
        noise = torch.randn((x.shape[0], self.noise_dim)).to(x.device)
        x = torch.cat((x, noise), dim=-1)
        x = self.fc(x)
        return x

class Sampler(nn.Module):
    def __init__(self, hidden_dim, emb_layer, padding_ids, max_length, gru_layers, topk, device):
        super(Sampler, self).__init__()
        self.padding_idx = padding_ids
        self.emb_layer = emb_layer
        for param in self.emb_layer.parameters():
            param.requires_grad = True
        self.device = device
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        self.decision_gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=gru_layers, batch_first=True)
        self.decision_conv = nn.Conv1d(in_channels=hidden_dim, out_channels=self.max_length, kernel_size=1)

        self.selector_gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=gru_layers, batch_first=True)
        self.selector_linear = nn.Linear(hidden_dim, topk)
        self.init_params()
        
    def forward(self, inp, hidden_out, similar_words, max_replacements_ratio):
        decision_gru_out, _ = self.decision_gru(hidden_out)
        decision_gru_out = decision_gru_out.view(inp.shape[0], inp.shape[1], -1).permute(0, 2, 1)
        decision_gru_out = self.decision_conv(decision_gru_out)
        decision_gru_out = torch.max_pool1d(decision_gru_out, kernel_size=decision_gru_out.shape[-1])
        decision_gru_out = decision_gru_out.squeeze(-1)
        decision_probs = torch.sigmoid(decision_gru_out)
        
        selector_gru_out, _ = self.selector_gru(hidden_out.view(inp.shape[0], inp.shape[1], -1))
        selector_probs = self.selector_linear(selector_gru_out)
        selector_probs = torch.softmax(selector_probs, dim=-1)
        mask = torch.zeros_like(decision_probs).to(inp.device)
        inp_padding = F.pad(inp, (0, self.max_length - inp.shape[1]), value=self.padding_idx)
        mask[inp_padding != self.padding_idx] = 1
        decision_probs = decision_probs * mask
        max_replacements = int(max_replacements_ratio * inp.shape[1])
        replaced_inp = self.sample_replace_tokens(inp, selector_probs, decision_probs, similar_words, max_replacements)
        replaced_emb = self.emb_layer(replaced_inp)
        # 返回预测概率和替换词
        return replaced_emb
    
    def sample_replace_tokens(self, inp, selector_probs, decision_probs, similar_words, max_replacements):
        _, topk_indices = torch.topk(decision_probs, max_replacements, dim=-1)
        inpclone = inp.clone()
        batch_indices = torch.arange(inp.shape[0], device=topk_indices.device).view(-1, 1).expand_as(topk_indices)
        original_word_indices = inp[batch_indices, topk_indices]
        selected_synonym_indices = torch.argmax(selector_probs[batch_indices, topk_indices], dim=-1)
        new_word_indices = similar_words[original_word_indices, selected_synonym_indices]
        inpclone[batch_indices, topk_indices] = new_word_indices
        return inpclone
    
    def gen_samples(self, inp, hidden_out, hidden, similar_words, args):
        decision_gru_out, _ = self.decision_gru(hidden_out)
        decision_gru_out = decision_gru_out.view(inp.shape[0], inp.shape[1], -1).permute(0, 2, 1)
        decision_gru_out = self.decision_conv(decision_gru_out)
        decision_gru_out = torch.max_pool1d(decision_gru_out, kernel_size=decision_gru_out.shape[-1])
        decision_gru_out = decision_gru_out.squeeze(-1)
        decision_probs = torch.sigmoid(decision_gru_out)
        
        selector_gru_out, _ = self.selector_gru(hidden_out.view(inp.shape[0], inp.shape[1], -1))
        selector_probs = self.selector_linear(selector_gru_out)
        selector_probs = torch.softmax(selector_probs, dim=-1)
        
        mask = torch.zeros_like(decision_probs).to(inp.device)
        inp_padding = F.pad(inp, (0, self.max_length - inp.shape[1]), value=self.padding_idx)
        mask[inp_padding != self.padding_idx] = 1
        decision_probs = decision_probs * mask
        num_samples_replace = int(args.num_samples_replace_ratio * inp.shape[1])
        _, topk_indices = torch.topk(decision_probs, num_samples_replace, dim=-1)
        batch_size, sequence_length = inp.shape
        samples = torch.zeros((args.num_samples, batch_size, sequence_length), dtype=inp.dtype, device=inp.device)
        for sample_idx in range(args.num_samples):
            inpclone = inp.clone()
            top_replace_words_vals, top_replace_words_idxs = torch.topk(selector_probs[torch.arange(batch_size).unsqueeze(-1), topk_indices], args.topk_prob_tokens_rep, dim=-1)
            top_replace_words_vals = F.softmax(top_replace_words_vals, dim=-1)
            selected_idx = torch.multinomial(top_replace_words_vals.view(-1, args.topk_prob_tokens_rep), 1).squeeze()
            selected_words = top_replace_words_idxs.view(-1, args.topk_prob_tokens_rep)[torch.arange(batch_size * num_samples_replace), selected_idx]
            inpclone[torch.arange(batch_size).unsqueeze(-1), topk_indices] = similar_words[inp[torch.arange(batch_size).unsqueeze(-1), topk_indices], selected_words.view(batch_size, num_samples_replace)]
            samples[sample_idx] = inpclone
        return samples
        
    def init_params(self):
        for param in self.parameters():
            if param.requires_grad:
                if isinstance(param, nn.Linear):
                    nn.init.kaiming_uniform_(param.weight, nonlinearity='leaky_relu')
    
    def init_hidden(self, batch_size):
        h = torch.zeros(1, batch_size, self.hidden_dim)
        return h.to(self.device)

class GRUGenerator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, padding_idx, noise_dim, device):
        super(GRUGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.device = device
        self.temperature = 1.0
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(2 * embedding_dim, hidden_dim, batch_first=True)
        self.gru2out = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        
        # self.privacy_injector = PrivacyInjector(hidden_dim, noise_dim)
        self.init_params()

    def forward(self, inp, emb, hidden, need_hidden=False):
        emb_gen = self.embeddings(inp)  # batch_size * seq_len * embedding_dim
        emb_cat = torch.cat((emb_gen, emb), dim=-1)
        out, hidden = self.gru(emb_cat, hidden)
        hidden_out = out.contiguous().view(-1, self.hidden_dim)  # out: (batch_size * len) * hidden_dim
        # out = self.privacy_injector(out)
        out = self.gru2out(hidden_out)  # (batch_size * seq_len) * vocab_size
        # out = self.temperature * out  # temperature
        pred = self.softmax(out)
        if need_hidden:
            return pred, hidden_out, hidden
        else:
            return pred, hidden_out


    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                torch.nn.init.normal_(param, std=stddev)

    def init_hidden(self, batch_size):
        h = torch.zeros(1, batch_size, self.hidden_dim)
        return h.to(self.device)
    
    def gen_samples(self, embeddings, real_ids, padding_ids, args):
        num_samples, num_samples_replace, topk, bottomk = args.num_samples, args.num_samples_replace, args.topk_sim_words, args.bottomk_sim_words
        try:
            similar_tokens = torch.load(f"/large_disk/lkh/python_projects/llm/fl_project/spliteroberta_tpgan/model_caches/{args.llm}_similar_words_{topk}.pth").to(real_ids.device)
        except:
            similar_tokens = similar_words(embeddings, args.llm, topk=topk)
        try:
            unsimilar_tokens = torch.load(f"/large_disk/lkh/python_projects/llm/fl_project/spliteroberta_tpgan/model_caches/{args.llm}_unsimilar_words_{bottomk}.pth").to(real_ids.device)
        except:
            unsimilar_tokens = similar_words(embeddings, args.llm, bottomk=bottomk)

        real_ids_expanded = real_ids.unsqueeze(0).expand(num_samples, -1, -1)
        random_indices = torch.randint(0, num_samples_replace, real_ids_expanded.size())

        pos_samples = similar_tokens[real_ids_expanded, random_indices]
        pos_samples[:, real_ids == padding_ids] = padding_ids

        neg_samples = unsimilar_tokens[real_ids_expanded, random_indices]
        neg_samples[:, real_ids == padding_ids] = padding_ids
        return pos_samples, neg_samples


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Discriminator(nn.Module):
    def __init__(self, vocab_size, d_model=16, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1):
        super(Discriminator, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.classifier = nn.Linear(d_model, 2)
        self.log_soft = nn.LogSoftmax(dim=-1)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)
        output = torch.sum(output, dim=1)
        output = self.classifier(output)
        output = self.log_soft(output)
        return output
