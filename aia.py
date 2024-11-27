import torch
from torch import nn
from torch.nn import functional as F
from load_datasets import load_dataloader
from utils import load_model
from cti import get_ui_values
from sapgan import GRUGenerator
from dp_algorithm import get_dp_output
from write_log import write_logger
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='blog', help='Name of dataset')
parser.add_argument('--embed_dim', type=int, default=100, help='Embedding dim')
parser.add_argument('--num_filters', type=int, default=100, help='Number of filters')
parser.add_argument('--filter_sizes', type=list, default=[3, 4, 5], help='Filter sizes')
parser.add_argument('--output_dim', type=int, default=2, help='Output dim')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout')
parser.add_argument('--llm', type=str, default='qwen', help='Name of llm', choices=['roberta'])
parser.add_argument('--batch', type=int, default=32, help='Batch size')
parser.add_argument('--hid_size', type=int, default=32, help='Hidden layer size')
parser.add_argument('--device', type=str, default="cuda:0", help='Device')
parser.add_argument('--attack_type', type=str, default="aia", help='attack type')
parser.add_argument('--use_disturb', type=str, default='t', help='Use disturb', choices=['t', 'f'])
parser.add_argument('--tpgan_cache', type=str, default="20240916", help='Path of tpgan cache')
parser.add_argument('--dp_eta', type=int, default=450, help='eta of differential privacy')
parser.add_argument('--disturb_type', type=str, default='nn', help='Disturb type', choices=['dp', 'nn'])
parser.add_argument('--random_seed', type=int, default=42, help='Random seed for dataloader')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for training model')
parser.add_argument('--use_cti', type=str, default="t", help='use cti or not', choices=["t", "f"])
args = parser.parse_args()


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)
    
def train(model, optimizer, client_model, criterion, train_loader, top_k_tokens_per_class, device):
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, labels, metadata = batch[0]['input_ids'].to(device), batch[1].to(device), batch[2]
            age = list()
            gender = list()
            for m in metadata:
                if 0 in m:
                    age.append(1)
                else:
                    age.append(0)
                if 1 in m:
                    gender.append(1)
                else:
                    gender.append(0)
            age = torch.tensor(age).long().to(device)
            gender = torch.tensor(gender).long().to(device)
            if args.use_disturb == 't':
                if args.disturb_type == 'nn':
                    client_model.eval()
                    num_feat = client_model.embeddings.word_embeddings.weight.shape[1]
                    vocab_size = client_model.embeddings.word_embeddings.weight.shape[0]
                    padding_ids = client_model.embeddings.padding_idx
                    tpg = GRUGenerator(num_feat, args.hid_size, vocab_size, padding_ids, args.hid_size, args.device).to(args.device)
                    tpg.load_state_dict(torch.load(f"/home/likunhao/python_projects/spliteroberta_tpgan/model_caches/{args.dataset}_tpr_{args.llm}_{args.tpgan_cache}.pth", map_location='cuda:0'))
                    tpg.to(args.device)
                    tpg.eval()
                    outputs = client_model(input_ids)
                    hidden_states = tpg.init_hidden(input_ids.shape[0])
                    gen_emb, _ = tpg(input_ids, outputs, hidden_states)
                    input_ids = torch.argmax(gen_emb, dim=-1).view(input_ids.shape[0], -1)
                if args.disturb_type == 'dp':
                    outputs = get_dp_output(outputs, input_ids, client_model.embeddings.word_embeddings.weight, top_k_tokens_per_class, args)
                    word_embeddings = client_model.embeddings.word_embeddings.weight
                    reshp = client_outputs.shape[1]
                    batch_size = client_outputs.shape[0]
                    client_outputs = client_outputs.reshape(-1, client_outputs.shape[-1])

                    word_embeddings_norm = F.normalize(word_embeddings, p=2, dim=1)
                    client_outputs_norm = F.normalize(client_outputs, p=2, dim=1)

                    cosine_similarities = torch.mm(client_outputs_norm, word_embeddings_norm.T)

                    cosine_similarities = cosine_similarities.view(batch_size, reshp, -1)
                    input_ids = torch.argmax(cosine_similarities, -1)
            outputs = model(input_ids)
            loss = criterion(outputs, gender)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} loss: {total_loss / len(train_loader)}")
    return model

def eval(model, val_loader, criterion, top_k_tokens_per_class, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, labels, metadata = batch[0]['input_ids'].to(device), batch[1].to(device), batch[2]
            age = list()
            gender = list()
            for i in metadata:
                if i == {0}:
                    age.append(0)
                    gender.append(0)
                elif i == {1}:
                    age.append(1)
                    gender.append(1)
                elif len(i) == 0:
                    age.append(1)
                    gender.append(0)
                elif i == {0, 1}:
                    age.append(0)
                    gender.append(1)
            age = torch.tensor(age).to(device)
            gender = torch.tensor(gender).to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, gender)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == gender).sum().item()
    message = f"Val loss: {total_loss / len(val_loader)}, accuracy: {correct / len(val_loader.dataset)}"
    print(message)
    write_logger(args, message)
    return

train_loader, val_loader, test_loader, mini_loader, llm_model, num_classes, _, _, ds = load_dataloader(args.dataset, args.llm, args.batch, args.attack_type, "random", args.random_seed)
model = load_model(llm_model, num_classes)
model.to(args.device)

INPUT_DIM = model.embeddings.word_embeddings.weight.shape[0]
EMBEDDING_DIM = args.embed_dim
N_FILTERS = args.num_filters
FILTER_SIZES = args.filter_sizes
OUTPUT_DIM = args.output_dim
DROPOUT = args.dropout

model = TextCNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()

if args.use_cti == "t":
    ui_values = get_ui_values(ds, num_classes)
    k = 100
    top_k_tokens_per_class = {}
    for label in range(num_classes):
        sorted_tokens = sorted(ui_values.items(), key=lambda item: item[1][label], reverse=True)[:k]
        top_k_tokens_per_class[label] = [token for token, _ in sorted_tokens]
else:
    top_k_tokens_per_class = None

train(model, optimizer, model, criterion, train_loader, top_k_tokens_per_class, args.device)
eval(model, test_loader, criterion, top_k_tokens_per_class, args.device)
