from transformers import get_scheduler
from load_datasets import load_dataloader
from torch.optim import AdamW
from train_eval import train_model, eval_model
from utils import load_model
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='blog', help='Name of dataset')
parser.add_argument('--llm', type=str, default='qwen', help='Name of llm')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training model')
parser.add_argument('--pretrain_threshold', type=float, default=1.0, help='pretrain threshold')
parser.add_argument('--tpg_epochs', type=int, default=10, help='Number of tpg epochs')
parser.add_argument('--tpr_epochs', type=int, default=10, help='Number of tpq epochs')
parser.add_argument('--smp_epochs', type=int, default=10, help='Number of smp epochs')
parser.add_argument('--tpgan_cache', type=str, default="20241015", help='Path of tpgan cache')
parser.add_argument('--tprep_cache', type=str, default="20241015", help='Path of tprep cache')
parser.add_argument('--model_cache', type=str, default="20241015", help='Path of model cache')
parser.add_argument('--tpsmp_cache', type=str, default="20241015", help='Path of tpsmp cache')
parser.add_argument('--use_tpgan', type=str, default='t', help='Use tpgan', choices=['t', 'f'])
parser.add_argument('--use_pg_cache', type=str, default='f', help='Use tpg cache', choices=['t', 'f'])
parser.add_argument('--use_pr_cache', type=str, default='f', help='Use tpr cache', choices=['t', 'f'])
parser.add_argument('--train_or_test', type=str, default='train', help='Train or test directly', choices=['train', 'test'])
parser.add_argument('--hid_size', type=int, default=32, help='Hidden layer size')
parser.add_argument('--batch', type=int, default=6, help='Batch size')
parser.add_argument('--gru_layers', type=int, default=2, help='gru layers')
parser.add_argument('--num_samples', type=int, default=100, help='Number of samples for positive set of tpg')
parser.add_argument('--num_samples_replace_ratio', type=float, default=0.01, help='Number of indices replacement for sampling positive set of tpg')
parser.add_argument('--sample_loss_ratio', type=float, default=0.5, help='ratio of sampling loss of tpr')
parser.add_argument('--topk_sim_words', type=int, default=30, help='Topk similar words of the vocabulary')
parser.add_argument('--topk_gen_samples', type=int, default=5, help='Topk similar words for generating samples')
parser.add_argument('--topk_prob_tokens_rep', type=int, default=5, help='Topk probability tokens for replacement')
parser.add_argument('--bottomk_sim_words', type=int, default=100, help='Bottomk similar words of the vocabulary')
parser.add_argument('--infer_text_example_num', type=int, default=1, help='The number of infer text example')
parser.add_argument('--device', type=str, default="cuda:0", help='Device')
parser.add_argument('--froze_client', type=str, default="y", help='Froze parameters of client model', choices=["y", "n"])
parser.add_argument('--lr', type=float, default=0.00015, help='learning rate')
parser.add_argument('--tpg_lr', type=float, default=0.0005, help='tpg_learning rate')
parser.add_argument('--tpd_lr', type=float, default=0.0002, help='tpd_learning rate')
parser.add_argument('--tpr_lr', type=float, default=0.002, help='tpr_learning rate')
parser.add_argument('--sampler_lr', type=float, default=0.00018, help='smp_learning rate')
parser.add_argument('--random_seed', type=int, default=42, help='Random seed for dataloader')
parser.add_argument('--attack_type', type=str, default="eia", help='attack type', choices=['eia', 'aia'])
args = parser.parse_args()

train_loader, val_loader, test_loader, mini_loader, llm_model, num_classes, max_position_embeddings, padding_id, _ = load_dataloader(args.dataset, args.llm, args.batch, args.attack_type, "same_random", args.random_seed)
llm_model = load_model(llm_model, num_classes)

if args.froze_client == "y":
    optimizer = AdamW(filter(lambda p: p.requires_grad, llm_model.parameters()), lr=args.lr)
else:
    optimizer = AdamW(llm_model.parameters(), lr=args.lr)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=args.epochs * len(train_loader),
)

use_tpgan = True if args.use_tpgan == 't' else False
use_tpg_cache = True if args.use_pg_cache == 't' else False
use_tpr_cache = True if args.use_pr_cache == 't' else False

if args.train_or_test == "train":
    llm, tpg = train_model(llm_model, train_loader, mini_loader, optimizer, lr_scheduler, use_tpgan, use_tpg_cache, use_tpr_cache, max_position_embeddings, padding_id, args)
    eval_model(test_loader, use_tpgan, args, padding_id, llm, tpg)
else:
    eval_model(test_loader, use_tpgan, args, padding_id)

# print args
print("args:", args)
