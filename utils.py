import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import re
from tqdm import tqdm

path = "/home/likunhao/python_projects/spliteroberta_tpgan/model_caches/"

def load_model(llm_model, num_classes=3):
    for param in llm_model.get_input_embeddings().parameters():
        param.requires_grad = False
    if llm_model.config.architectures[0] in ["BertForMaskedLM", "RobertaForMaskedLM"]:
        pattern = r'\((\w+)\): Linear'
        linear_layers = re.findall(pattern, str(llm_model.modules))
        target_modules = list(set(linear_layers))
    if llm_model.config.architectures[0] in ["Qwen2ForCausalLM"]:
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, target_modules=target_modules, inference_mode=False, r=8,
                             lora_alpha=32, lora_dropout=0.1)
    llm_model = get_peft_model(llm_model, peft_config)
    llm_model.print_trainable_parameters()
    return llm_model


def infer_input_text(client_indices, input_ids, num=2):
    tokenizer = RobertaTokenizer.from_pretrained("/home/likunhao/python_projects/roberta-base", trust_remote_code=True)
    decoded_texts = []
    for i in range(client_indices.shape[0]):
        decoded_text = tokenizer.decode(client_indices[i], skip_special_tokens=True)
        decoded_texts.append(decoded_text)
    decoded_texts_true = []
    for i in range(input_ids.shape[0]):
        decoded_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
        decoded_texts_true.append(decoded_text)
    for idx, (i, j) in enumerate(zip(decoded_texts, decoded_texts_true)):
        if idx == num:
            break
        print("simulate text:", i)
        print("real     text:", j)


def prepare_dis_data(pos_samples, neg_samples, device):
    """Build inp and target"""
    inp = torch.cat((pos_samples, neg_samples), dim=0).long().detach()
    target = torch.ones(inp.size(0)).long()
    target[pos_samples.size(0):] = 0
    
    inp_fake = neg_samples
    target_fake = torch.ones(inp_fake.size(0)).long()
    # shuffle
    perm = torch.randperm(inp.size(0))
    inp = inp[perm]
    target = target[perm]
    return inp.to(device), target.to(device), inp_fake.to(device), target_fake.to(device)

def eia_accuracy(real_tokens, fake_tokens, padding_id=1):
    mask = (real_tokens != padding_id) & (fake_tokens != padding_id)
    different_tokens = torch.sum((real_tokens != fake_tokens) & mask).item()
    total_tokens = torch.sum(mask).item()
    return different_tokens, total_tokens

def similar_words(embeddings, llm, topk=None, bottomk=None):
    if embeddings.shape[0] < 60000:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        cosine_similarities = torch.mm(embeddings, embeddings.T)
        if topk is not None:
            _, sim_indices = torch.topk(cosine_similarities, topk+1, dim=1)
            sim_indices = sim_indices[:, 1:]
            torch.save(sim_indices, path + f"{llm}_similar_words_{topk}.pth")
            return sim_indices
        if bottomk is not None:
            _, unsim_indices = torch.topk(cosine_similarities, bottomk, dim=1, largest=False)
            unsim_indices = unsim_indices[:, 1:]
            torch.save(unsim_indices, path + f"{llm}_unsimilar_words_{bottomk}.pth")
            return unsim_indices
    else:
        chunk_size = 5000
        num_embeddings = embeddings.size(0)
        topk_values_list = []
        topk_indices_list = []
        for i in tqdm(range(0, num_embeddings, chunk_size)):
            end_i = min(i + chunk_size, num_embeddings)
            chunk = embeddings[i:end_i]
            similarity = torch.matmul(chunk, embeddings.T)
            topk_values, topk_indices = torch.topk(similarity, k=topk, dim=1)
            topk_values_list.append(topk_values)
            topk_indices_list.append(topk_indices)
        topk_indices = torch.cat(topk_indices_list, dim=0)
        torch.save(topk_indices, path + f"{llm}_similar_words_{topk}.pth")
        if topk is not None:
            return topk_indices
