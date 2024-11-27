import torch
from torch import nn
from torch.nn import functional as F
from utils import prepare_dis_data, eia_accuracy, similar_words
from sapgan import  GRUGenerator, Discriminator, Sampler
from transformers import AutoModelForCausalLM, RobertaModel, BertModel
from peft import PeftModel
from write_log import write_logger
import copy
import datetime
import time

model_caches_path = "/home/likunhao/python_projects/spliteroberta_tpgan/model_caches/"
model_path = "/home/likunhao/python_projects/"
time_today = datetime.datetime.now()
time_today = time_today.strftime('%Y%m%d')


def pretrain_tpgan(embeddings, num_feat, vocab_size, llm, data_loader, padding_ids, max_position_embeddings, args):
    print("-------------Start pretrain Generator-------------")
    since = time.time()
    similar_tokens_path = model_caches_path + f"{args.llm}_similar_words_{args.topk_sim_words}.pth"
    try:
        similar_tokens = torch.load(similar_tokens_path, weights_only=True).to(args.device)
    except:
        print(f"Similar words file with {args.topk_sim_words} words not found, generating now...")
        similar_tokens = similar_words(embeddings, args.llm, args.topk_sim_words)
    
    # Pretrain Generator 
    tpg = GRUGenerator(num_feat, args.hid_size, vocab_size, padding_ids, args.hid_size, args.device).to(args.device)
    g_optimizer = torch.optim.Adam(tpg.parameters(), lr=args.tpg_lr)
    tpg.train()
    embedding_layer = llm.get_input_embeddings()
    for epoch in range(args.tpg_epochs):
        total_g_loss = 0
        for (X, y) in data_loader:
            X, y = X.to(args.device), y.to(args.device)
            input_ids = X['input_ids']
            real_emb = embedding_layer(input_ids)
            g_optimizer.zero_grad()
            hidden = tpg.init_hidden(y.shape[0])
            gen_emb, hidden_out, hidden = tpg(input_ids, real_emb, hidden, need_hidden=True)
            g_loss = F.nll_loss(gen_emb, input_ids.view(-1), ignore_index=padding_ids)
            g_loss.backward(retain_graph=True)
            g_optimizer.step()
            total_g_loss += g_loss.item()
        epoch_g_loss = total_g_loss / len(data_loader)
        if epoch_g_loss < args.pretrain_threshold:
            break
        print(f"Pretrain Generator Epoch {epoch}, g_loss: {epoch_g_loss}")

    # Pretrain Sampler
    since1 = time.time()
    cost1 = since1 - since
    print(f"Pretrain Generator finished, time cost: {cost1}")
    llm.eval()
    embedding_layer_copy = copy.deepcopy(llm.get_input_embeddings())
    smp = Sampler(args.hid_size, embedding_layer_copy, padding_ids, max_position_embeddings, args.gru_layers, args.topk_sim_words, args.device).to(args.device)
    s_optimizer = torch.optim.Adam(smp.parameters(), lr=args.sampler_lr)
    smp.train()
    for epoch in range(args.smp_epochs):
        total_s_loss = 0
        for (X, y) in data_loader:
            X, y = X.to(args.device), y.to(args.device)
            input_ids = X['input_ids']
            real_emb = embedding_layer(input_ids)
            hidden = smp.init_hidden(y.shape[0])
            gen_emb, hidden_out, hidden = tpg(input_ids, real_emb, hidden, need_hidden=True)
            replaced_emb = smp(input_ids, hidden_out, similar_tokens, args.num_samples_replace_ratio)
            s_loss = F.mse_loss(replaced_emb, real_emb)
            s_loss.backward()
            s_optimizer.step()
            total_s_loss += s_loss.item()
        epoch_s_loss = total_s_loss / len(data_loader)
        print(f"Pretrain Sampler Epoch {epoch}, s_loss: {epoch_s_loss}")    
    
    
    torch.save(tpg.state_dict(), model_caches_path + f"{args.dataset}_tpg_{args.llm}_{time_today}.pth")
    torch.save(smp.state_dict(), model_caches_path + f"{args.dataset}_smp_{args.llm}_{time_today}.pth")
    cost = time.time() - since1
    print(f"TPG training finished, model saved as {args.dataset}_tpg_{args.llm}_{time_today}.pth, SMP training finished, model saved as {args.dataset}_smp_{args.llm}_{time_today}.pth, time cost: {cost}")
    return tpg, smp

def train_tpr(tpg, smp, embeddings, num_feat, vocab_size, llm, data_loader, padding_ids, args):
    print("-------------Start finetune Generator-------------")
    since = time.time()
    similar_words_path = model_caches_path + f"{args.llm}_similar_words_{args.topk_sim_words}.pth"
    try:
        similar_tokens = torch.load(similar_words_path, weights_only=True).to(args.device)
    except:
        print(f"Similar words file with {args.topk_sim_words} words not found, generating now...")
        similar_tokens = similar_words(embeddings, args.llm, args.topk_sim_words)
    tpd = Discriminator(vocab_size).to(args.device)
    r_optimizer = torch.optim.Adam(tpg.parameters(), lr=args.tpr_lr)
    d_optimizer = torch.optim.Adam(tpd.parameters(), lr=args.tpd_lr)
    embedding_layer = llm.get_input_embeddings()
    for epoch in range(args.tpr_epochs):
        total_g_loss = 0
        total_d_loss = 0
        for _, (X, y) in enumerate(data_loader):
            X, y = X.to(args.device), y.to(args.device)
            input_ids = X['input_ids']
            real_emb = embedding_layer(input_ids)
            hidden = tpg.init_hidden(y.shape[0])
            gen_emb, hidden_out = tpg(input_ids, real_emb, hidden)
            g_loss1 = F.nll_loss(gen_emb, input_ids.view(-1), ignore_index=padding_ids)
            r_optimizer.zero_grad()
            d_optimizer.zero_grad()
            
            gen_samples = torch.argmax(gen_emb, dim=-1).view(input_ids.shape[0], -1)
            gen_samples[input_ids == padding_ids] = padding_ids
            inp, target, fake_inp, fake_target = prepare_dis_data(input_ids, gen_samples, args.device)
            pred = tpd(inp)
            pred_fake = tpd(fake_inp)
            d_loss = F.nll_loss(pred, target)
            g_loss2 = F.nll_loss(pred_fake, fake_target)
            
            pos_samples = smp.gen_samples(input_ids, hidden_out, hidden, similar_tokens, args)
            g_loss3 = 0
            for i in range(args.num_samples):
                g_loss3 += F.nll_loss(gen_emb, pos_samples[i].view(-1), ignore_index=padding_ids)
            g_loss3 /= args.num_samples
            g_loss = (g_loss1 + g_loss2 + args.sample_loss_ratio * g_loss3) / 3
            g_loss.backward()
            r_optimizer.step()
            d_loss.backward()
            d_optimizer.step()
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
        epoch_g_loss = total_g_loss / len(data_loader)
        epoch_d_loss = total_d_loss / len(data_loader)
        print(f"Train TPR Epoch {epoch}, g_loss: {epoch_g_loss}, d_loss: {epoch_d_loss}")
    cost = time.time() - since
    torch.save(tpg.state_dict(), model_caches_path + f"{args.dataset}_tpr_{args.llm}_{time_today}.pth")
    print(f"TPR training finished, model saved as {args.dataset}_tpr_{args.llm}_{time_today}.pth, time cost: {cost}")
    return tpg


def train_model(llm, train_loader, mini_loader, optimizer, lr_scheduler, tpgan_flag, tpg_flag, tpr_flag, max_position_embeddings, padding_ids, args):
    llm.to(args.device)
    embeddings = llm.get_input_embeddings().weight
    num_feat = llm.get_input_embeddings().embedding_dim
    vocab_size = llm.get_input_embeddings().num_embeddings
    tpg = None
    if tpgan_flag == True:
        if tpg_flag == False:
            tpg, smp = pretrain_tpgan(embeddings, num_feat, vocab_size, llm, train_loader, padding_ids, max_position_embeddings, args)
        else:
            tpg = GRUGenerator(num_feat, args.hid_size, vocab_size, padding_ids, args.hid_size, args.device)
            tpg.load_state_dict(torch.load(model_caches_path + f"{args.dataset}_tpg_{args.llm}_{args.tpgan_cache}.pth", map_location='cuda:0'))
            tpg.to(args.device)
            embedding_layer_copy = copy.deepcopy(llm.get_input_embeddings())
            smp = Sampler(args.hid_size, embedding_layer_copy, padding_ids, max_position_embeddings, args.gru_layers, args.topk_sim_words, args.device)
            smp.load_state_dict(torch.load(model_caches_path + f"{args.dataset}_smp_{args.llm}_{args.tpsmp_cache}.pth", map_location='cuda:0'))
            smp.to(args.device)
            print(f"-------------TPG loaded as {args.dataset}_tpg_{args.llm}_{args.tpgan_cache}.pth; SMP loaded as {args.dataset}_smp_{args.llm}_{args.tpsmp_cache}.pth-------------")
        if tpr_flag == False:
            tpg = train_tpr(tpg, smp, embeddings, num_feat, vocab_size, llm, mini_loader, padding_ids, args)
        else:
            tpg = GRUGenerator(num_feat, args.hid_size, vocab_size, padding_ids, args.hid_size, args.device)
            tpg.load_state_dict(torch.load(model_caches_path + f"{args.dataset}_tpr_{args.llm}_{args.tprep_cache}.pth", map_location='cuda:0'))
            tpg.to(args.device)
            print(f"-------------TPR loaded as {args.dataset}_tpr_{args.llm}_{args.tprep_cache}.pth-------------")
        tpg.eval()
    llm.train()
    embedding_layer = llm.get_input_embeddings()
    print("-------------Start model training-------------")
    outputs_all = list()
    y_all = list()
    since = time.time()
    for epoch in range(args.epochs):
        total_loss = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(args.device), y.to(args.device)
            input_ids = X['input_ids']
            attention_mask = X['attention_mask']
            y_all.append(y)
            optimizer.zero_grad()
            client_outputs = embedding_layer(input_ids)
            if tpgan_flag == True:
                hidden = tpg.init_hidden(y.shape[0])
                tpg_emb, _ = tpg(input_ids, client_outputs, hidden)
                tpg_emb = tpg_emb.detach()
                fake_indices = torch.argmax(tpg_emb, dim=-1).view(input_ids.shape[0], -1)
                fake_indices[input_ids == padding_ids] = padding_ids
                outputs = llm.forward(input_ids=fake_indices, attention_mask=attention_mask)
            else:
                outputs = llm.forward(input_ids=input_ids, attention_mask=attention_mask)
            outputs_all.append(outputs.logits)
            loss = nn.functional.cross_entropy(outputs.logits, y)
            loss.backward()
            total_loss += loss
            optimizer.step()
            lr_scheduler.step()
        epoch_loss = total_loss / len(train_loader)
        if i % 1 == 0:
            print(f"Epoch {epoch}, loss: {epoch_loss.item()}")
    llm.save_pretrained(model_caches_path + f"{args.dataset}_finetuned_s_{args.llm}_{time_today}.pth")
    cost = time.time() - since
    print(f"Training finished, model saved as {args.dataset}_finetuned_c_{args.llm}_{time_today}.pth and {args.dataset}_finetuned_s_{args.llm}_{time_today}.pth, time cost: {cost}")
    return llm, tpg


def eval_model(test_loader, tpgan_flag, args, padding_ids, llm=None, generator=None):
    if llm is not None:
        pass
    else:
        peft_path = model_caches_path + f"{args.dataset}_finetuned_s_{args.llm}_{time_today}.pth"
        if args.llm == "qwen":
            llm_path = model_path + f"Qwen2-0.5B-Instruct"
            llm_origin = AutoModelForCausalLM.from_pretrained(llm_path)
        if args.llm == "roberta":
            llm_origin = RobertaModel.from_pretrained(llm_path)
            llm_path = model_path + f"roberta-base"
        if args.llm == "roberta-large":
            llm_origin = RobertaModel.from_pretrained(llm_path)
            llm_path = model_path + f"roberta-large"
        if args.llm == "bert":
            llm_origin = BertModel.from_pretrained(llm_path)
            llm_path = model_path + f"bert-base-uncased"
        
        llm = PeftModel.from_pretrained(llm_origin, peft_path)
        llm.to(args.device)

    num_feat = llm.get_input_embeddings().embedding_dim
    vocab_size = llm.get_input_embeddings().num_embeddings
    if tpgan_flag == True:
        if generator is not None:
            tpg = generator
        else:
            tpg_path = None
            if args.use_pr_cache == 't':
                tpg_path = model_caches_path + f"{args.dataset}_tpr_{args.llm}_{args.tprep_cache}.pth"
            elif args.use_pg_cache == 't':
                tpg_path = model_caches_path + f"{args.dataset}_tpg_{args.llm}_{args.tpgan_cache}.pth"
            if tpg_path is not None:
                tpg = GRUGenerator(num_feat, args.hid_size, vocab_size, padding_ids, args.hid_size, args.device)
                tpg.load_state_dict(torch.load(tpg_path, map_location='cuda:0'))
            else:
                raise Exception("No cache model loaded, please select a cache model of tpg or tpr, or train a new model")
            tpg.to(args.device)
        tpg.eval()
    llm.eval()
    

    total = 0
    correct = 0
    total_tokens_num = 0
    different_tokens_num = 0
    embedding_layer = llm.get_input_embeddings()
    for _, (X, y) in enumerate(test_loader):
        X, y = X.to(args.device), y.to(args.device)
        input_ids = X['input_ids']
        attention_mask = X['attention_mask']
        with torch.no_grad():
            client_outputs = embedding_layer(input_ids)
            batch, _, dim = client_outputs.shape
            if tpgan_flag == True:
                hidden = tpg.init_hidden(y.shape[0])
                tpg_emb, _ = tpg(input_ids, client_outputs, hidden)
                fake_indices = torch.argmax(tpg_emb, dim=-1).view(input_ids.shape[0], -1)
                fake_indices[input_ids == padding_ids] = padding_ids
                fake_emb = embedding_layer(fake_indices)

                outputs = llm(input_ids=fake_indices, attention_mask=attention_mask)
                dif_tokens_num, tokens_num = eia_accuracy(input_ids, fake_indices)
                total_tokens_num += tokens_num
                different_tokens_num += dif_tokens_num
            else:
                outputs = llm(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)

        total += y.size(0)
        correct += (predicted == y).sum().item()
    acc = correct / total
    if tpgan_flag == True:
        emp = different_tokens_num / total_tokens_num
        message = f"Accuracy: {acc}; Empirical Privacy: {emp}"
        print(message)
        write_logger(args, message)
    else:
        message = f"Accuracy: {acc}"
        print(message)
        write_logger(args, message)
