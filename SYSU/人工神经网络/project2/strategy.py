import json
import jieba
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import os
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
import numpy as np

# 设置随机种子保证可重复性
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

nltk.data.path.append(r'C:\Users\85013\.conda\envs\ainet\nltk_data')

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
rcParams['axes.unicode_minus'] = False


# 1. 数据预处理
class NMTDataset(Dataset):
    def __init__(self, file_path, src_vocab=None, trg_vocab=None, max_len=50, build_vocab=False):
        self.src_sentences = []
        self.trg_sentences = []

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                src_tokens = [tok.lower() for tok in jieba.cut(data['zh'])]
                trg_tokens = [tok.lower() for tok in word_tokenize(data['en'])]

                if len(src_tokens) <= max_len and len(trg_tokens) <= max_len:
                    self.src_sentences.append(src_tokens)
                    self.trg_sentences.append(trg_tokens)

        if build_vocab:
            print("构建词汇表中...")
            self.src_vocab = self._build_vocab(self.src_sentences)
            self.trg_vocab = self._build_vocab(self.trg_sentences)
            print(f"中文词汇表大小: {len(self.src_vocab)}")
            print(f"英文词汇表大小: {len(self.trg_vocab)}")
        else:
            self.src_vocab = src_vocab
            self.trg_vocab = trg_vocab

    def _build_vocab(self, sentences, min_freq=3):
        counter = Counter()
        for sentence in sentences:
            counter.update(sentence)

        vocab = {
            '<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3
        }
        sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        for token, count in sorted_words:
            if count >= min_freq and len(vocab) < 30000:
                vocab[token] = len(vocab)
        return vocab

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        trg_sentence = self.trg_sentences[idx]

        src_indices = [self.src_vocab.get(token, self.src_vocab['<unk>']) for token in src_sentence]
        trg_indices = [self.trg_vocab.get(token, self.trg_vocab['<unk>']) for token in trg_sentence]

        src_indices = [self.src_vocab['<sos>']] + src_indices + [self.src_vocab['<eos>']]
        trg_indices = [self.trg_vocab['<sos>']] + trg_indices + [self.trg_vocab['<eos>']]

        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(trg_indices, dtype=torch.long)


def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=0, batch_first=True)
    trg_batch = nn.utils.rnn.pad_sequence(trg_batch, padding_value=0, batch_first=True)
    return src_batch, trg_batch


# 2. 模型定义
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_output_dim, dec_hid_dim, attn_type='additive'):
        super().__init__()
        self.enc_output_dim = enc_output_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn_type = attn_type

        if attn_type == 'additive':
            self.attn = nn.Sequential(
                nn.Linear(self.dec_hid_dim + self.enc_output_dim, dec_hid_dim),
                nn.Tanh(),
                nn.Linear(dec_hid_dim, 1)
            )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        batch_size = hidden.size(0)

        if self.attn_type == 'additive':
            hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
            encoder_outputs = encoder_outputs.permute(1, 0, 2)

            combined = torch.cat((hidden, encoder_outputs), dim=2)
            energy = self.attn(combined).squeeze(2)

        attention = self.softmax(energy)
        return attention


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim + enc_hid_dim, dec_hid_dim)

        self.fc_out = nn.Linear(dec_hid_dim + enc_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dec_hid_dim)

    def forward(self, input, hidden, encoder_outputs, return_attention=False):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))

        a = self.attention(hidden, encoder_outputs)

        if return_attention:
            attention_weights = a.clone()

        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)

        rnn_input = torch.cat((embedded, weighted), dim=2)

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        hidden = self.layer_norm(hidden.squeeze(0))

        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        embedded = embedded.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        if return_attention:
            return prediction, hidden, attention_weights
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)

        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs


# 3. 训练和评估函数
def train(model, iterator, optimizer, criterion, clip, teacher_forcing_ratio=0.5):
    model.train()
    epoch_loss = 0

    for src, trg in tqdm(iterator, desc="Training"):
        src = src.permute(1, 0).to(device)
        trg = trg.permute(1, 0).to(device)

        optimizer.zero_grad()

        output = model(src, trg, teacher_forcing_ratio)
        output_dim = output.shape[-1]
        output = output[1:].reshape(-1, output_dim)
        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    all_trg = []
    all_pred = []

    with torch.no_grad():
        for src, trg in tqdm(iterator, desc="Evaluating"):
            src = src.permute(1, 0).to(device)
            trg = trg.permute(1, 0).to(device)

            output = model(src, trg, 0)  # 评估时使用Free Running
            output_dim = output.shape[-1]

            output_loss = output[1:].reshape(-1, output_dim)
            trg_loss = trg[1:].reshape(-1)
            loss = criterion(output_loss, trg_loss)
            epoch_loss += loss.item()

            pred_tokens = output.argmax(-1).cpu().tolist()
            trg_tokens = trg.cpu().tolist()

            batch_size = len(pred_tokens[0]) if pred_tokens else 0
            for i in range(batch_size):
                pred = [str(pred_tokens[j][i]) for j in range(1, len(pred_tokens))]
                ref = [str(trg_tokens[j][i]) for j in range(1, len(trg_tokens))]

                pred = [x for x in pred if x != '0']
                ref = [x for x in ref if x != '0']

                if pred and ref:
                    all_pred.append(pred)
                    all_trg.append([ref])

    smoothie = SmoothingFunction().method4
    bleu = corpus_bleu(all_trg, all_pred, smoothing_function=smoothie) if all_pred else 0.0

    return epoch_loss / len(iterator), bleu


# 4. 解码策略相关函数
def beam_search_decode(model, src, src_vocab, trg_vocab, device, max_len=50, beam_width=5):
    """使用beam search解码句子"""
    model.eval()

    # 编码源句子
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src)

    # 初始beam: (序列, 分数, 隐藏状态)
    start_token = trg_vocab['<sos>']
    beams = [([start_token], 0.0, hidden)]

    for _ in range(max_len):
        candidates = []

        for seq, score, hidden_state in beams:
            # 如果序列以<eos>结束，则保留不再扩展
            if seq[-1] == trg_vocab['<eos>']:
                candidates.append((seq, score, hidden_state))
                continue

            # 获取最后一个token
            last_token = torch.tensor([seq[-1]]).to(device)

            # 解码
            with torch.no_grad():
                output, new_hidden = model.decoder(last_token, hidden_state, encoder_outputs)
                log_probs = F.log_softmax(output, dim=1)

            # 获取top-k候选
            topk_scores, topk_tokens = log_probs.topk(beam_width)
            topk_scores = topk_scores.squeeze().cpu().numpy()
            topk_tokens = topk_tokens.squeeze().cpu().numpy()

            # 创建新候选
            for i in range(beam_width):
                new_seq = seq + [topk_tokens[i]]
                new_score = score + topk_scores[i]
                candidates.append((new_seq, new_score, new_hidden))

        # 按分数排序并选择top beam_width
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

    # 返回分数最高的序列
    best_sequence = beams[0][0]
    trg_tokens = [list(trg_vocab.keys())[list(trg_vocab.values()).index(idx)]
                  for idx in best_sequence[1:-1]]  # 跳过<sos>和<eos>

    return trg_tokens


def evaluate_with_strategy(model, iterator, strategy='greedy', beam_width=5):
    """使用指定解码策略评估模型"""
    model.eval()
    all_trg = []
    all_pred = []

    with torch.no_grad():
        for src, trg in tqdm(iterator, desc=f"Evaluating ({strategy})"):
            src = src.permute(1, 0).to(device)
            trg = trg.permute(1, 0).to(device)

            batch_size = src.shape[1]
            for i in range(batch_size):
                src_sentence = src[:, i:i + 1]
                trg_sentence = trg[:, i]

                # 根据策略选择解码方法
                if strategy == 'greedy':
                    pred_tokens = greedy_decode(model, src_sentence, src_vocab, trg_vocab, device)
                elif strategy == 'beam':
                    pred_tokens = beam_search_decode(
                        model, src_sentence, src_vocab, trg_vocab, device, beam_width=beam_width
                    )

                # 处理目标句子
                trg_indices = trg_sentence.cpu().tolist()
                trg_tokens = [list(trg_vocab.keys())[list(trg_vocab.values()).index(idx)]
                              for idx in trg_indices if idx != 0 and idx != trg_vocab['<sos>']]

                # 收集结果
                all_pred.append(pred_tokens)
                all_trg.append([trg_tokens])  # 注意：BLEU需要参考翻译列表

    # 计算BLEU
    smoothie = SmoothingFunction().method4
    bleu = corpus_bleu(all_trg, all_pred, smoothing_function=smoothie) if all_pred else 0.0
    return bleu


def greedy_decode(model, src_tensor, src_vocab, trg_vocab, device, max_len=50):
    """使用greedy解码句子"""
    model.eval()

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

    trg_indices = [trg_vocab['<sos>']]

    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indices[-1]]).to(device)

        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)

        pred_token = output.argmax(1).item()
        trg_indices.append(pred_token)

        if pred_token == trg_vocab['<eos>']:
            break

    trg_tokens = [list(trg_vocab.keys())[list(trg_vocab.values()).index(idx)]
                  for idx in trg_indices[1:-1]]

    return trg_tokens


def prepare_sentence(sentence, vocab, device):
    """准备句子张量"""
    tokens = [tok.lower() for tok in jieba.cut(sentence)]
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    indices = [vocab['<sos>']] + indices + [vocab['<eos>']]
    return torch.LongTensor(indices).unsqueeze(1).to(device)


def display_decoding_comparison(example_sentence, greedy_translation, beam_translation):
    """显示解码策略对比结果"""
    print("\n解码策略对比示例:")
    print(f"源句子: {example_sentence}")
    print(f"Greedy解码: {' '.join(greedy_translation)}")
    print(f"Beam Search: {' '.join(beam_translation)}")


def plot_decoding_strategy_comparison(greedy_bleu, beam_bleu):
    """绘制解码策略对比图"""
    plt.figure(figsize=(10, 6))
    strategies = ['Greedy', 'Beam Search']
    bleu_scores = [greedy_bleu, beam_bleu]

    plt.bar(strategies, bleu_scores, color=['blue', 'green'])
    plt.title('解码策略BLEU对比')
    plt.ylabel('BLEU Score')

    for i, v in enumerate(bleu_scores):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')

    plt.savefig('decoding_strategy_comparison.png')
    plt.close()
    print("解码策略对比图已保存为 'decoding_strategy_comparison.png'")


# 5. 主程序
if __name__ == "__main__":
    # 参数设置
    DROPOUT = 0.5
    CLIP = 3
    BATCH_SIZE = 32
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    EMB_DIM = 512
    N_EPOCHS = 30
    TEACHER_FORCING_RATIO = 1.0  # 固定使用Teacher Forcing
    ATTENTION_TYPE = 'additive'  # 固定使用additive注意力机制
    BEAM_WIDTH = 5  # Beam search的宽度

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    print("加载数据中...")
    train_dataset = NMTDataset('train_10k.jsonl', build_vocab=True)
    src_vocab = train_dataset.src_vocab
    trg_vocab = train_dataset.trg_vocab

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_dataset = NMTDataset('valid.jsonl', src_vocab, trg_vocab)
    test_dataset = NMTDataset('test.jsonl', src_vocab, trg_vocab)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)


    # 初始化权重函数
    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)


    # 创建模型 (固定使用additive注意力机制)
    print("\n=== 创建模型 ===")
    enc = Encoder(len(src_vocab), EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DROPOUT)
    attn = Attention(ENC_HID_DIM * 2, DEC_HID_DIM, attn_type=ATTENTION_TYPE)
    dec = Decoder(len(trg_vocab), EMB_DIM, ENC_HID_DIM * 2, DEC_HID_DIM, DROPOUT, attn)
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 训练模型 (使用Teacher Forcing)
    print("\n=== 训练模型 (Teacher Forcing) ===")
    best_valid_loss = float('inf')
    best_bleu = 0.0
    no_improve = 0
    train_losses = []
    valid_losses = []
    valid_greedy_bleus = []  # 改为分别记录两种策略
    valid_beam_bleus = []  # 改为分别记录两种策略

    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, CLIP,
                           teacher_forcing_ratio=TEACHER_FORCING_RATIO)

        # 使用原始evaluate函数获取损失
        valid_loss, _ = evaluate(model, valid_loader, criterion)

        # 分别计算两种解码策略的BLEU
        valid_greedy_bleu = evaluate_with_strategy(model, valid_loader, strategy='greedy')
        valid_beam_bleu = evaluate_with_strategy(model, valid_loader, strategy='beam', beam_width=BEAM_WIDTH)

        scheduler.step()

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_greedy_bleus.append(valid_greedy_bleu)
        valid_beam_bleus.append(valid_beam_bleu)

        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')
        print(f'\t Val. Greedy BLEU: {valid_greedy_bleu:.4f}')
        print(f'\t Val. Beam BLEU: {valid_beam_bleu:.4f}')

        # 使用Greedy BLEU作为早停标准
        if valid_greedy_bleu > best_bleu:
            best_bleu = valid_greedy_bleu
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best-model.pt')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 5:
                print(f"早停触发，最佳Greedy BLEU: {best_bleu:.4f}")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load('best-model.pt'))

    # 在测试集上对比
    test_greedy_bleu = evaluate_with_strategy(model, test_loader, strategy='greedy')
    test_beam_bleu = evaluate_with_strategy(model, test_loader, strategy='beam', beam_width=BEAM_WIDTH)

    print(f"\n测试集Greedy解码BLEU: {test_greedy_bleu:.4f}")
    print(f"测试集Beam Search BLEU: {test_beam_bleu:.4f}")

    # 示例对比
    example_sentence = "1929年还是1989年?"
    src_tensor = prepare_sentence(example_sentence, src_vocab, device)

    # Greedy解码
    greedy_translation = greedy_decode(model, src_tensor, src_vocab, trg_vocab, device)

    # Beam search解码
    beam_translation = beam_search_decode(
        model, src_tensor, src_vocab, trg_vocab, device, beam_width=BEAM_WIDTH
    )

    display_decoding_comparison(example_sentence, greedy_translation, beam_translation)

    # 绘制解码策略对比图
    plot_decoding_strategy_comparison(test_greedy_bleu, test_beam_bleu)

    # 绘制训练曲线 - 添加双策略BLEU曲线
    plt.figure(figsize=(15, 10))

    # 训练损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)

    # 验证损失曲线
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(valid_losses) + 1), valid_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.grid(True)

    # 双解码策略BLEU曲线
    plt.subplot(2, 2, 3)
    epochs = range(1, len(valid_greedy_bleus) + 1)
    plt.plot(epochs, valid_greedy_bleus, label='Greedy')
    plt.plot(epochs, valid_beam_bleus, label='Beam Search')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.title('Validation BLEU (by Decoding Strategy)')
    plt.legend()
    plt.grid(True)

    # 最终测试结果对比
    plt.subplot(2, 2, 4)
    strategies = ['Greedy', 'Beam Search']
    bleu_scores = [test_greedy_bleu, test_beam_bleu]
    plt.bar(strategies, bleu_scores)
    plt.ylabel('BLEU Score')
    plt.title('Test BLEU Comparison')

    # 添加数值标签
    for i, v in enumerate(bleu_scores):
        plt.text(i, v + 0.001, f"{v:.4f}", ha='center')

    plt.tight_layout()
    plt.savefig('training_and_decoding_results.png')
    plt.close()
    print("训练和解码结果图已保存为 'training_and_decoding_results.png'")
    print("\n实验完成!")