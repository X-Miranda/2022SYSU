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


# 4. 解码策略相关函数 (新增)
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

    # 转换为token并过滤特殊token
    trg_tokens = []
    for idx in trg_indices[1:-1]:  # 跳过<sos>和<eos>
        token = [k for k, v in trg_vocab.items() if v == idx]
        if token:
            trg_tokens.append(token[0])

    return trg_tokens


def evaluate_with_strategy(model, iterator, src_vocab, trg_vocab, device):
    """使用统一解码策略评估模型"""
    model.eval()
    all_trg = []
    all_pred = []

    with torch.no_grad():
        for src, trg in tqdm(iterator, desc="Evaluating"):
            src = src.permute(1, 0).to(device)
            trg = trg.permute(1, 0).to(device)

            batch_size = src.shape[1]
            for i in range(batch_size):
                src_sentence = src[:, i:i + 1]
                trg_sentence = trg[:, i]

                # 使用独立解码
                pred_tokens = greedy_decode(model, src_sentence, src_vocab, trg_vocab, device)

                # 处理目标句子
                trg_indices = trg_sentence.cpu().tolist()
                trg_tokens = []
                for idx in trg_indices[1:-1]:  # 跳过<sos>和<eos>
                    if idx == 0:  # 跳过<pad>
                        continue
                    token = [k for k, v in trg_vocab.items() if v == idx]
                    if token:
                        trg_tokens.append(token[0])

                # 收集结果
                all_pred.append(pred_tokens)
                all_trg.append([trg_tokens])  # 注意：BLEU需要参考翻译列表

    # 计算BLEU
    smoothie = SmoothingFunction().method4
    bleu = corpus_bleu(all_trg, all_pred, smoothing_function=smoothie) if all_pred else 0.0
    return bleu


def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=50):
    model.eval()

    tokens = [tok.lower() for tok in jieba.cut(sentence)]
    indices = [src_vocab.get(token, src_vocab['<unk>']) for token in tokens]
    indices = [src_vocab['<sos>']] + indices + [src_vocab['<eos>']]

    sentence_tensor = torch.LongTensor(indices).unsqueeze(1).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(sentence_tensor)

    trg_indices = [trg_vocab['<sos>']]
    attentions = []

    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indices[-1]]).to(device)

        with torch.no_grad():
            output, hidden, attention = model.decoder(
                trg_tensor, hidden, encoder_outputs, return_attention=True
            )
            attentions.append(attention.squeeze().cpu().numpy())

        pred_token = output.argmax(1).item()
        trg_indices.append(pred_token)

        if pred_token == trg_vocab['<eos>']:
            break

    # 转换为token并过滤特殊token
    trg_tokens = []
    for idx in trg_indices[1:-1]:  # 跳过<sos>和<eos>
        token = [k for k, v in trg_vocab.items() if v == idx]
        if token:
            trg_tokens.append(token[0])

    attention_matrix = torch.tensor(attentions).to(device) if attentions else torch.zeros(1, len(indices)).to(device)

    return trg_tokens, attention_matrix


def display_attention(sentence, translation, attention, strategy_name):
    attention = attention.squeeze(1).cpu().detach().numpy()
    if attention.max() == attention.min() == 0:
        print("警告：注意力权重全为零！")
        return

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attention, cmap='bone')
    fig.colorbar(cax)

    src_tokens = ['<sos>'] + [t.lower() for t in jieba.cut(sentence)] + ['<eos>']
    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45)
    ax.set_yticks(range(len(translation)))
    ax.set_yticklabels(translation)
    ax.set_title(f'{strategy_name} Attention')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.tight_layout()
    filename = f"{strategy_name}_attention.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    print(f"注意力图像已保存至: {filename}")


# 5. 绘制训练策略对比图
def plot_training_strategy_comparison(results, strategies):
    plt.figure(figsize=(18, 12))

    # 1. 训练损失曲线
    plt.subplot(2, 2, 1)
    for strategy in strategies:
        name = strategy['name']
        epochs = range(1, len(results[name]['train_loss']) + 1)
        plt.plot(epochs, results[name]['train_loss'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)

    # 2. 验证损失曲线
    plt.subplot(2, 2, 2)
    for strategy in strategies:
        name = strategy['name']
        epochs = range(1, len(results[name]['valid_loss']) + 1)
        plt.plot(epochs, results[name]['valid_loss'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)

    # 3. 验证BLEU分数曲线
    plt.subplot(2, 2, 3)
    for strategy in strategies:
        name = strategy['name']
        epochs = range(1, len(results[name]['valid_bleu']) + 1)
        plt.plot(epochs, results[name]['valid_bleu'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.title('Validation BLEU Score Comparison')
    plt.legend()
    plt.grid(True)

    # 4. 最终测试结果对比
    plt.subplot(2, 2, 4)
    names = [s['name'] for s in strategies]
    bleu_scores = [results[name]['test_bleu'] for name in names]

    plt.bar(names, bleu_scores)
    plt.xlabel('Training Strategy')
    plt.ylabel('BLEU Score')
    plt.title('Test BLEU Score Comparison')

    for i, v in enumerate(bleu_scores):
        plt.text(i, v + 0.001, f"{v:.4f}", ha='center')

    plt.tight_layout()
    plt.savefig('training_strategy_comparison.png')
    plt.close()
    print("训练策略对比图已保存为 'training_strategy_comparison.png'")


# 6. 主程序
if __name__ == "__main__":
    # 参数设置
    DROPOUT = 0.5
    CLIP = 3
    BATCH_SIZE = 32
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    EMB_DIM = 512
    N_EPOCHS = 30
    ATTENTION_TYPE = 'additive'  # 固定使用additive注意力机制

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

    # 定义训练策略
    training_strategies = [
        {'name': 'Teacher_Forcing', 'ratio': 1.0},
        {'name': 'Free_Running', 'ratio': 0.0},
        {'name': 'Mixed', 'ratio': 0.5}
    ]

    # 初始化结果字典
    results = {}
    for strategy in training_strategies:
        results[strategy['name']] = {
            'train_loss': [],
            'valid_loss': [],
            'valid_bleu': [],
            'test_loss': None,
            'test_bleu': None
        }


    # 初始化权重函数
    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)


    # 比较不同训练策略
    for strategy in training_strategies:
        print(f"\n=== 训练策略: {strategy['name']} (Teacher Forcing Ratio={strategy['ratio']}) ===")

        # 创建模型 (固定使用additive注意力机制)
        enc = Encoder(len(src_vocab), EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DROPOUT)
        attn = Attention(ENC_HID_DIM * 2, DEC_HID_DIM, attn_type=ATTENTION_TYPE)
        dec = Decoder(len(trg_vocab), EMB_DIM, ENC_HID_DIM * 2, DEC_HID_DIM, DROPOUT, attn)
        model = Seq2Seq(enc, dec, device).to(device)
        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        best_valid_loss = float('inf')
        best_bleu = 0.0
        no_improve = 0

        for epoch in range(N_EPOCHS):
            # 使用当前策略的teacher forcing比例
            train_loss = train(model, train_loader, optimizer, criterion, CLIP,
                               teacher_forcing_ratio=strategy['ratio'])

            # 计算验证集损失和BLEU
            valid_loss = evaluate(model, valid_loader, criterion)[0]
            valid_bleu = evaluate_with_strategy(model, valid_loader, src_vocab, trg_vocab, device)

            scheduler.step()

            # 记录当前epoch的指标
            results[strategy['name']]['train_loss'].append(train_loss)
            results[strategy['name']]['valid_loss'].append(valid_loss)
            results[strategy['name']]['valid_bleu'].append(valid_bleu)

            print(f'Epoch: {epoch + 1:02} | Strategy: {strategy["name"]}')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f}')
            print(f'\t Val. BLEU: {valid_bleu:.4f}')  # 显示4位小数

            if valid_bleu > best_bleu:
                best_bleu = valid_bleu
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), f'best-model-{strategy["name"]}.pt')
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 5:
                    print(f"早停触发，最佳BLEU: {best_bleu:.4f}")
                    break

        # 测试最佳模型
        model.load_state_dict(torch.load(f'best-model-{strategy["name"]}.pt'))
        test_loss = evaluate(model, test_loader, criterion)[0]
        test_bleu = evaluate_with_strategy(model, test_loader, src_vocab, trg_vocab, device)

        # 记录最终测试结果
        results[strategy['name']]['test_loss'] = test_loss
        results[strategy['name']]['test_bleu'] = test_bleu

        print(f'\n{strategy["name"]} 测试结果:')
        print(f'\tTest Loss: {test_loss:.3f}')
        print(f'\tTest BLEU: {test_bleu:.4f}')  # 显示4位小数

        # 示例翻译
        example_sentence = "1929年还是1989年?"
        translation, attention = translate_sentence(example_sentence, src_vocab, trg_vocab, model, device)
        print(f'\n{strategy["name"]} 示例翻译:')
        print(f'Original: {example_sentence}')
        print(f'Translation: {" ".join(translation)}')
        display_attention(example_sentence, translation, attention, strategy['name'])

    # 绘制训练策略对比图
    plot_training_strategy_comparison(results, training_strategies)

    # 打印最终结果
    print("\n不同训练策略性能比较:")
    for strategy in training_strategies:
        name = strategy['name']
        print(f"{name}:")
        print(f"\tTest Loss: {results[name]['test_loss']:.4f}")
        print(f"\tTest BLEU: {results[name]['test_bleu']:.4f}")