import os
import pickle

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel, TfidfModel, CoherenceModel
from matplotlib import pyplot as plt, rcParams
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import networkx as nx
from itertools import combinations

# 配置区域
gba_tokenized_dir = r'.\分词结果'  # 大湾区词库
fgw_tokenized_dir = r'D:\self\work\pythonProject\大湾区\fagaiwei\分词结果'  # 发改委词库

# 配置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

def load_documents(tokenized_dir):
    """加载分词文件"""
    documents = []
    for filename in os.listdir(tokenized_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(tokenized_dir, filename), 'r', encoding='utf-8') as f:
                documents.append(f.read().split())
    return documents


def cross_corpus_tfidf_filter(gba_docs, fgw_docs, top_n=4000, high_freq_threshold=0.001):
    """
    改进的跨语料库关键词筛选：
    1. 使用相对频率代替TF-IDF
    2. 动态调整阈值
    3. 确保至少保留政策核心术语
    4. 过滤掉词频为1的词
    5. 过滤掉在发改委和大湾区都高频的词
    """

    def get_term_freq(docs):
        freq = defaultdict(int)
        for doc in docs:
            for word in doc:
                freq[word] += 1
        total = sum(freq.values())
        return {k: v / total for k, v in freq.items()}

    gba_freq = get_term_freq(gba_docs)
    fgw_freq = get_term_freq(fgw_docs)

    # 找出在两个语料库中都高频的词
    both_high_freq_terms = set()
    for term in gba_freq:
        if term in fgw_freq and gba_freq[term] > high_freq_threshold and fgw_freq[term] > high_freq_threshold:
            both_high_freq_terms.add(term)

    keep_terms = []
    base_ratio = len(fgw_docs) / len(gba_docs)
    policy_stops = {'总', '新', '作工', '土建', '书书书', '第年'}

    # 筛选条件
    for term in gba_freq:
        if term not in policy_stops:
            if term in both_high_freq_terms:
                print(term)
                continue
            if term not in fgw_freq:  # 大湾区独有词
                keep_terms.append((term, gba_freq[term]))
            else:
                ratio = (gba_freq[term] / fgw_freq[term])
                if ratio > 1.5:  # 大湾区相对频率高50%以上
                    keep_terms.append((term, gba_freq[term]))

    # 按频率排序
    keep_terms.sort(key=lambda x: x[1], reverse=True)
    # return [term for term, _ in keep_terms[:top_n]]
    return [term for term, _ in keep_terms]


def apply_c_tfidf(lda_model, corpus, dictionary, texts, topn=10):
    """修正版的c-TF-IDF优化"""
    # 1. 文档分组（保留加权信息）
    topic_docs = defaultdict(list)
    for doc_idx, doc in enumerate(corpus):
        topic_probs = lda_model.get_document_topics(doc)
        if topic_probs:
            dominant_topic = max(topic_probs, key=lambda x: x[1])[0]
            topic_docs[dominant_topic].append(doc_idx)

    # 2. 构建主题-词频矩阵（保留权重）
    id2word = dictionary.id2token
    topic_term_freq = defaultdict(lambda: defaultdict(float))

    for topic, doc_indices in topic_docs.items():
        for doc_idx in doc_indices:
            for word_id, freq in corpus[doc_idx]:  # 使用加权后的词频
                word = id2word[word_id]
                topic_term_freq[topic][word] += freq

    # 3. 计算c-TF-IDF
    all_topics = sorted(topic_term_freq.keys())
    all_terms = list(dictionary.token2id.keys())

    # 构建矩阵：行=主题，列=词
    tf_matrix = np.zeros((len(all_topics), len(all_terms)))
    df_vector = np.zeros(len(all_terms))

    for i, topic in enumerate(all_topics):
        for j, term in enumerate(all_terms):
            tf_matrix[i, j] = topic_term_freq[topic].get(term, 0)
            if term in topic_term_freq[topic]:
                df_vector[j] += 1

    # 平滑处理避免除零
    df_vector = np.where(df_vector == 0, 1, df_vector)
    idf_vector = np.log(len(all_topics) / df_vector) + 1
    c_tfidf_matrix = tf_matrix * idf_vector

    # 4. 提取关键词
    optimized_topics = {}
    for i, topic in enumerate(all_topics):
        topic_scores = c_tfidf_matrix[i]
        top_indices = np.argsort(topic_scores)[-topn:][::-1]
        optimized_topics[topic] = [all_terms[idx] for idx in top_indices
                                   if topic_scores[idx] > 0]

    # 5. 计算一致性（使用过滤后的文本）
    valid_topics = [words for words in optimized_topics.values() if words]
    if not valid_topics:
        return optimized_topics, 0.0

    coherence_model = CoherenceModel(
        topics=valid_topics,
        texts=texts,  # 确保与原始LDA一致
        dictionary=dictionary,
        coherence='c_v'
    )
    return optimized_topics, coherence_model.get_coherence()

def visualize_topic_keywords(lda_model, dictionary, num_topics, topn=15):
    """
    可视化每个主题下的关键词及其概率分布
    """
    plt.figure(figsize=(15, 10))

    for topic_id in range(num_topics):
        plt.subplot(num_topics // 4 + 1, 4, topic_id + 1)

        # 获取主题词及其概率
        topic_terms = lda_model.show_topic(topic_id, topn=topn)
        terms = [term for term, _ in topic_terms]
        probs = [prob for _, prob in topic_terms]

        # 绘制条形图
        plt.barh(terms, probs, color='skyblue')
        plt.title(f'主题 {topic_id + 1}')
        plt.xlabel('概率')
        plt.gca().invert_yaxis()  # 重要词在上方

    plt.tight_layout()
    plt.show()


def save_topic_keywords(lda_model, num_topics, filename="topic_keywords.csv", topn=15):
    """
    将主题关键词及概率保存到CSV文件
    """
    import csv

    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['主题ID', '关键词', '概率'])

        for topic_id in range(num_topics):
            topic_terms = lda_model.show_topic(topic_id, topn=topn)
            for term, prob in topic_terms:
                writer.writerow([topic_id + 1, term, f"{prob:.4f}"])



if __name__ == '__main__':
    # 1. 加载数据
    gba_documents = load_documents(gba_tokenized_dir)
    fgw_documents = load_documents(fgw_tokenized_dir)
    combined_docs = gba_documents + fgw_documents

    # 2. 执行跨语料库TF-IDF筛选
    keep_terms = cross_corpus_tfidf_filter(gba_documents, fgw_documents)
    print(f"保留大湾区特有高频词数量: {len(keep_terms)}")

    # 3. 构建过滤后的词典
    dictionary = Dictionary(gba_documents + fgw_documents)
    keep_ids = [dictionary.token2id[term] for term in keep_terms if term in dictionary.token2id]

    # 重建词典确保ID连续
    dictionary.filter_tokens(good_ids=keep_ids)
    dictionary.compactify()
    print(f"过滤后词典大小: {len(dictionary)}")

    # 过滤极端词（可选）
    dictionary.filter_extremes(no_below=2)
    print(f"再过滤后词典大小: {len(dictionary)}")

    # 4. 构建大湾区语料
    corpus = [dictionary.doc2bow(doc) for doc in gba_documents]

    # 过滤后的分词结果（仅保留词典中的词）
    filtered_gba_documents = [
        [word for word in doc if word in dictionary.token2id]
        for doc in gba_documents
    ]

    total_words = sum(len(doc) for doc in gba_documents)
    covered_words = sum(len(dictionary.doc2bow(doc)) for doc in gba_documents)
    print(f"大湾区总词汇覆盖率: {covered_words}/{total_words} ({covered_words / total_words:.1%})")

    min_topics, max_topics, step = 1, 20, 1
    topic_range = range(min_topics, max_topics + 1, step)
    original_metrics = {'perplexity': [], 'coherence': []}
    optimized_metrics = {'perplexity': [], 'coherence': []}
    all_topics = {}

    for num_topics in topic_range:
        print(f"\n=== 主题数: {num_topics} ===")

        # 原始LDA
        lda = LdaModel(
            corpus, num_topics=num_topics,
            id2word=dictionary, passes=100, alpha='auto'
        )
        perplexity = np.exp(-lda.log_perplexity(corpus))
        coherence = CoherenceModel(
            model=lda, texts=filtered_gba_documents,
            dictionary=dictionary, coherence='c_v'
        ).get_coherence()
        original_metrics['perplexity'].append(perplexity)
        original_metrics['coherence'].append(coherence)

        # c-TF-IDF优化
        optimized_topics, opt_coherence = apply_c_tfidf(lda, corpus, dictionary, gba_documents)
        opt_perplexity = np.exp(-lda.log_perplexity(corpus))
        optimized_metrics['perplexity'].append(opt_perplexity)  # 困惑度不变
        optimized_metrics['coherence'].append(opt_coherence)
        all_topics[num_topics] = (lda, optimized_topics)

        # 打印当前主题数指标
        print(f"原始LDA - 困惑度: {perplexity:.1f}, c_v: {coherence:.3f}")
        print(f"优化后LDA - c_v: {opt_coherence:.3f}")

        if num_topics >5:
            for topic_id in range(num_topics):
                # 原始主题词
                original_words = [word for word, _ in lda.show_topic(topic_id, topn=10)]
                # 单独计算当前主题c_v
                topic_coherence = CoherenceModel(
                    topics=[original_words],
                    texts=filtered_gba_documents,
                    dictionary=dictionary,
                    coherence='c_v'
                ).get_coherence()

                print(f"主题 {topic_id + 1} (c_v={topic_coherence:.3f})   {original_words}")

            print(f"\n")

            for topic_id in range(num_topics):
                # 优化主题词
                optimized_words = optimized_topics.get(topic_id, [])

                # 单独计算当前主题c_v
                topic_coherence = CoherenceModel(
                    topics=[optimized_words],
                    texts=filtered_gba_documents,
                    dictionary=dictionary,
                    coherence='c_v'
                ).get_coherence()
                print(f"主题 {topic_id + 1} (c_v={topic_coherence:.3f})   {optimized_words}")

        # 可视化主题关键词
        visualize_topic_keywords(lda, dictionary, num_topics)

        # 保存到CSV文件
        save_topic_keywords(lda, num_topics, f"topic_keywords_{num_topics}.csv")

    # 可视化对比
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(topic_range, original_metrics['perplexity'], 'b-o', label='原始LDA')
    ax1.plot(topic_range, optimized_metrics['perplexity'], 'g--^', label='优化后LDA')
    ax1.set_ylabel("困惑值")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(topic_range, original_metrics['coherence'], 'r-s', label='原始LDA')
    ax2.plot(topic_range, optimized_metrics['coherence'], 'm--d', label='优化后LDA')
    ax2.set_xlabel("主题数")
    ax2.set_ylabel("一致性(c_v)")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle("LDA模型优化前后指标对比")
    plt.tight_layout()
    plt.show()


