
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 加载数据
df = pd.read_csv(r"/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")

# 将情感标签转换为二进制值：positive -> 1, negative -> 0
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# 初始化TF-IDF向量化器
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# 拟合并转换训练数据
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# 仅转换测试数据
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 初始化逻辑回归模型
logistic_model = LogisticRegression()

# 训练模型
logistic_model.fit(X_train_tfidf, y_train)

# 预测测试集
y_pred_logistic = logistic_model.predict(X_test_tfidf)

# 计算准确率
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print(f"逻辑回归模型的准确率: {accuracy_logistic:.4f}")

# 初始化SVM模型
svm_model = SVC(kernel='linear')

# 训练模型
svm_model.fit(X_train_tfidf, y_train)

# 预测测试集
y_pred_svm = svm_model.predict(X_test_tfidf)

# 计算准确率
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM模型的准确率: {accuracy_svm:.4f}")

print(f"逻辑回归模型的准确率: {accuracy_logistic:.4f}")
print(f"SVM模型的准确率: {accuracy_svm:.4f}")

if accuracy_logistic > accuracy_svm:
    print("逻辑回归模型的准确率高于SVM模型。")
elif accuracy_logistic < accuracy_svm:
    print("SVM模型的准确率高于逻辑回归模型。")
else:
    print("两个模型的准确率相同。")