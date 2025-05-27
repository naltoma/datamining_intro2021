import json, pickle
import pandas as pd

# ラベル辞書
label2id = {"entailment": 0, "contradiction": 1, "neutral": 2}
id2label = {v: k for k, v in label2id.items()}

# pickleファイルからの読み込み
def get_data(filename):
    # 学習データを読み込む
    with open(filename, "rb") as f:
        train_data = pickle.load(f)

    # BoWのための分かち書き, word2vecデータ, labelを読み取る
    sent1 = [ex["sent1"] for ex in train_data]
    sent2 = [ex["sent2"] for ex in train_data]
    bow = [ex["bow_input"] for ex in train_data]
    w2v = [ex["w2v_input"] for ex in train_data]
    label = [label2id[ex["label"]] for ex in train_data]
    df = pd.DataFrame({"sent1":sent1,
                       "sent2":sent2,
                       "bow":bow,
                       "w2v":w2v,
                       "label":label})
    return df

# 評価出力用
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 混同行列出力用のコード
def plot_confusion(caption, y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f"Confusion Matrix ({caption})")
    plt.grid(False)
    plt.show()


# 失敗事例分析用のコード
import numpy as np

def show_misclassified_examples(
    sent1_list,  # 前提文のリスト
    sent2_list,  # 仮説文のリスト
    true_labels, # 正解ラベル
    pred_labels, # 予測ラベル
    id2label,    # ラベルID→ラベル名に変換するための辞書
    samples_per_class=3, # ラベル毎の出力件数
    random_seed=None     # デフォルトはNone（ランダム）
):
    """
    モデルの失敗事例を、ラベルごとにN件ランダム抽出して表示する。
    
    - random_seed=None のとき: 毎回異なる結果
    - random_seed=数値 を指定: 再現性あるサンプリング
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # データ構造化
    df = pd.DataFrame({
        "sent1": sent1_list,
        "sent2": sent2_list,
        "true_label": true_labels,
        "pred_label": pred_labels
    })

    # 間違いのみ抽出
    df_wrong = df[df["true_label"] != df["pred_label"]].copy()
    df_wrong["true_label_name"] = df_wrong["true_label"].map(id2label)
    df_wrong["pred_label_name"] = df_wrong["pred_label"].map(id2label)

    # ラベルごとにランダム抽出
    samples = []
    for label_id, label_name in id2label.items():
        sub = df_wrong[df_wrong["true_label"] == label_id]
        if len(sub) == 0:
            continue
        sampled = sub.sample(n=min(len(sub), samples_per_class), replace=False)
        samples.append(sampled)

    result_df = pd.concat(samples).reset_index(drop=True)
    return result_df
