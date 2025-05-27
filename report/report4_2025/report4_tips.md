# レポート4で用意したコードの実行方法
今回は実行結果を予め用意しているため必ずしも個々人で実行する必要は無いが、各自で実行したい場合には以下を参考にしてください。

---
## コードの全体像
分かち書きに時間がかかる（當間環境で約20分）ため、実験コードを実行し直す都度分かち書きからやり直すのは時間がもったいない。そこでここでは分かち書きだけを行い、結果を（pickle形式）ファイルとして保存しておきます。機械学習する側のコードでは、このファイルを読み込むだけの時間で実験を再開することができるようになります。

---
## 事前準備
まず、以下のファイルをPCにダウンロードしておいてください。

- コード: ノートブックやpythonスクリプト
    - 前処理用コード: {download}`preprocessing.ipynb </assets/report4_2025/preprocessing.ipynb>`
    - ライブラリや関数群をまとめた便利ツール: {download}`utils.py </assets/report4_2025/utils.py>`
    - BoWベースの分類器構築コード: {download}`bow.ipynb </assets/report4_2025/bow.ipynb>`
    - word2vecベースの分類器構築コード: {download}`word2vec.ipynb </assets/report4_2025/word2vec.ipynb>`
    - BERTベースの分類器構築コード: {download}`bert.ipynb </assets/report4_2025/bert.ipynb>`
- データセット
    - [Github: JGLUE/datasets/jnil-v1.3/](https://github.com/yahoojapan/JGLUE/tree/main/datasets/jnli-v1.3) にある以下のファイル
        - `train-v1.3.json`（学習用データセット）
        - `val-v1.3.json`（検証用データセット）
        - `test-v1.3.json`（テスト用データセット）

---
## 前処理の実行
前処理は以下の手順で実行できます。
- (1) Google Driveにアクセスし、Colab Notebooks フォルダの中に移動する。課題4用のフォルダ report4 を作成。
- (2) report4フォルダの中に、ダウンロードしておいた `preprocessing.ipynb` をアップロードし、開く。Google Colabで開ければOK。
- (3) 3つのjsonファイルをノートブック内にアップロードする。
- (4) 全てのセルを実行する（約20分）。
- (5) 実行終了後、3つのpklファイルをダウンロードする。
    - preprocessed_train.pkl
    - preprocessed_val.pkl
    - preprocessed_test.pkl

---
## BoWの実行
- 用意するファイル
    - 事前準備で用意した utils.py
    - 前処理で準備した3つの pkl ファイル
    - 学習用ファイル bow.ipynb
- bow.ipynbの補足
    - `bow.ipynb` では、Bag-of-Words で説明変数を用意し、各説明変数のスコアを TF-IDF で採点することで「各文書の特徴ベクトル」を構築し、ロジスティック回帰により分類器を構築している。また分類器を評価するために [分類レポート（classification_report）](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) と [混同行列（ConfusionMatrixDisplay）](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html) により出力すると共に、失敗事例分析用の結果も出力している。なお、全失敗事例を観察したい人向けに `mis_all_df_bow.xlsx` という名前のExcelファイルも出力しているため、必要な人はダウンロードすると良いだろう。

```{hint}
show_misclassified_examples関数の補足。
- 学習データに対する失敗事例分析をするための関数です。全データを参照するのは大変なので、ラベル毎に失敗事例N件をランダムに出力しています。
    - `samples_per_class`: 出力件数。デフォルトで3件出力します。増減したい場合には調整してください。
    - `random_seed`: 乱数生成に用いるシード値。特定の整数を設定すると出力結果が固定されます。毎回ランダム抽出してほしい場合には `None` を設定してください。
```

---
## word2vecの実行
- 用意するファイル
    - 事前準備で用意した utils.py
    - 前処理で準備した3つの pkl ファイル
    - 学習用ファイル word2vec.ipynb
- word2vec.ipynbの補足
    - `word2vec.ipynb` では、spacy(ja_ginza)を用いてトークン毎の分散ベクトルを取得し、それらの平均ベクトルを求めることで特徴ベクトルを構築した。このベクトルを用いて [MLPClassifier（シンプルなNN）](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)で分類器を構築している。

---
## BERTの実行
- 用意するファイル
    - 事前準備で用意した utils.py
    - 前処理で準備した3つの pkl ファイル
    - 学習用ファイル bert.ipynb
- bert.ipynbの補足
    - ファイル内に書いてありますが、 **GPU** を利用するように指定してください。そうしないと恐らく数時間かかります（未確認）。
    - `bert.ipynb` では、事前学習済みBERTの一つである[cl-tohoku/bert-base-japanese-v3](https://huggingface.co/tohoku-nlp/bert-base-japanese-v3)をベースとし、分類タスク用の線形層を追加した上で分類学習を行っている。
