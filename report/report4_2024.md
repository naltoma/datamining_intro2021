# 課題4：クチコミデータへ自然言語処理を適用してみよう（深層学習版）。
課題3で取り上げた[Japanese Realistic Textual Entailment Corpus](https://github.com/megagonlabs/jrte-corpus)を題材に、深層学習で取り組んでみよう。

---
## Level 1: ファインチューニング済みモデルを用いた識別。（1.5時間想定）
- 前提
    - 課題3の続きである。pn.tsvをダウンロードして準備しておくこと。
- やること
    - コード例を参考に、識別学習を実行する。
        - カスタマイズされた事前学習済みモデルとして [mr4/bert-base-jp-sentiment-analysis](https://huggingface.co/mr4/bert-base-jp-sentiment-analysis) を使うことを想定しているが、これ以外のモデルでも構わない。
- 報告事項
    - (1) 利用したモデル名。
    - (2) テスト用データに対する分類精度（[classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)）。
    - (3) テスト用データに対する混合行列（[ConfusionMatrixDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html)）。
    - (4) classification_reportに対する考察。長くて数行程度。
    - (5) ConfusionMatrixDisplayに対する考察。長くて数行程度。
- ヒント
    - pipelineを用いると良い。
        - 参考: [pipelineを用いた推定例](../4-nlp/sample-pipeline.ipynb)
    - ラベル名不一致の解消。
        - JRTEデータセットにおけるラベル（df["sentiment"]）は 1, 0, -1 の3種類である。これに対し mr4/bert-base-jp-sentiment-analysis では positive, negative の2ラベルを出力する。この不揃いを解消する必要がある。
        - JRTEラベル名に合わせても良いし、モデル出力に合わせても良い。課題説説明文では "positve", "normal", "negative" に合わせるものとする。
    - classification_report, ConfusionMatrixDisplayではラベル名も出力すること。
        - 参考: [classification_evaluations.ipynb](../2-ml-intro/classification_evaluations.ipynb)
    - 考察については、結果から論理的に説明できる事実もしくは可能性について言及すること。また言及内容が事実と可能性どちらなのかを読み取れるように述べること。（可能性に過ぎないことを断定したり、事実であることを曖昧を持たせた形で書いた場合には減点します）

---
## Level 2: 事前学習済みモデルをファインチューニングする。（1〜1.5時間想定）
- 背景
    - 別データでファインチューニングされたモデルではあまり精度が良くないようだ。そこで独自にファインチューニングすることにした。
    - なお、Level 1で用いた "mr4/bert-base-jp-sentiment-analysis" を AutoModelForSequenceClassification で用意して学習することはできない。これは、AutoModelForSequenceClassificationは「事前学習済みモデルを用意し、最終層にLinear層を新規追加した上で分類学習する」ことを想定しているためだ。これに対し "mr4/bert-base-jp-sentiment-analysis" はすでにそのような層が追加された状態になっており、文脈ベクトルが見えない（2クラスに対するスコアを出力する）モデルになっている。このためモデルを構築することができず、エラーとなる。これを避けるため事前学習だけをしたモデルを別途用意することにした。
- やること
    - コード例を参考に、LLMをファインチューニングする。
        - 事前学習済みモデルとして [tohoku-nlp/bert-base-japanese-v3](https://huggingface.co/tohoku-nlp/bert-base-japanese-v3) を使うことを想定しているが、これ以外のモデルでも構わない。
        - ヒントにあるように上記モデルを用いた場合は結果を含めてノートブックに記載している。このため必ずしも実行は不要だが、途中過程を含めて理解することをお勧めする。
- 報告事項
    - (1) 利用したモデル名。
    - (2) テスト用データに対する分類精度（[classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)）。
    - (3) テスト用データに対する混合行列（[ConfusionMatrixDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html)）。
    - (4) classification_reportに対する考察。Level 1 との比較を含めること。長くて10行程度。
    - (5) ConfusionMatrixDisplayに対する考察。Level 1 との比較を含めること。長くて10行程度。
- ヒント
    - BERTを用いたファインチューニング
        - 参考: [jrte_pn_bert_train.ipynb](../4-nlp/jrte_pn_bert_train.ipynb)
        - 自身で実行し直す場合、T4 GPUを用いると環境構築含めて20分程度要する。（CPUだと数時間かかると思われる）

---
## Level 3: 失敗要因分析（1時間想定）
- やること
    - Level 2のコード例（BERTを用いたファインチューニング）における「具体的な失敗例」には、学習データとテストデータに対する失敗サンプルを「全て」示している。[課題3, Lv3の失敗要因分析](./report3_2024.md)を参考に要因分析せよ。
- 報告事項
    - (1) 分析方法。
    - (2) 分析結果。

---
## Level 4: 過剰適合に対する対策検討。（1時間想定）
- やること
    - 調査と検討
        - Level 2のコード例（BERTを用いたファインチューニング）における学習中の損失推移を眺めると、学習データに対する損失は、頭打ちになっているようだが基本的には減る方向にうまく学習できているように見える。しかし検証データに対する損失は、エポック2までしか減っておらずそれ以降は増加し続けている。これは[過学習もしくは過剰適合（over-fitting）](https://en.wikipedia.org/wiki/Overfitting#Machine_learning)している可能性が高そうだ。
        - 仮に過学習に陥っているとした場合、どのような対策をすると良いだろうか。ニューラルネットワーク系モデル（深層学習も含む）における過学習や過剰適合について調査し、出典付きで対策案を説明せよ。
- 報告事項
    - (1) 対策案。100字以上で説明すること（手法名を書くだけでは不十分）。
    - (2) 出典。

---
## オプション例
- 過学習への対策案を実施してみる。
- 失敗要因分析に基づき改善案を考え実施してみる。
