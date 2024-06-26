# Transformer (Hugging Face) 入門

## Huggin Faceとは
Transformerベースの機械学習モデルの開発や共有をするためのプラットフォームの一つ。関連ライブラリやデータセット共有も行われ、研究者・開発者間で広まった。最近ではGoogle Colabのようなクラウド環境も提供している。実際に動かしながら確認するチュートリアルやドキュメントも充実しているので、興味がある場所から覗くことをお勧めします。

- 関連ページ
    - 公式サイト：https://huggingface.co/
    - ドキュメント例
        - [Transformersチュートリアル](https://huggingface.co/docs/transformers/v4.41.3/ja/index)
        - [NLPコース](https://huggingface.co/learn/nlp-course/ja/chapter0/1)
            - 左上の「NLP Course」をクリックすることで他コースも選択可能。
            - 公式トップページの「Learn」からも様々なコースあり。
        - [Datasetsチュートリアル](https://huggingface.co/docs/datasets/tutorial)

---
## 大まかな全体像
以下のような様々な機能を共通して使える点がとても便利。
- [Models](https://huggingface.co/docs/hub/models)
    - 事前学習済みモデル（トークナイザ含む）をリポジトリ管理している部分。基本的にはModel Cardsに説明を書いており、共通方法でトークナイザやモデルをダウンロードすることが可能。一部のモデルはアクセス制限あり。
- [Datasets](https://huggingface.co/docs/hub/datasets)
    - データが登録されているのは当然として、統一された方法でデータ参照することが可能。
- [Accelerate](https://huggingface.co/docs/accelerate/index)
    - CPU, GPU, TPU等のハードウェア部分を抽象化しており、切り替えが簡単に（なっているはず）。
- [Evaluate](https://huggingface.co/docs/evaluate/index)
    - タスクごとに異なる様々な評価指標（metrics）を抽象化して使いやすくしたもの。

---
## 使用例
文章の一部を隠したテキストを用意し、その隠した部分を推定するタスク（**穴埋めタスク**; fill mask）を実行する例を示す。

穴埋めタスクでは、``琉球大学は[MASK]にあります`` のように文中の一部を ``[MASK]`` に置き換えたデータを入力として用意し、この<u>マスク部分を置き換えるもっともらしい単語を推定するタスク</u>として実行することになることが多い。実際の出力は「全トークンidに対する対数確率」となる。

BERT系モデルでは [MASK] が一つの特殊トークン（専用トークン）として登録されており、このトークンが見つかった場合にはそこを推定するモデルとして機能する。言い換えると、それ以外のモデルで [MASK] と書いた場合に推定できるかどうかはモデルの実装次第である。

```{tip}
どのような特殊トークンが設定されているか、それがどのような役割を持っているのかはモデル依存である。2022年〜2023年頃までは特殊トークンを用いた設計も少なくなかったが、最近は「全て通常の自然文で指示する」ことを想定したモデルになっていることが多い。
```

### step 0: 環境構築。
Google Colaboratory を利用する際にはこのステップはほぼ不要。それ以外の環境で使う際には、必要に応じてGPU周りの開発環境を構築するために CUDA Toolkit をインストールし、そのOSやバージョンに合致するようにTransformers関連ライブラリをインストールする必要がある。様々な要因が関連するため結構大変。

### step 1: モデルを選ぶ。
https://huggingface.co/models にモデル一覧が登録されている。2024年6月20日時点で725,441件のモデルが登録されているようだ。この中から希望するモデルを探すには、左側の検索窓からキーワードを入力するか、もしくは絞り込み用のタグを選ぶこともできる。

例として、「日本語でチャット応答できるモデル」を探したい場合には、、、
- Natural Language Processingから「Text Genaration」を選択。
    - 「Text2Text Genration」を選ぶのも良い。どのようなタグが設定されているかはモデル開発者に依存している。Text2Textは「モデルの入力がテキスト、出力もテキスト」の意味。
    - Text Generationを選んだ場合、約11万件に絞り込まれる。
- 右上の検索窓に「japanese」を入力。
    - ja, japan, jp といった日本語を意味しそうな他キーワードも試してみるのも良い。
    - japaneseで絞り込むと、この時点で38件に絞り込まれる。この中からソート（ダウンロード数、Like数等）して選ぶと良いだろう。
    - 本来ならモデルサイズも考慮したいが必ずしも明示されていないことも多い。この場合には個別にモデル解説ページで確認することになる。
    - 例えば、 [elyza/ELYZA-japanese-Llama-2-7b](https://huggingface.co/elyza/ELYZA-japanese-Llama-2-7b) は、モデル名を見るだけでパラメータ数が 7B（70億個）だと分かる。
    - これに対して [doc2query/msmacro-japanese-mt5-base-v1](https://huggingface.co/doc2query/msmarco-japanese-mt5-base-v1) は、ページ下部にて [google/mt5-base](https://huggingface.co/google/mt5-base) をファインチューニングしたということが述べられている。mt5-baseのページを参照すると[論文へのリンク](https://arxiv.org/pdf/2010.11934)があり、そこによると 580M（5.8億パラメータ）であることが分かる。

上記は探し方の一例である。今回は「穴埋めタスク」用のモデルを使いたい。穴埋めタスクはBERT系モデルであることが多い。また日本語で穴埋めしたい。そこで[書籍: 大規模言語モデル入門](https://gihyo.jp/book/2023/978-4-297-13633-8)でも採用しているBERT系モデルの一つ、[tohoku-nlp/bert-base-japanese-v3](https://huggingface.co/tohoku-nlp/bert-base-japanese-v3) を使うことにしよう。

### step 2: トークナイザとモデルを用意する。
選んだモデル名、今回は「tohoku-nlp/bert-base-japanese-v3」を指定してトークナイザとモデルを用意しよう。pipelineを利用する方法と、そうではない方法とでコードが大きく異なる。

[pipeline](https://huggingface.co/docs/transformers/v4.41.3/ja/main_classes/pipelines)はモデルとタスクを適切に用意できる場合にはとても楽に利用することができる。モデルや入出力をカスタマイズせず、シンプルに使う場合にはこちらの方が良いだろう。しかしながら「どのモデルがどのタスクとして用意されているのか」は明示的ではなく、適切な組み合わせを探し出すことが難しいことも多い。このためこちらでは2種類の方法を示す。

なお、どちらの場合であっても「初めて利用する際にはモデルのダウンロード」が必要である。そして Google Colab で実行する際には「毎回環境がリセットされる」ため、何も考えずに実行すると毎回ダウンロードすることになる。今回はこちらの方法（毎回ダウンロード）を示す。大規模なモデルを利用する場合には、このダウンロードだけで数時間かかることもあるため工夫が必要になることがある。ここでいう工夫とは以下のような対応を指す。

- 大規模モデルを利用する際の工夫例
    - 初めて実行する場合
        - モデルをダウンロードする。
        - 自身のGoogleドライブにアクセス許可を設定する。
        - 自身のドライブ内に、モデルをファイルとして保存する。
    - 2回目以降の実行
        - 自身のGoogleドライブにアクセス許可を設定する。
        - 自身のドライブ内にあるファイルへのパスを指定して、モデルを読み込む。
    - 参考: [Google Colabで学習済みモデル保存する](https://qiita.com/yyyyy__/items/7a75c8e5302e36e6b5e4)
        - 学習済みモデルと書いていますが、学習していない状態のモデルでも同様の手順で保存できます。

#### step 2-1: pipelineを利用する方法
基本的にはpipelineの一覧からタスクに応じた利用方法を探すことになる。今回のタスクならば「Natural Language Processing」にある「FillMaskPipeline」が該当する。そのはずだが、今回のモデルでは失敗してしまう（ありがち）。。仕方がないので素のpipelineを利用することにした。

```Python
from transformers import pipeline
import pandas as pd

# モデルの用意
model_name = "tohoku-nlp/bert-base-japanese-v3"
fill_mask = pipeline(task="fill-mask", model=model_name)

# GPU利用したい場合
# fill_mask = pipeline(task="fill-mask", model=model_name, device="cuda:0")
```

#### step 2-2: pipelineを使わない方法
トークナイザとモデルを用意する必要がある。

```Python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd

# トークナイザとモデルの用意
model_name = "tohoku-nlp/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
```

### step 3: 用意したトークナイザとモデルを使う。
#### step 3-1: pipelineを利用する方法
ほぼ何も考えずに丸投げすることができます。

```Python
from transformers import pipeline
import pandas as pd

# モデルの用意
model_name = "tohoku-nlp/bert-base-japanese-v3"
fill_mask = pipeline(task="fill-mask", model=model_name)
# GPU利用したい場合
# fill_mask = pipeline(task="fill-mask", model=model_name, device="cuda:0")

# 問題文の用意
masked_text = "琉球大学は[MASK]にあります"
outputs = fill_mask(masked_text)

# 上位k件
top_k = 3
df = pd.DataFrame(outputs[:top_k])
df
```

#### step 3-2: pipelineを使わない方法
まずトークナイザでトークンID系列に変換する。それをモデルに入力を与えると logits を取得することができる。logsitsには「各トークン位置に対する全ての単語の確率スコア（対数確率）」が保存されている。ここからMASK位置のスコアを取得し、そのスコア一覧における上位k件のトークンを抽出する。

ということをつらつらと書く必要があるため、pipelineと比べると手間がかかる。ただし個別にカスタマイズしやすい（pipelineではほぼ内部で完結してしまうためカスタマイズしづらい）ため、目的に応じて使い分けると良いだろう。

```Python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd

# トークナイザとモデルの用意
model_name = "tohoku-nlp/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# 問題文の用意
masked_text = "琉球大学は[MASK]にあります"

# テキストをトークン化し、テンソルに変換
inputs = tokenizer(masked_text, return_tensors="pt")

# モデルによる推定
with torch.no_grad(): # 勾配を求めないモード（推論専用モード）
    outputs = model(**inputs)

# 各トークン位置に対するすべての単語の確率スコア（対数確率）を取得
# [batch_size, sequence_length, vocab_size]形式。
# 例えば logits[0, 0] の中には「バッチ0番目、トークン0番目に対する全ての語彙に対する対数確率」が保存されている。
logits = outputs.logits

# マスク位置を取得
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

# 上位3件の予測を取得
top_k = 3
top_k_token_ids = logits[0, mask_token_index].topk(top_k, dim=-1).indices[0].tolist()

# トークンをデコードして、人間が読める形式に戻す
predicted_tokens = [tokenizer.decode(token_id) for token_id in top_k_token_ids]

# 結果を表示
print("予測された単語:")
for i, token in enumerate(predicted_tokens, 1):
    print(f"{i}: {token}")
```

---
## piplineの他例
参考: [tranformers.pipelineを用いた推定例](./sample-pipeline.ipynb)
