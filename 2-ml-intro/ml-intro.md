# 機械学習外観

## 達成目標
以下の用語について説明できるようになろう。
- 仮説、モデル、入出力、特徴ベクトル、特徴量、データセット
- 過学習とは何か、この状況に陥ったモデルが何故悪いのかを説明できる。
- コード例を元に、教師あり学習のイメージを掴む。

---

## 機械学習とは
> "a computer can be programmed so that it will learn to play a better game of checkers than can be played by the person who wrote the program."
>
> by Arthur Samuel (1959), ["Some Studies in Machine Learning Using the Game of Checkers"](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.368.2254&rep=rep1&type=pdf)

人手で書いたプログラムよりも、うまくプレイできるように学ぶことができるようにプログラムされたコンピュータのこと。

> "we say that a machine learns with respect to a particular task T, performance metric P, and type of experience E, if the system reliably improves its performance P at task T, following experience E."
>
> by Tom Mitchell, ["The Discipline of Machine Learning"](http://www.cs.cmu.edu/~tom/pubs/MachineLearning.pdf)

ある特定のタスクT、パフォーマンス評価指標となるP、得られた経験Eに基づいて学習することで、そのパフォーマンスを改善しうるシステムのことを **{index}`機械学習<きかいがくしゅう-機械学習>`** と呼ぶ。

これらの定義は抽象的で分かりづらいが、そもそも機械学習とは何かしらの入出力関係で表現される事象をうまく表現するための手法全般を指すことが多い。入出力関係で表現される事象とは例えば、過去1年間の最大気温推移（これが入力）から今年の推移（これが出力）を推測したいというように、入力と出力とがペアとなるように用意されたデータセットと看做すことができる。このように用意されたデータセットにおいて入出力関係の傾向を見出し、用意されなかった入力から出力を推測しようとするのが機械学習である。

```{tip}
良くも悪くも、教師あり機械学習は入出力関係の傾向をモデル化しようとするだけである。機械学習自体は本来の事象を観察することなく、与えられたデータセットだけから傾向を捉えようとする。この行為を例えると、一度きりの又聞き(=用意したデータセット)だけで事象を理解しようとしている状況に近い。

例として、とある政党が新しい政策を提唱している状況を想像してみよう。この政策を理解するためには背景・経緯・方針・方策等を含む提案書を読み解く必要があるだろう。このとき説明が不十分であるならばその点について尋ねたり、欠落している観点を指摘したり、発案政党以外の第三者と議論を行う等のフィードバックを得ながら十分な理解に繋げていくことだろう。つまり政策という事象を理解するためには、事象に対する直接的なフィードバックが必要だ。

これに対し機械学習というシステムは、実証研究を除き、原則として事象に対するフィードバックは行うことができない。問題解決したい人が、その人のポリシーに則って用意したデータセットのみを観測することができる。このデータセットは何かしら偏っている可能性が高く、観測誤りも含まれるだろう。言い換えると、データセットは事象そのものを表してはおらず、事象を観測したユーザの観測ポリシーや観測誤差等の「事象とは異なる要因」を含んだものになっている。加えてそのユーザとやりとりすることもできない。このように「一度きりの又聞きだけで理解しよう」とするため、本来の事象とは異なる傾向を得てしまうことがしばしばある。この問題は **{index}`過学習<かがくしゅう-過学習>` ({index}`overfitting`)** と呼ばれ、一般人ならば起こし得ない誤りを出力してしまうことがしばしばある。

代表的な誤りの例として **{index}`敵対的サンプル<てきたいてきさんぷる-敵対的サンプル>` ({index}`adversarial attack`)** がある。
- 例: [第2回 ～　AIを騙す攻撃 – 敵対的サンプル -　～](https://jpsec.ai/attacks-that-deceive-ai/)

このように機械学習が獲得するモデルは誤りを含むため、業務システムとして運用するにはこのことを前提とした運用プロセスを検討することが重要である。このことは [MLOps](https://ja.wikipedia.org/wiki/MLOps) と呼ばれ、盛んに活用事例をベースに議論が行われている最中である。
```

---

### 演習
機械学習を適用したい事象を例示してみよう。その事象における入力や出力はなんだろうか。

---

## 問題設定
> "In general, a learning problem considers a set of n samples of data and then tries to predict properties of unknown data. If each sample is more than a single number and, for instance, a multi-dimensional entry (aka multivariate data), is it said to have several **attributes** or **features**."
>
> by [Machine Leaning: the problem setting](http://scikit-learn.org/stable/tutorial/basic/tutorial.html)

n個のサンプルからなるデータセットを用意し、未知データの特性を予測しようと試みる。もし、各サンプルが2個以上の数値で構成されているなら、そのサンプルを多次元データ（＝ベクトル）と呼ぶ。この場合、各ベクトルを構成する個々の要素を **{index}`属性<ぞくせい-属性>`**、もしくは **{index}`特徴<とくちょう-特徴>`** と呼ぶ。

一般には、そもそも単一特徴であることはほぼ無い。ただし、アルゴリズムの理解を促すため等、特定条件下においては単一特徴からなるサンプル、すなわちスカラーを用いることがある。

---

### 機械学習の種類、体系化の例
- [Business Intelligence and its relationship with the Big Data, Data Analytics and Data Science](https://www.linkedin.com/pulse/business-intelligence-its-relationship-big-data-geekstyle/)
- [Choosing the right estimator](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)

- **{index}`教師あり学習<きょうしありがくしゅう-教師あり学習>`（{index}`supervised learning`）**
  - **{index}`分類<ぶんるい-分類>`（{index}`classification`）**
    - In Classification, the samples belong to two or more classes and we want to learn from already labeled data how to predict the class of unlabeled data.
    - モデルの出力がカテゴリやラベル（数値ではない）
  - **{index}`回帰<かいき-回帰>`（{index}`regression`）**
    - If the desired output consists of one or more continuous variables, then the task is called regression.
    - モデルの出力が連続値
- **{index}`教師なし学習<きょうしなしがくしゅう-教師なし学習>`（{index}`unsupervised learning`）**
  - **{index}`クラスタリング<くらすたりんぐ-クラスタリング>`（{index}`clustering`）**
    - Clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense or another) to each other than to those in other groups (clusters).
    - 類似したものをグループにまとめる
  - **{index}`次元削減<じげんさくげん-次元削減>`（{index}`dimensionality reduction`）**
    - PCA is used to decompose a multivariate dataset in a set of successive orthogonal components that explain a maximum amount of the variance.
    - 分散の最大量を説明しやすい、直行成分上の空間に圧縮するために使われる。
- **{index}`強化学習<きょうかがくしゅう-強化学習>`（{index}`reinforcement learning`）**
  - Reinforcement learning (RL) is an area of machine learning concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward.
  - 累積報酬最大化のために取るべき行動を獲得するための手法。

````{admonition} Check your understading
あなたは[マンゴー栽培](https://ja.wikipedia.org/wiki/マンゴー#日本)を行う農家だとする。マンゴーの収穫は年に一度であり、水に弱く、品質維持が困難であり、完熟後は冷蔵でも1週間程度しか保存が効かない。栽培品目としては難易度が高く、栽培や収穫、流通、小売等多岐にわたり注意を要する。このため如何にして収穫量を増やすか、品質を維持できるか等様々な懸念事項がある。

今回はマンゴーの品質としてその果肉に含まれる糖度に着目し、一定程度の糖度を含む果実となるよう栽培方法や環境要因について模索することを目指したい。つまり高い糖度を持つマンゴーを収穫するための栽培方法や環境要因を特定し、マニュアル化したい。そのための検討材料としてサンプル毎に糖度だけではなく、開花からの受粉時期、日毎の最低気温と最高気温、土壌の水分量や肥料の種別・量、、等について測定し、データセットを構築した。このとき、糖度以外の特徴を用いて糖度を見積もるタスクは以下のうちどれに相当するか。

```{dropdown} 回帰
**正解！**：タスク対象である糖度は数値（連続値）であることから、回帰問題である。
```
```{dropdown} 分類
**残念！**：出荷可能か否かという2値分類として考えるならば分類タスクとして検討できる余地はある。ただし今回のタスク対象である糖度は数値（連続値）であることから、回帰問題である。
```
```{dropdown} クラスタリング
**残念！**：クラスタリングとは「サンプルをいくつかのクラスタに分ける」タスクであり、今回のタスクとは異なる。クラスタリングの適用方法としては、多数の異なる農家におけるデータを多数集めた上で似通った栽培方法をグルーピングし、新たなブランドの出発点としたり、現栽培方法を大きく変更すること無くスムーズに移行しやすくするための10年間計画を立案したりといった、「類似性」に根ざした意思決定を行えるかもしれない。
```
```{dropdown} 強化学習
**残念！**：強化学習は「Aを実行した結果、Bになった」というような何かしらのシミュレータを用意することが大前提と考えて良い。例えば、道路と車・バイク・歩行者・信号機を用意した仮想都市を構築し、そこでどのように信号機を制御すると効率良く移動体を流すことができるか、といった状況だ。今回の農家の例でいうと、自動水やりシステムにおける水やりタイミング最適化（任意の水やり結果をシミュレーションできるとする）、収穫時の収穫時期や量に応じた搬送計画最適化といった目的には適用できるかもしれない。
```
````

<hr>

### 演習：教師あり学習における処理フローの確認
回帰タスクのコード例を動かして、教師あり学習における処理フローを眺めてみよう。
- [regression_diabetes.ipynb](./regression_diabetes.ipynb)
  - requirements
    - Python: >=3.7.1
    - scikit-learn: >=0.20.1
    - numpy: >=1.15.4
    - pandas: >= 1.3.5
    - matplotlib: >=3.0.2
    - jupyter (or Google Colab)
  - [Diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
    - 442サンプル
      - 10個の標準化した特徴量
        - 年齢、性別、ボディマス指数、平均血圧、6つの血清測定結果
      - ベースラインからの、1年後における糖尿病進行度合い
    - 本データセットの目的
      - 10個の測定結果から、1年後の進行度合いに関する実データを442サンプル得た。これをベースに、未知のサンプルに対して1年後の進行度合いを予測したい。

````{admonition} Check your understading
下記項目はどのようにして確認するといいだろうか？

1. Diabetesデータセットにおける各サンプルは、何次元のベクトルだろうか？
2. 各ベクトルは何個の特徴量で構成されているだろうか？
3. サンプルと教師データはどのように用意されているだろうか？
4. モデルはどう用意しているだろうか？
5. 用意したデータセットとモデルを使って、どのように学習しているだろうか？
6. 学習済みモデルの評価をどのようにしているだろうか？
7. 学習済みモデルは、学習した結果何を得ているのだろうか？
8. 新しい患者のデータを入手したとする。その患者の標準化されたBMI値は 0.0 であった。この患者に対する糖尿病進行度合いの予測値はどのぐらいだろうか？
````

---

## 教師あり学習における処理フローの例、公開データの例
代表的には以下の作業項目に分けて取り組むことが多い。なお、各作業がスムーズ終わり次に進めるとは限らず、一つ以上手前に戻って取り組み直すこともよくある。

- 処理フローの例
  - タスク種別を設定し、評価方法を検討する。
  - データを用意する。
    - 機械学習について学ぶことが主目的なら、[公開データの例](https://docs.google.com/document/d/e/2PACX-1vRgMcscQFuB-rRgmpoqc4oZAa3rZdzoy0cNcOfm58AUJ1kG9fkhl9egCfPYvjCcR3voF8pzvvH4eXH_/pub)や、[Kaggle](https://www.kaggle.com/datasets)等コンペティションで提供されているデータセットを選択するのが楽。
  - モデルを用意する。
  - 学習する。
  - 学習済みモデルを評価する。
  - 必要に応じて、データ・モデル・ハイパーパラメータ等をチューニングしながら改善を目指す。
  - 目標性能に達したのであれば、（これまで人手でやっていたことを）学習済みモデルに置き換える。

<hr>

(report1)=
## 課題レポート1：機械学習してみよう
[Secondary Mushroom](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset)からデータをダウンロードし、以下の課題に取り組め。
### Level 0: 準備。（報告不要）
- (a) ダウンロード。
  - ``risk+factor+prediction+of+chronic+kidney+disease.zip`` というファイルがダウンロードされるはずだ。
- (b) zipファイルの展開。
  - 展開すると ``MushroomDataset`` フォルダが作成される。フォルダ内に以下の4件のテキストファイルが入っていることを確認しよう。
    - ``primary_data_meta.txt``: primary_data.csvに関する説明書。
    - ``primary_data.csv``: `;`を区切り文字としたCSVファイル。
    - ``secondary_data_meta.txt``: secondary_data.csvに関する説明書。
    - ``secondary_data.csv``: `;`を区切り文字としたCSVファイル。<u>今回利用するのはこちら。</u>

### Level 1. データセット調査。（15分想定）
Secondary Mushroomがどのようなデータセットなのか概説せよ。少なくとも以下の内容を含めること。
- タスク種別は何か。
- 何を目的として収集または構築したのか。

```{hint}
おおよその説明はWebページと secondary_data_meta.txt に書かれています。
興味のある人は論文（Introductory Paper）も覗いてみるとよいでしょう。
```

### Level 2. 特徴量調査。（1時間想定）
データセットを構成する各要素（下記）について、各々1行程度で簡潔に解説せよ。
- サンプル数
- 分類クラス数
- クラスの説明
- 特徴ベクトルの次元数
- 各特徴の説明とデータ形式
  - noimnalな特徴を2件、metricalな特徴を2件、それぞれ選択し、分かる範囲で説明すること。

```{hint}
- secondary_data_meta.txtの5番目に「Class infomation」があり、これがクラスに関する説明をしています。
- 同じファイルの6番目に「Variable Information」があり、これが説明変数すなわち特徴について説明をしています。
```

### Level 3. 分類学習に用いるモデルを選べ。（15分想定）
レポートには、(a) 選んだモデルの名前と、(b) 2個程度のハイパーパラメータについて簡潔に解説せよ。パラメータ解説は直訳程度で構わない。

```{hint}
モデルは分類タスクに適用できるものから自由に選んで構わない。[Flow Chart](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)を参考にするのも良い。
- ここではscikit-learnを想定して記述しているが、Keras等、別の機械学習ライブラリを用いても良い。
- モデルが持つパラメータは大別して次の2種類に分けられます。
  - **{index}`ハイパーパラメータ`**: 開発者自身が手動で設定する必要があるパラメータ。例えばクラスタリングの多くのアルゴリズムでは、クラスタ数を手動で設定する必要があります。
  - 上記以外のパラメータ（重みとも呼ばれる）: 学習データから自動で獲得するパラメータ。例えば線形回帰モデル $y = ax + b$ におけるa,bは、学習データから自動で求めます。
- モデルを選んだら、一度ドキュメントを参照し、簡単な使い方やハイパーパラメータについて確認しよう。例えば[sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)なら、**Parameters**欄に並んでいる引数は手動調整可能である。**Examples**欄には使い方の例が示されている。**Methods**欄には、このオブジェクトが持っている関数が示されている。
```

### Level 4. 実際にコードを書いて分類学習せよ。（1〜3時間想定）
今回の意図は、全体の流れを理解することである。実験結果が悪くても構わないので、流れを理解しながら取り組もう。レポートには (a) 主要コード上限50行を示し、解説せよ。また (b) 分類結果についても示し、解説せよ。
- 条件
  - Level 2で調査したとおり、説明変数には数値データ（metrical）とそうではないデータ（nominal）が混在している。今回は metrical だけを説明変数として利用すること。
  - 特徴量は何も処理を加えずそのまま用いること。
  - 評価は5分割検定の平均値とすること。

````{hint}
- Google Colab で外部ファイルを読み込む方法
  - (1) ファイルをアップロードする。
    - ノートブックを開き、左の「ファイル」アイコンをクリック。
    - 「セッションストレージにアップロード」をクリックし、ファイルを選択。
  - (2) Pythonでファイルを読み込む。
    - 多くの場合は csv ファイル, tsv ファイル, Excelファイルだろう。ファイルに応じてコードが変わる。
    - また、ファイル内にヘッダ（列名）があるかどうかでも変わってくる。
```Python
# case 1: csvファイル
import pandas as pd
filename = 'test.csv' # アップロードしたファイル名を指定
df = pd.read_csv(filename, header=0) # ヘッダがある場合
df = pd.read_csv(filename, header=None) # ヘッダがない場合

# case 2: tsvファイル
import pandas as pd
filename = 'test.tsv' # アップロードしたファイル名を指定
df = pd.read_csv(filename, sep='\t', header=0) # ヘッダがある場合
df = pd.read_csv(filename, sep='\t', header=None) # ヘッダがない場合

# case 3: Excelファイル（拡張子が ``.xlsx``）
import pandas as pd
filename = 'test.xlsx' # アップロードしたファイル名を指定
df = pd.read_excel(filename, header=0) # ヘッダがある場合
df = pd.read_excel(filename, header=None) # ヘッダがない場合
```

- 分類タスクのコード例
  - [scikit-learn公式例](https://scikit-learn.org/stable/auto_examples/index.html#classification)
  - 授業資料例
    - [ダウンロードしたデータセットに機械学習を適用する流れ](./intro_ml_flow.ipynb)
````

### Options：余裕があれば取り組んでみよう
- 例1: 混同行列（[confusion matrix](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py)）により、精度の良し悪しに偏りのあるクラスがあるかどうかを確認してみよう。
- 例2: 失敗事例について要因分析してみよう。
- 例3: 選択したモデルにハイパーパラメータ（手動調整するパラメータ）があるならば、それをチューニングして精度改善を試みてみよう。

```{note}
データセット次第では、それをどのように読み込めばよいのか、読み込んだデータをどのように特徴ベクトルとして利用するのかを検討すること自体が難しいこともあり得る。相談場をTeamsに用意するので積極的に活用しよう。
```

---

## 予習代わり：課題取り組みを通した疑問等
前述の課題レポート1に取り組み、気になる事柄があれば次回授業の前日までに、別途用意するフォームに入力すること。
