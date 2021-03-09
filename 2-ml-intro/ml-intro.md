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

ある特定のタスクT、パフォーマンス評価指標となるP、得られた経験Eに基づいて学習することで、そのパフォーマンスを改善しうるシステムのことを機械学習と呼ぶ。

これらの定義は抽象的で分かりづらいが、そもそも機械学習とは何かしらの入出力関係で表現される事象をうまく表現するための手法全般を指すことが多い。入出力関係で表現される事象とは例えば、過去1年間の最大気温推移（これが入力）から今年の推移（これが出力）を推測したいというように、入力と出力とがペアとなるように用意されたデータセットと看做すことができる。このように用意されたデータセットにおいて入出力関係の傾向を見出し、用意されなかった入力から出力を推測しようとするのが機械学習である。

---

## 問題設定
> "In general, a learning problem considers a set of n samples of data and then tries to predict properties of unknown data. If each sample is more than a single number and, for instance, a multi-dimensional entry (aka multivariate data), is it said to have several **attributes** or **features**."
>
> by [Machine Leaning: the problem setting](http://scikit-learn.org/stable/tutorial/basic/tutorial.html)

n個のサンプルからなるデータセットを用意し、未知データの特性を予測しようと試みる。もし、各サンプルが2個以上の数値で構成されているなら、そのサンプルを多次元データ（＝ベクトル）と呼ぶ。この場合、各ベクトルを構成する個々の要素を**属性**、もしくは**特徴**と呼ぶ。

一般には、そもそも単一特徴であることはほぼ無い。ただし、アルゴリズムの理解を促すため等、特定条件下においては単一特徴からなるサンプル、すなわちスカラーを用いることがある。

---

### 機械学習の種類、体系化の例
- [Business Intelligence and its relationship with the Big Data, Data Analytics and Data Science](https://www.linkedin.com/pulse/business-intelligence-its-relationship-big-data-geekstyle/)
- [Choosing the right estimator](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)

- **教師あり学習（supervised learning）**
  - **分類（classification）**
    - In Classification, the samples belong to two or more classes and we want to learn from already labeled data how to predict the class of unlabeled data.
    - モデルの出力がカテゴリ値（数値ではない）
  - **回帰（regression）**
    - If the desired output consists of one or more continuous variables, then the task is called regression.
    - モデルの出力が連続値
- **教師なし学習（unsupervised learning）**
  - **クラスタリング（clustering）**
    - Clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense or another) to each other than to those in other groups (clusters).
    - 類似したものをグループにまとめる
  - **次元削減（dimensionality reduction）**
    - PCA is used to decompose a multivariate dataset in a set of successive orthogonal components that explain a maximum amount of the variance.
    - 分散の最大量を説明しやすい、直行成分上の空間に圧縮するために使われる。
- **強化学習（reinforcement learning）**
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

### 演習1：教師あり学習における処理フローの確認
回帰タスクのコード例を動かして、教師あり学習における処理フローを眺めてみよう。
- [regression_diabetes.ipynb](./regression_diabetes.ipynb)
  - requirements
    - Python: >=3.7.1
    - scikit-learn: >=0.20.1
    - numpy: >=1.15.4
    - matplotlib: >=3.0.2
    - jupyter (or Google Colab)
  - Diabetes dataset
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
````

---

## 教師あり学習における処理フローの例、公開データの例
代表的には以下の作業項目に分けて取り組むことが多い。なお、各作業がスムーズ終わり次に進めるとは限らず、一つ以上手前に戻って取り組み直すこともよくある。

- 処理フローの例
  - タスク種別を設定し、評価方法を検討する。
  - データを用意する。
    - 機械学習について学ぶことが主目的なら、[公開データの例](https://docs.google.com/document/d/e/2PACX-1vRgMcscQFuB-rRgmpoqc4oZAa3rZdzoy0cNcOfm58AUJ1kG9fkhl9egCfPYvjCcR3voF8pzvvH4eXH_/pub)や、Kaggle等コンペティションで提供されているデータセットを選択するのが楽。
  - モデルを用意する。
  - 学習する。
  - 学習済みモデルを評価する。
  - 必要に応じて、データ・モデル・ハイパーパラメータ等をチューニングしながら改善を目指す。
  - 目標性能に達したのであれば、（これまで人手でやっていたことを）学習済みモデルに置き換える。

<hr>

## **課題レポート1：機械学習してみよう**
[Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)等の一般公開されているデータセットから、分類タスクのデータセット（登録日2000年以降）を一つ選び、以下の課題に取り組め。
- Level 1. どのようなデータセットなのか、100〜200字程度で概説せよ。
- Level 2. データセットを構成する各要素（下記）について、各々1行程度で簡潔に解説せよ。
    - サンプル数
    - 特徴ベクトルの次元数
    - 各特徴の説明とデータ形式（冒頭3つの特徴まで）
    - 分類クラス数
    - クラスごとの説明
- Level 3. 分類学習に用いるモデルを選べ。
    - レポートには、選んだモデルと、2個程度のハイパーパラメータについて簡潔に解説せよ。パラメータ解説は直訳程度で構わない。
      - モデルは分類タスクに適用できるものから自由に選んで構わない。[Flow Chart](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)を参考にするのも良い。
        - ここではscikit-learnを想定して記述しているが、Keras等、別の機械学習ライブラリを用いても良い。
      - モデルを選んだら、一度ドキュメントを参照し、簡単な使い方やハイパーパラメータについて確認しよう。例えば[sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)なら、**Parameters**欄に並んでいる引数は手動調整可能である。**Examples**欄には使い方の例が示されている。**Methods**欄には、このオブジェクトが持っている関数が示されている。
- Level 4. 実際にコードを書いて分類学習せよ。
    - 今回の意図は、全体の流れを理解することである。実験結果が悪くても構わないので、流れを理解しながら取り組もう。レポートには主要コード上限50行を示し、解説せよ。また分類結果についても示し、解説せよ。
      - データセットをダウンロードし、前半50%を training set、後半50%を testing setとしよう。なお、サンプル数や次元数が大きすぎると思われる場合には、自由に削減して構わない。例えば学習に要する時間が1時間かかるようなら、サンプル数や次元数を半分〜10%程度に削減することで実験しやすくなるはずである。
      - [分類タスクのコード例](https://scikit-learn.org/stable/auto_examples/index.html#classification)を参考に、学習させてみよ。
      - 学習済みモデルを用いて、テストデータに対する評価を行え。ここで、評価は「サンプル毎に分類成功したか否か」に基づいた精度により行うものとする。例えば、100サンプル中1個成功したのなら、精度は1%である。
- Options：余裕があれば取り組んでみよう
    - 例1: 混同行列（[confusion matrix](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py)）により、精度の良し悪しに偏りのあるクラスがあるかどうかを確認してみよう。
    - 例2: 失敗事例について要因分析してみよう。
    - 例3: 選択したモデルにハイパーパラメータ（手動調整するパラメータ）があるならば、それをチューニングして精度改善を試みてみよう。

---

## 予習代わり：課題取り組みを通した疑問等
前述の課題レポート1に取り組み、気になる事柄があれば次回授業の前日までに、別途用意するフォームに入力すること。
