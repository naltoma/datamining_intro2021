# 特徴量エンジニアリング：テキストデータに対する特徴量設計2（word2vecによる分散表現）
- 参考文献
  - [word2vecによる自然言語処理](https://www.oreilly.co.jp/books/9784873116839/)
    - [Github: word2vec](https://github.com/tmikolov/word2vec)
    - [Word2Vec：発明した本人も驚く単語ベクトルの驚異的な力](https://deepage.net/bigdata/machine_learning/2016/09/02/word2vec_power_of_word_vector.html)
    - [絵で理解するWord2vecの仕組み](https://qiita.com/Hironsan/items/11b388575a058dc8a46a)
  - [ゼロから作るDeep Learning ❷ ――自然言語処理編](https://www.oreilly.co.jp/books/9784873118369/)
  - [gensim](https://radimrehurek.com/gensim/)
  - [fastText](https://fasttext.cc)
- ＜目次＞
  - <a href="#intro">word2vecとは（概要）</a>
  - <a href="#model">word2vecの簡易実装を通した推論学習</a>
  - <a href="#trained-model">実用的なライブラリや学習済みモデル</a>

<hr>

## <a name="intro">word2vecとは（概要）</a>
- 背景
  - Bag-of-Wordsのような、シンプルなカウントベースの特徴ベクトルでは単語の意味を含むことが困難。
  - 分布仮説に基づいた、シンプルな共起情報ベースの特徴ベクトルではある程度意味らしさを含むが、共起行列は極めて素なベクトルであり、無駄が大きい。次元削減するにしてもコストが大きい。
  - 問題意識
    - 従来以上に潜在的な意味を汲み取る方法は無いだろうか？　現実世界での意味的な距離をできるだけ保つような低次元ベクトルでの表現方法はないだろうか？
    - SVDやPCAは次元削減に寄与するが、SVDやPCA自体の計算コストが高いため、大規模なコーパスに対しては現実的な解法になっていない。より適切な語彙情報を含めるためにはより大規模なコーパスに対して適用したいが、現実的な推論方法は無いだろうか？
- word2vec
  - Tomas Mikolovらによって提案されたニューラルネットワーク（CBOW, Skip-gram）で、単語を分散表現に変換する。SVDやPCAと同じく次元圧縮に用いられることが多い。
  - 2種類の推論方法
    - CBOWモデルでは、着目単語に対する文脈（ウィンドウサイズで指定）が入力として与えられ、そこから着目単語を推定するように学習する。
    - Skip-gramモデルでは、着目単語が入力として与えられ、そこから文脈中の1単語を推定するように学習する。
      - 深層学習でいうところの **Autoencoder** と呼ばれるものに似た考え方。
        - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
        - Autoencoderでは、元々はノイズ除去が主目的。これが後々応用〜拡張され、高解像度化にも繋がっている。
    - 図：[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)の Figure 1 より。
    - ![表現の例](./figs/word2vec_figure1.png)
- 何がすごいのか？
  - ``Paris - France + Italy = Rome`` のように、単語ベクトルの引き算＋足し算で様々な「関係」を表現できた。シソーラス等の事前データ無しに、大量のテキストからこのような潜在的な語彙関係を内包する特徴空間に、単語ベクトルを写像できた。
  - 図：[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)の Table 8 より。
  - ![表現の例](./figs/word2vec_table8.png)

<hr>

## <a name="model">word2vecの簡易実装を通した推論学習</a>
- 3階層ニューラル・ネットワーク（入力層・中間層・出力層）。
  - [Understanding Neural Networks](https://towardsdatascience.com/understanding-neural-networks-19020b758230)
  - [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com)
  - コード例
    - [train_momotaro.ipynb](./deep-learning-from-scratch-2/ch03/train_momotaro.ipynb)
    - [train_momotaro.py](./deep-learning-from-scratch-2/ch03/train_momotaro.py)
- CBOWモデル、Skip-gramにおける推論とは
  - ![推論の定式化](./figs/cbow-skipgram-model.png)
- 補足（振り返り）
  - そもそもモデルや損失関数（コスト関数）とは何だったか？
    - [線形回帰の例](https://github.com/naltoma/datamining_intro/blob/master/2-ml-intro/ml-intro.md#model-and-learn)
    - 事象の入出力をパラメータ付き数式で表現したものがモデル。CBOWやSkip-gramでは、前述したモデルをNNで表現しているが、中身は同様にパラメータ付き数式。NNにおけるパラメータは、中間層への入出力重み。初期値は乱数で設定する。その時点では誤りが大きい、でたらめな推測しかできない。
    - 誤り度合いを設定しているのがコスト関数。コスト関数が小さくなるように、勾配法でパラメータを調整する。NNのパラメータ更新は「誤差逆伝播法」と呼ばれており、出力層における誤差（交差エントロピー誤差）を元に出力層側から遡る形で更新していく。
      - 詳細は[ゼロから作る Deep Learning ❷](https://www.oreilly.co.jp/books/9784873118369/)おすすめ。
- 実装上の工夫
  - CBOWでは、中間層の重みを文脈間で共有している。（共有する＝一種の制約条件を付与する）
    - 一般的なNN形式で「vocab_size=5, window_size=2, hidden_size=3」のNNを構築しようとすると、「10x3の中間層」を用意する必要がある。CBOWだと文脈間で共有するため、「5x3の中間層」で処理される。計算コストは変わらないが、vocab_sizeが数十万〜百万規模になるとメモリが膨大に要求されるため、実用面では共有した方が低コストに抑えられる。また、「大規模なパラメータを少ない回数だけ更新する」よりも「少量のパラメータを何度も更新する」方が結果的には学習効果が得られやすいため、結果的には共有した方がベターとなるケースが多い。
    - Skip-gramでも重みは共有している。
    - 共有にはデメリットもあり、単語の順番は無視されている。例えば「w_{t-2},w_{t-1},w_{t+1},w_{t+2}」という順番で入力した場合と、「w_{t+2},w_{t+1},w_{t-1},w_{t-2}」とでは全く同一として扱われる。
  - Embeddingレイヤの導入。
    - vocab_sizeが数十万〜百万規模になると、行列演算のコストが膨大。入力ベクトルが実数値ベクトルならば厳密に演算する必要があるが、word2vecではone-hotベクトルである。one-hotベクトルの場合、実際には「入力ベクトルの中身はほとんどが0」であり、演算結果は「特定の行を抜き出す」だけ。このためだけに厳密な演算をするのは計算リソースが勿体ないので、抽出する処理に置き換えよう。
  - Negative Samplingの導入。
    - 大規模なコーパスに対してはsoftmaxが重くなるため、これを別の計算で代替するアプローチ。softmaxでは多値分類を想定しているが、これを2値分類の組み合わせとして考え直す。加えて、2値分類の際に正事例に対して出力が1.0に近づくなるように学習するだけではなく、「ランダムな小数の負事例（= negative sampling）に対して出力が0.0に近づくように学習する」。
- カウントベース手法との違い
  - 推論ベースでは、コーパスの一部を何度も見ながら学習（ミニバッチ学習）することができる。これは、新しい用語を増やす等、コーパスを追加したいときに柔軟に対応しやすいことを意味する。例えば新しく10語を追加したい場合、入力層を10個増やした上で学習済みの重みをコピーする（新規部分は乱数のママ）ことで、学習済み重みを初期値として利用した上で学習することができる。
  - カウントベースでは、共起行列を作成し直し、そこからPCA/SVD等による次元削減が必要になる。
  - ちなみに、[Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://www.aclweb.org/anthology/Q15-1016/)によると、性能的には大きな差はないらしい。

<hr>

## <a name="trained-model">実用的なライブラリや学習済みモデル</a>
- [gensim](https://radimrehurek.com/gensim/)
  - word2vec, doc2vec, fastTextの代表的な実装。
- [fastText](https://fasttext.cc)
  - Facebookが開発した軽量な分散表現学習器。基本的には文書分類タスクを想定。157言語の学習済みモデルを提供。
  - ![YFCC100Mにおける比較結果](./figs/fasttext_table5.png)
  - [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)より引用。
  - 「Efficient learning for text classification」 by [facebook research](https://research.fb.com/fasttext/)
    - "To be efficient on datasets with very large number of categories, it uses a hierarchical classifier instead of a flat structure, in which the different categories are organized in a tree (think binary tree instead of list)."
    - "FastText also represents a text by a low dimensional vector, which is obtained by summing vectors corresponding to the words appearing in the text. In fastText, a low dimensional vector is associated to each word of the vocabulary. "
  - [How does fastText (Facebook) work? Is there any research paper or blog post explaining the theory behind its working?](https://www.quora.com/How-does-fastText-Facebook-work-Is-there-any-research-paper-or-blog-post-explaining-the-theory-behind-its-working)
    - "The gist of fastText is that instead of directly learning a vector representation for a word (as with word2vec), we learn a representation for each character n-gram. Each word is represented as a bag of character n-grams, so the overall word embedding is a sum of these character n-grams."
    - "As a simple example, if we set n=3, the vector for the word “where” would be represented by a sum of trigrams: <wh, whe, her, ere, re> (where <, > are boundary symbols denoting the beginning and end of a word)."
  - subwordという考え方
    - [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)
    - [Subword Neural Machine Translation](https://github.com/rsennrich/subword-nmt)
    - 主なメリット
      - 文字繋がりで単語を表現するため、次元数を減らすことが可能。
      - 未知語・新語に対しても、その構成要素である文字群からある程度の推定が可能。（かなり強い恩恵。BoWベースやword2vecでは未知語に対応できない）
    - subwordの弊害
      - [fastTextのsubword(部分語)の弊害](http://studylog.hateblo.jp/entry/2016/09/20/103724)
      - 文字繋がりの影響が強すぎる。
- [Github: BERT](https://github.com/google-research/bert)
  - [Google AI Blog: Open Sourcing BERT: State-of-the-Art Pre-training for Natural Language Processing](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)
  - [paper: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
    - ![Table 2](./figs/bert_table2.png)
    - 上記 Table 2より引用。
  - 汎用的なタスク（general-purpose "language understanding" model）を想定した分散表現学習器。学習済みモデルも公開。
  - 2つのステップ
    - 1. pre-training
      - 一つのモデルに対し、複数の異なる教師なし学習で学習。論文では Masked LM（隠した単語の予測タスク）と、Next Sentence Prediction; NSP（隣接文の予測タスク）の2つで学習させたとのこと。この際に Transformer の双方向的（周囲全体）な学習を採用。
    - 2. fine-tuning
      - タスクに応じた学習。
  - Transformerという考え方
    - [Transformer: A Novel Neural Network Architecture for Language Understanding](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
    - [論文解説 Attention Is All You Need (Transformer)](http://deeplearning.hatenablog.com/entry/transformer)
    - [BERT解説：自然言語処理のための最先端言語モデル](https://ainow.ai/2019/05/21/167211/)
