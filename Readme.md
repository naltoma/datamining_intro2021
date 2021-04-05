# データ・マイニング
- [琉大知能情報コース](https://ie.u-ryukyu.ac.jp/~tnal/2021/dm/)3年次向けの選択科目、かつ、工学部の融合選択科目。2021年度から前期開講になるため、Numpy/Pandas/Matplotlib周りの演習も加える予定。

---
## 授業の流れ
- 導入
  - [データマイニング外観](./1-intro/intro.md)
  - [前提](./1-intro/env.md)
- Part 1: 機械学習入門
  - [機械学習外観](./2-ml-intro/ml-intro.md)
  - [コード例（線形回帰）](./2-ml-intro/regression_diabetes.ipynb)
  - [プログラミング演習（Numpy, Pandas, Matplotlib入門）](./2-ml-intro/data_wrangling.ipynb)
  - [機械学習の中身を覗いてみよう](./2-ml-intro/ml-intro2.md)
  - [1次元データセットを通した勾配法の理解](./2-ml-intro/gradient_descent.ipynb)
- Part 2: 特徴量エンジニアリング
  - 特徴ベクトル、数値データに対する前処理: [資料](./3-feature-engineering/preprocess-number.md), [コード例](./3-feature-engineering/preprocess_numerical.ipynb)
  - カテゴリデータに対する前処理: [資料](./3-feature-engineering/preprocess-category.md), [コード例](./3-feature-engineering/preprocess_categorical.ipynb)
- Part 3: 特徴量エンジニアリング：自然言語処理
  - シソーラス、カウントと推論に基づいた設計、次元削減
  - 代表的な自然言語処理（日本語を例に）
  - テキストデータに対する特徴量設計2（word2vecによる分散表現）
  - 転移学習外観
- タスク＆アルゴリズム例
  - グラフマイニング
  - 頻出パターンマイニング
  - 推薦システム
  - 時系列データ分析

---
## 参考文献
- データマイニング・機械学習全般
  - [Data Mining: Practical Machine Learning Tools and Techniques, 3rd Edition](http://www.cs.waikato.ac.nz/ml/weka/book.html)
  - [Python Machine Learning: Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow 2, 3rd Edition](https://www.amazon.co.jp/dp/1789955750)
  - [Pythonで動かして学ぶ！あたらしい機械学習の教科書 第2版](https://www.shoeisha.co.jp/book/detail/9784798159911)
  - [機械学習のエッセンス 実装しながら学ぶPython、数学、アルゴリズム](https://www.sbcr.jp/product/4797393965/)
- 特徴量エンジニアリング
  - [機械学習のための特徴量エンジニアリング――その原理とPythonによる実践](https://www.oreilly.co.jp/books/9784873118680/)
  - [5.3. Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)
