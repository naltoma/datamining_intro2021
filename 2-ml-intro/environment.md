# サンプルコードを動かすための環境構築
- Python外部パッケージのインストール
  - PC上で直接実行したい場合1（venv仮想環境, pipで個別にインストール）
  - PC上で直接実行したい場合2（Anaconda）
  - Google Colaboratoryを使う場合
- 開発環境
- 動作確認

---
## Python外部パッケージのインストール
### PC上で直接実行したい場合1
- Python 3.7系 or 3.8系最新版をインストール。
  - ``python --version`` でバージョンを確認。3.7系 or 3.8系であることを確認できたら以下の手順で環境構築。
- venvを使って仮想環境構築。pip終了後はpythonで実行できる。

```shell
python -m venv .venv/dm
source .venv/dm/bin/activate
pip install requests numpy pandas matplotlib seaborn scikit-learn jupyterlab

# サンプルコードを実行したいなら、それをダウンロードする。
# 例えば test.pyなら、、
python test.py

# ipynb形式（jupyer notebook形式）なら、
# jupyter-lab でjupyterサーバを起動。
# 起動すると自動でブラウザで起動したディレクトリを開きます。
# そこから実行したいファイルを開いて実行しよう。
jupyter-lab
```

- ターミナルを終了すると、仮想環境から自動で抜けてしまう。再度仮想環境に入り直す場合には、2行目の ``source .venv/dm/bin/activate`` を実行。

---
### PC上で実行したい場合2
細かいことを気にしないなら[Anaconda](https://www.anaconda.com)でまとめてインストール。

---
### Google Colaboratoryを使う場合
- Googleアカウントを作成し、[Colab](https://colab.research.google.com/notebooks/intro.ipynb) にアクセス。
- 新規ファイルから始めたいなら、ファイルメニューから新規作成を選ぶ。
- サンプルコードを実行したいなら、ipynbファイルをGoogleドライブにアップロード。それをダブルクリックで開く際に Colab で選ぼう。選べない場合にはアプリ検索から探せるはず。

---
## 開発環境
- [VSCode](https://azure.microsoft.com/ja-jp/products/visual-studio-code/) か [PyCharm](https://www.jetbrains.com/pycharm/) を推奨。デバッグしやすいです。
- VSCodeなら[インタプリタの設定](https://ie.u-ryukyu.ac.jp/~tnal/2020/prog1/vscode.pdf)を参考に、``which python`` で表示されるインタプリタを設定してやろう。

---
## 動作確認
[iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html#sphx-glr-auto-examples-datasets-plot-iris-dataset-py)の plot_iris_dataset.py や plot_iris_dataset.ipynb を実行すると、このページにあるようなグラフが描画されるならOK。
