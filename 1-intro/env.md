# 前提

## 想定学生
- プログラミングのスキルが必須。
  - 授業ではコード例を多数示す。課題で書いてもらう部分もあるが、コード例があるためゼロから書くことはない。ただし下記2点のスキルが必要である。
    - (1) **コード例を読んで、実行して、動作を確認しながら自分なりにカスタマイズして利用する力**。
    - (2) ライブラリの使い方を検索し、使う力。
- プログラミング言語としては Python を使う。
  - ただしC言語やJava等で条件分岐、ループ処理、関数定義、配列処理あたりをしたことがあるなら、Python自体が初めてでも対応可能な範囲だと思われる。Python初心者ならば、[Progate](https://prog-8.com/)での事前勉強を強く勧める。じっくり取り組みたいのであれば、[知能情報コース「プログラミング1」](https://ie.u-ryukyu.ac.jp/~tnal/2024/prog1/static/Readme.html)も参考になるだろう。

---
## ファイル名の拡張子を表示する
macOS, Windowsを問わず、どちらも標準設定ではファイル名の拡張子を隠す設定になっている。これが原因となり「同じファイル名が複数あるが、どれが何を保存しているのか良く分からない」状況になることが多いし、セキュリティ上も拡張子を表示しておいたほうが良い。

- ファイル拡張子の表示
    - Windows: [Windows の一般的なファイル名拡張子](https://support.microsoft.com/ja-jp/windows/windows-の一般的なファイル名拡張子-da4a4430-8e76-89c5-59f7-1cdbbc75cb01)
        - エクスプローラーを起動する。「表示」タブの「表示/非表示」グループを選び、「ファイル名拡張子」にチェックを付ける。
    - macOS:[Macでファイル名拡張子を表示する/非表示にする](https://support.apple.com/ja-jp/guide/mac-help/mchlp2304/mac)
        - Finderを起動する。「設定」メニューから「詳細」を選び、「すべてのファイル名拡張子を表示」を選択する。

---
## 開発環境の前に
- 知能情報学生
  - e18, e19, e20, e22 の学生は Miniconda で環境構築済みのはず。**残念ながら、conda環境では一部のプログラム(自然言語処理, spacy)が動作しない可能性があります**。「開発環境の設定」を参考に、pipで環境構築することを推奨します。
  - e21学生は pip + venv で部分的に環境構築済みのはず。適宜モジュールをインストールしよう。
- 他コース学生
  - 環境が全くわからないため、まずは動作確認をしてみよう。一度もPythonプログラミング環境を構築したことがないのであれば動作確認をスキップして、環境構築からやろう。

---
## 動作確認
- step 1: サンプルコードの用意。
  - [The Iris Dataset](http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)にある　``plot_iris_dataset.ipynb`` か、もしくは ``plot_iris_dataset.py`` をダウンロード。
    - 拡張子が ``.ipynb`` が Jupyter Notebook 形式。Google Colab上で実行する際にはこちらを選ぼう。
- step 2: VSCodeか、Pythonインタプリタで実行。
  - step 1のサイト上に掲載されているように2つのグラフが生成されたらOK。

何かしらのエラーが出るなら、エラーに基づいて対応が必要。ググるなり當間なりに相談ください。以下は、ゼロから環境構築する場合（macOS編, Windows編）についてと、Google Colabを使う例を示します。

```{note}
- おまけ: [デバッグ実行入門](https://ie.u-ryukyu.ac.jp/~tnal/2024/prog1/static/debug.html)

動作確認しながらコードを読み進めるためにデバッガの利用（デバッグ実行）に慣れることを推奨します。
```

---
## 開発環境の設定
### ゼロから環境構築：macOS編
- 想定
  - macOS: 13.x以降。
    - 12.xぐらいでも恐らく問題ないが、未確認。
  - Python: 3.9〜3.11系。
    - 2024年4月時点では、3.12系は避けたほうが良い。（参考: <a href="https://pytorch.org/get-started/locally/">PyTorch PREREQUISITES</a>）

---
#### pip を使う場合
- step 1: Python仮想環境を構築。
以下では ``~/.venv/dm/`` 以下に仮想環境を構築しようとしている例です。
```shell
which python3 # /usr/bin/python3 想定
/usr/bin/python3 -m venv ~/.venv/dm
source ~/.venv/dm/bin/activate
which python # ~/.venv/bin/python 想定
pip install --upgrade pip
pip install scikit-learn pandas matplotlib seaborn jupyterlab ginza ja_ginza
```

- step 2: VSCodeをインストール。
  - 参考: [VSCodeのインストールから実行まで](http://ie.u-ryukyu.ac.jp/~tnal/2023/prog1/vscode.pdf)
- step 3: 動作確認。
  - 冒頭のサンプルコードで動作確認してみよう。
- [venv + pip な環境の補足](./venv.md)

---
#### condaを使う場合（**推奨しません**）
2022年4月時点では、condaでは[自然言語処理ライブラリspacy](https://spacy.io/)が動作しません。前述の pip を推奨します。

- step 1: [Anaconda公式サイト](https://www.anaconda.com)から自分にあったインストーラを選んでインストール。
- step 2: Pythonインタプリタのパスを確認。
  - ``which python``
  - 恐らく anaconda なり miniconda なりの名称を含むパスが表示されるはず。それをコピーしておこう。
- step 3: VSCodeをインストール。
  - [VSCode公式サイト](https://azure.microsoft.com/ja-jp/products/visual-studio-code/)から自分にあった版をダウンロード。
    - M1チップ搭載モデルを購入した人は「Apple Silicon」。
    - Intel CPU搭載モデルを購入した人は「Intel Chip」。
    - どちらか良く分からない人は「Universal」。
  - 「Visual Studio Code.app」がダウンロードされるので、これを Applicationsフォルダにドラッグ&ドロップで設置。これでインストール終了。インストール後はDockに登録しておくと便利。
- step 4: VSCodeの設定。
  - VSCodeを起動。
  - Welcome, Release Notesタブが表示されていると思うので、その中の Tools and languages から **Python** をクリック。
  - 新規ファイルを作成。
  - ``print("hello")`` ぐらいを書いて保存。ファイル名を ``sample.py`` とする。
  - VSCodeウィンドウ左下、紫色背景色に「Python 2」や「Python 3」とか表示されている箇所をクリック。
    - インタプリタ変更のため、``Enter interpreter path...`` を選択。
    - 入力欄に入力できる状態になるので、step 2 でコピーしたPythonインタプリタのパスをペーストし、Enter。
- step 5: 動作確認。
  - 冒頭の方法で動作確認してみよう。

---
### ゼロから環境構築：Windows編
- 想定
  - Python: 3.9〜3.11系
- 大別して WSL2 により Linux系OSをインストールしてそこで環境構築するか、Windows上で直接環境構築するかの選択肢がある。

```{warning}
担当教員はWindows環境がないため、動作確認していません。環境構築で躓く場合には相談ください。
```

---
#### WSL2を使う場合
- step 1: [WSL2 + pyenv + Visual Studio Codeを使ったPython3の開発環境の作り方](https://aadojo.alterbooth.com/entry/2020/08/19/095654)を参考に、VSCode上で hello world 出力するところまでやる。
  - 良く分からない場合には[インストール大会](https://ie.u-ryukyu.ac.jp/students/install/2021/post/os/windows/windows/)の「python環境」「wsl2設定」が参考になるかもしれません。　
- step 2: 追加パッケージのインストール。
  - WSL2側で以下のコマンドを実行。
```shell
pip install --upgrade pip
pip install scikit-learn pandas matplotlib seaborn jupyterlab ginza ja_ginza
```
- step 3: 動作確認。
  - 冒頭のサンプルコードで動作確認してみよう。

---
#### Windows上で直接環境構築する場合
- step 1: [Visual Studio Code を使用して Python 初心者向けの開発環境をセットアップする](https://docs.microsoft.com/ja-jp/learn/modules/python-install-vscode/)を参考にVSCodeインストールするところまでやる。
  - **オペレーティングシステムを選択する** でWindowsを選択するのを忘れずに。
- step 2: 追加パッケージのインストール。
  - コマンドプロンプトを起動し、以下のコマンドを実行。
```shell
pip install --upgrade pip
pip install scikit-learn pandas matplotlib seaborn jupyterlab
```
- step 3: 動作確認。
  - 冒頭のサンプルコードで動作確認してみよう。

---
### Google Colabを使う場合
何らかの理由でPython実行環境を準備できない人は、Google Colaboratory と呼ばれる無料のクラウド環境を利用してプログラミングできます。GoogleのサービスなのでGoogleアカウント（無料）が必要です。

- step 1: 下記URLをブラウザで開く。
  - [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb?hl=ja)
- step 2: 右上の「ログイン」から、G-mailアカウントでログイン。
- step 3: ファイルメニューから「ノートブックを新規作成」を選択。
  - Untitled0.ipynb という名前のついたファイルが作成される。もし名前を変更したい場合には、拡張子（.ipynb）はそのままにし、その前の「Untilted0」だけを変更しよう。気にしない場合にはそのままでOK。
- step 4: 実行してみる。
  - 「右向き矢印」と「灰色背景」の欄があるはず。この欄を「セル」と呼んでおり、このセル内にコードを記述していく。
  - 記述し終えたら「右向き矢印」をクリックするか、Shift + Enter すると、　そのセル内のコードを実行してくれる。
    - 1回目の実行は、環境構築のため時間がかかります。
- step 5: 動作確認。
  - 冒頭のサンプルコードで動作確認してみよう。

---
## Python演習のための準備
### Pythonとは
プログラミング言語の一種。データサイエンスや機械学習（≒AI）を中心に広まってきました。ファイル名の拡張子は ``.py``（スクリプトファイル）, ``.ipynb``（ノートブック）。授業ではノートブックで開発をするため、拡張子は .ipynb を使います。

(colab)=
### Google Colabratory
通称 Google Colab。Googleのアカウント（無料）を作成することで、誰でもブラウザ上でプログラムを編集したり実行したりすることができるようになる。個別にPCへインストールする必要がなく、同じ環境を利用することができることから授業での利用も広まってきた。使いやすいが、Google Colab特有の使いづらさもあることに注意。

- 特徴
    - ユーザからのリクエストが届くと、Googleが固有の実行環境（仮想マシン）をクラウド上に用意します。ユーザは用意された仮想マシン内でノートブックを実行することになります。
- 利点
    - ネットワーク接続できるPCならば、いつでも誰でも同一環境で演習することが可能。
    - プログラムと実行結果を一つのファイルにまとめることが可能。
    - 実行し終えた状態で続きのプログラムを書き、続けて実行することができる。
- 欠点
    - 一定時間アイドル状態（操作していない状態）が続くと、仮想マシンが削除されます。削除されてしまった場合には、改めて冒頭から実行し直す必要があります。
    - 連続実行時間は最長12時間。無料アカウントでこれ以上長く利用することはできませんし、有料でも最長24時間の制限があります。

### Google Colabratoryを使う準備
- Googleのアカウントを作成。
    - 個人アカウントを既に持っている人は、それを流用して構いません。今回の授業用に別途作成しても構いません。やりやすい方でアカウントを用意してください。
- case 1: colab経由でノートブック作成する。
  - [Colabratory](https://colab.research.google.com/)にアクセス。
  - 画面左上の「ファイル」メニューをクリックし、「ノートブックを新規作成」を選ぶ。
- case 2: googleドライブ上でノートブック作成する。
  - googleドライブを開く。（[google](https://www.google.com/)にアクセスし、アプリから「ドライブ」を選ぶ）
  - 「+新規」から「その他」=>「Google Colabratory」を選ぶ。

### 基本的な使い方
- 動作確認
  - ``print("Hello, world!")`` と書き、左側にある実行ボタンをクリック。
  - その下に Hello, world! と出力されたらOK。
- セル単位で実行する。
- セルには「テキストセル（Markdown）」と「コードセル（Python）」がある。
- Markdown
    - [Markdown記法 サンプル集](https://qiita.com/tbpgr/items/989c6badefff69377da7)
