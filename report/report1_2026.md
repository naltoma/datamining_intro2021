## 課題レポート1：機械学習してみよう

[Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) からデータをダウンロードし、以下の課題に取り組め。

---

### Level 0: 準備。（報告不要）

- (a) ダウンロード。
  - `student+performance.zip` というファイルがダウンロードされるはずだ。
- (b) zipファイルの展開。
  - 展開すると `student/` フォルダが作成される。フォルダ内に以下のファイルが入っていることを確認しよう。
    - `student.txt` : 特徴量の説明書。
    - `student-mat.csv` : 数学コースのデータ（`;` 区切り）。
    - `student-por.csv` : ポルトガル語コースのデータ（`;` 区切り）。<u>今回利用するのはこちら。</u>
    - `student-merge.R` : 2つのデータセットを結合するためのRスクリプト（今回は使わない）。
- 備考
  - 目的変数は「G3 ≥ 12 を合格（1）、G3 < 12 を不合格（0）」とする。設定方法は後述する。この「12点以上」というのは、20点満点の6割を合格基準とする本学基準に合わせた設定です。

---

### Level 1. データセット調査。（15分想定）

Student Performance がどのようなデータセットなのか概説せよ。少なくとも以下の内容を含めること。

- タスク種別は何か（回帰か分類か）。
- 何を目的として収集または構築したのか。
- サンプル数・特徴量数を確認せよ。

````{hint}
おおよその説明は Web ページと student.txt に書かれている。

データを読み込んで確認するコード例：

```Python
import pandas as pd
df = pd.read_csv("student-por.csv", sep=";")
print(df.shape)   # (サンプル数, 列数)確認
print(df.head())  # 冒頭数サンプル確認
```
````

---

### Level 2. 特徴量調査。（30分想定）

データセットの各特徴について調べ、以下の点をレポートに述べよ。

目的変数は「G3 ≥ 12 を合格（1）、G3 < 12 を不合格（0）」として作成すること。この「12点以上」というのは、20点満点の6割を合格基準とする本学基準に合わせた設定です。

- (a) **特徴量の種類**：30列の特徴量を「数値（numeric）」と「カテゴリ（categorical）」に分類し、それぞれ何列あるかを述べよ。さらにカテゴリ変数のうち2列を選び、どのような値をとるか説明せよ。
- (b) **クラス分布**：合格・不合格のサンプル数を確認せよ。偏りはあるか。

````{hint}
- 各特徴の説明は `student.txt` の「Variable Information」欄を参照しよう。
- 目的変数の作成例：

```Python
y = (df["G3"] >= 12).astype(int)
print(y.value_counts())
```

- 数値列とカテゴリ列の確認例：

```Python
print(df.dtypes)
```

`object` 型になっている列がカテゴリ変数（文字列）である。
````

```{note}
今回の学習では、**数値特徴量のみ** を用いてまず試すことを主題にしている。このためカテゴリ変数は無視している。（カテゴリ変数を含む他特徴量の扱いはレポート2で取り上げる）
```

---

### Level 3. 分類学習に用いるモデルを選べ。（15分想定）

レポートには、(a) 選んだモデルの名前と、(b) そのモデルが持つハイパーパラメータを2個程度、簡潔に解説せよ。パラメータの解説は直訳程度で構わない。

```{hint}
モデルは分類タスクに適用できるものから自由に選んで構わない。[Flow Chart](https://scikit-learn.org/stable/machine_learning_map.html) を参考にするのも良い。

- モデルが持つパラメータは大別して次の2種類に分けられる。
  - **ハイパーパラメータ**: 開発者自身が手動で設定する必要があるパラメータ。
  - 上記以外のパラメータ（重みとも呼ばれる）: 学習データから自動で獲得するパラメータ。
- モデルを選んだら、一度ドキュメントを参照して使い方やハイパーパラメータを確認しよう。
  - 例）[sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) の **Parameters** 欄には手動調整可能な引数が並んでいる。
```

---

### Level 4. 実際にコードを書いて分類学習せよ。（1〜3時間想定）

今回の意図は、全体の流れを理解することである。実験結果が悪くても構わないので、流れを理解しながら取り組もう。

レポートには (a) 主要コードを示して解説せよ。また (b) 分類結果についても示して解説せよ。

**条件**

- `student-por.csv`（ポルトガル語コース）を使用すること。
- 特徴量には以下の**数値列13列のみ**を用いること（カテゴリ列・G1・G2 は除くこと）。

```Python
numeric_cols = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"
]
```

- 目的変数は Level 2 で作成した合否ラベル（G3 ≥ 12 → 1, それ以外 → 0）とすること。
- train/test 分割（test_size=0.2, random_state=2026）を行い、テストデータで評価すること。
- 評価指標は **Accuracy（正解率）と F1スコアの両方**を報告すること。

````{hint}
分類タスクの基本的な流れ：

```Python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# 特徴量・目的変数の準備
X = df[numeric_cols]
y = (df["G3"] >= 12).astype(int)

# train/test 分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2026
)

# モデルの訓練（モデルは自分で選ぶ）
model = ...         # 選んだモデルのインスタンスを用意
model.fit(X_train, y_train)

# テストデータで評価
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))
```

- scikit-learn の公式例：[Classification examples](https://scikit-learn.org/stable/auto_examples/index.html#classification)
````

```{note}
この段階ではカテゴリ変数（文字列の列）を**意図的に除外**している。
「除いた情報を活かせば精度が上がるかもしれない」と感じたら、オプションか、レポート2で扱おう。
```

---

### Options：余裕があれば取り組んでみよう

- **例1**：データを見て「この情報を組み合わせると予測に役立つかもしれない」と思う特徴量を自分で設計し、効果を検証してみよう。結果を示すだけではなく、コードと設計した特徴量について説明すること。改善しなかった場合、その理由も考察できるとなお良い。

- **例2**：混同行列（confusion matrix）を使い、どのクラスでよく間違えているかを確認してみよう。

  ```Python
  from sklearn.metrics import ConfusionMatrixDisplay
  ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
  ```

- **例3**：合格クラス（1）と不合格クラス（0）のサンプル数は偏っているか。この偏りが Accuracy の解釈にどう影響するか考えてみよう（ヒント：不均衡データ）。

- **例4**：G1（1学期成績）・G2（2学期成績）も特徴量に加えてみよう。性能はどう変わるか。また、実用的な予測場面においてこれらを使うことの問題点は何か考えてみよう。

```{note}
データの読み込み方やエラーの対処でつまずいたら、気軽に質問しよう。
```
