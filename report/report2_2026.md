## 課題レポート2：前処理してみよう

課題レポート1で取り組んだ Student Performance データセットについて、以下の課題に取り組め。

---

### Level 1. 前処理の適用。（30分想定、報告不要）

レポート1では**数値列のみ**を特徴量として用いた。今回はレポート1で除外していた**全カテゴリ列**を数値に変換して特徴量に加える。

変換には、列がとる値の種類に応じて以下の2種類の方法を使い分けること。

**(1) バイナリエンコーディング**：`"yes"/"no"` や2値の列を `0/1` に変換する。

**(2) One-Hotエンコーディング**：3種類以上の値をとる名義カテゴリ列を複数の0/1列に展開する。

````{hint}
バイナリエンコーディングの例：

```Python
# 変換用のマップを用意
binary_map = {
    "yes": 1, "no": 0,
    "GP": 0, "MS": 1,
    "F": 0,  "M": 1,
    "U": 0,  "R": 1,
    "LE3": 0, "GT3": 1,
    "T": 0,  "A": 1
}

# 変換対象の列姪を用意
binary_cols = [
    "school", "sex", "address", "famsize", "Pstatus",
    "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic"
]

# 元データdfを複製し、df_encとして用意。
# df_encに前処理を適用する。
df_enc = df.copy()
for col in binary_cols:
    df_enc[col] = df_enc[col].map(binary_map)
```

One-Hotエンコーディングの例（`pd.get_dummies` を使う）：

```Python
nominal_cols = ["Mjob", "Fjob", "reason", "guardian"]
df_enc = pd.get_dummies(df_enc, columns=nominal_cols, drop_first=True)
```

`pd.get_dummies` は文字列列を自動的にダミー変数に展開する。`drop_first=True` は[多重共線性](https://best-biostatistics.com/correlation_regression/dummy.html)を緩和するための設定で、ラベルを一つ削除します（[公式ドキュメント参照](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)）。なお、pandas 2.0 以降では展開後の値が `0/1` ではなく `False/True` で表示される場合があるが、scikit-learn はどちらも同様に扱うため動作上の問題はありません。

変換前後の1番目のサンプルの確認例：

```Python
feature_cols = [c for c in df_enc.columns if c not in ["G1", "G2", "G3"]]

print("=== 変換前（カテゴリ列のみ） ===")
print(df[binary_cols + nominal_cols].iloc[0])

print("\n=== 変換後（全特徴量） ===")
print(df_enc[feature_cols].iloc[0])
```
````

```{hint}
Level 1は実行するだけで良く、レポートでの報告は不要です。
```

---

### Level 2. 特徴量の変化。（30分想定）

前処理の前後で特徴量がどのように変わったかを確認し、レポートに示せ。少なくとも以下の点を含めること。

- (a) 前処理前後で特徴量の列数がどう変わったか。
- (b) バイナリエンコーディングを適用した列を1つ選び、変換前後の値を示せ。
- (c) One-Hotエンコーディングを適用した列を1つ選び、変換前後がどのように変わったかを示せ。

````{hint}
列数の確認：

```Python
# 前処理前
numeric_cols = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"
]
print("前処理前の特徴量数:", len(numeric_cols))

# 前処理後（G1, G2, G3 は目的変数・除外列なので除く）
feature_cols = [c for c in df_enc.columns if c not in ["G1", "G2", "G3"]]
print("前処理後の特徴量数:", len(feature_cols))
```

変換前後の値の確認例（`higher` 列の場合）：

```Python
# 元データがdfに保存されており、かつ、前処理適用後のデータが df_enc に保存されている前提の例
print("変換前:", df["higher"].value_counts())
print("変換後:", df_enc["higher"].value_counts())
```

One-Hotエンコーディングの変換後の確認例（`Mjob` 列の場合）：

```Python
# 変換前
print(df["Mjob"].value_counts())

# 変換後（展開されたダミー列を確認）
mjob_cols = [c for c in df_enc.columns if c.startswith("Mjob_")]
print(df_enc[mjob_cols].head())
```
````

---

### Level 3. 新しい条件による実験。（1〜2時間想定）

前処理後のデータを用い、レポート1と同じ条件で分類タスクを実行し、結果を比較・考察せよ。

**条件**

- 特徴量は Level 1・2 で準備したエンコーディング済みの列をすべて使うこと（G1・G2・G3 は除く）。
- モデル・train/test 分割の条件（test_size、random_state）はレポート1と同一にすること。
- 評価指標は **Accuracy と F1スコアの両方**を報告すること。

レポートには、(a) レポート1の結果と比較できるよう2つの実験結果を表にして示し、(b) 前処理が性能に寄与したかどうかを、結果の良し悪しを踏まえて考察せよ。

````{hint}
前処理後のデータで学習・評価する流れ：

```Python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

feature_cols = [c for c in df_enc.columns if c not in ["G1", "G2", "G3"]]
X2 = df_enc[feature_cols]
y2 = (df_enc["G3"] >= 12).astype(int)

# レポート1と同じ random_state を使うこと
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

model2 = ...   # レポート1と同じ種類のモデル
model2.fit(X2_train, y2_train)
y2_pred = model2.predict(X2_test)

print("Accuracy:", accuracy_score(y2_test, y2_pred))
print("F1 score:", f1_score(y2_test, y2_pred))
```

結果の比較表の例：

| 条件 | 特徴量数 | Accuracy | F1スコア |
|------|---------|----------|---------|
| レポート1（数値のみ） | 13 | ? | ? |
| レポート2（+カテゴリENC） | ? | ? | ? |
````

```{note}
レポート2では前処理を適用し、比較することを目的としている。性能が向上しなかったから減点するということはない。ありのまま報告し、なぜそうなったかを考察すること。
```

---

### Options：余裕があれば取り組んでみよう

- **例1**：数値列に対して標準化（StandardScaler）を追加したとき、性能はどう変わるか実験してみよう。変わった・変わらなかった理由も考察せよ。

  ```Python
  from sklearn.preprocessing import StandardScaler

  numeric_cols = [
      "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
      "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"
  ]
  scaler = StandardScaler()
  X2_train_scaled = X2_train.copy()
  X2_test_scaled  = X2_test.copy()
  X2_train_scaled[numeric_cols] = scaler.fit_transform(X2_train[numeric_cols])
  X2_test_scaled[numeric_cols]  = scaler.transform(X2_test[numeric_cols])
  ```

- **例2**：データを見て「この情報を組み合わせると予測に役立つかもしれない」と思う特徴量を自分で設計し、効果を検証してみよう。結果を示すだけではなく、コードと設計した特徴量について説明すること。改善しなかった場合、その理由も考察できるとなお良い。

- **例3**：G1（1学期成績）・G2（2学期成績）を特徴量に加えると性能はどう変わるか実験してみよう。また、この2列を加えることが実用的な場面でなぜ問題になりうるかを考察せよ（キーワード：データリーク）。
