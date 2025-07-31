# 課題5： AI活用における偏見・公平性について
機械学習もLLMもデータ駆動で行うため、その前提となるデータ自身に含まれるバイアスの影響を受けてしまうことは避けがたい。バイアスを減らす研究開発も取り組まれているがまだ発展途上である。

本課題では論文調査を通し、機械学習やLLMにおいてバイアスが与える影響やその対策について調査した結果を報告してください。ただし以下の条件や報告事項を守るようにしてください。

- やること
    - 指定キーワードを含めて論文検索し、一般公開されている論文の中から一つを選んでください。
        - 付属図書館での契約により「学内からだと無料で閲覧できる論文」もあります。そのため論文探しをする際には学内LANに接続した状態でやることをお勧めします。
    - 論文を読み、「AI活用における偏見・公平性」について小論文を作成してください。
- ヒント
    - 検索サイトは何を使っても良いですが、論文に絞り込みたいなら以下を推奨します。
        - [CiNii Research](https://cir.nii.ac.jp)
        - [ACM digital library](https://dl.acm.org)
        - [Google Scholar](https://scholar.google.co.jp)
        - 琉大付属図書館の[ありんくりんサーチ](https://www.lib.u-ryukyu.ac.jp)
    - 自分で探す・選ぶことができない人は、以下から選んでください。
        - [レクチャーシリーズ：「人工知能の今」〔第11 回〕AI 倫理指針における課題](https://doi.org/10.11517/jjsai.35.6_845), 2020
        - [AI に不適合なアルゴリズム回避論：機械的な人事採用選別と自動化バイアス](https://doi.org/10.24798/jicp.7.2_1), 2023
        - [From Prejudice to Parity: A New Approach to Debiasing Large Language Model Word Embeddings](https://arxiv.org/abs/2402.11512), 2024
        - [Bias and Fairness in Large Language Models: Evaluation and Mitigation Techniques](https://www.researchgate.net/publication/392124172_Bias_and_Fairness_in_Large_Language_Models_Evaluation_and_Mitigation_Techniques), 2025　＊LLM全般
        - [Racial bias in AI-mediated psychiatric diagnosis and treatment: a qualitative comparison of four large language models](https://www.nature.com/articles/s41746-025-01746-4), 2025　＊医療
        - [Debiasing Diffusion Model: Enhancing Fairness through Latent Representation Learning in Stable Diffusion Model](https://arxiv.org/abs/2503.12536), 2025　＊画像生成
- 論文条件
    - (1) 指定キーワード。最低限以下に示す必須単語を設定して検索してください。
        - 必須単語1: 「機械学習、人工知能、AI、LLM」やそれに類する単語から1つを選ぶ。
        - 必須単語2: 「偏見、バイアス、公平性、フェアネス、倫理」やそれに類する単語から1つを選ぶ。
        - 「類する単語」かどうか自身で判断しかねる場合には、質問チャンネルで相談ください。
    - (2) 日本語の場合1000字以上、英語の場合500単語以上で書かれた論文を選ぶこと。
- レポート条件
    - (1) レポート本文は1000字以上1500字未満とし、報告事項(1)〜(5)を含めること。事項毎のボリュームバランスは自由とする。ただし報告事項(1)〜(5)をどこで述べているのか明瞭に判断できるように見出し（もしくは節）を付与すること。
    - (2) [パラグラフ・ライティング](http://www.ams.eng.osaka-u.ac.jp/user/ishihara/?p=566)で段落構成すること。
    - (3) 著者が述べていることか、あなた自身の意見なのか、どちらなのかを明確に読み取れるように書くこと。
- 報告事項
    - (1) 検索時に用いた必須単語1、必須単語2を報告してください。これ以外の単語を追加した場合にはそれも報告してください。
    - (2) 検索結果のスクリーンショットを報告してください。
    - (3) 選んだ論文の概要を300字〜500字程度でまとめてください。
        - 生成AIを使っても良いですが、その場合には (a) 概要（abstract）については自身で目を通して大凡間違いがなさそうだということを確認し、(b) 利用した状況を確認できる内容を報告（例えばURL共有設定してそのリンクを記載）すること。この生成AI利用に関する報告部分は、字数としてカウントしません。
    - (4) 社会やビジネスに与えるリスクや問題点について、「あなた自身の考え」を述べてください。
    - (5) 選んだ論文の出典を明示してください。
        - 出典の書式は[情報処理学会の原稿執筆案内](https://www.ipsj.or.jp/journal/submit/ronbun_j_prms.html)「2.8.4　原稿執筆上の注意事項」に準ずる。[テンプレートファイル](https://www.ipsj.or.jp/journal/submit/style.html)の参考になるだろう。
        - 指定しているのは「出典の書式」のみです。それ以外の書式指定はありません。
        - 参考文献として列挙するだけでは不十分です。適切な箇所で本文から参照すること。
- オプション例
    - 「必須単語2」が機械学習モデル（LLMを含む）にどのような影響を与えるか、具体的な事例を挙げて説明してください。
    - ChatGPT以前と以降との論文を最低1件ずつ調査し、どのような違いがあるかまとめてください。
