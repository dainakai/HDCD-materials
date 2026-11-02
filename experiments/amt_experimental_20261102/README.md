# 新版モデルによる実験データ処理と上位80件レビュー

2026年11月2日に接続された16 TBディスクを読み取り専用で使用します。原画像、旧モデル、旧レビューを変更しません。数値ホログラム評価で固定したエンジンとOursのモデルを使い、学習は行いません。

- 対象は260715の14系列、各カメラ10,917枚。主カメラC002、副カメラC001。画像番号1に対する処理フレーム番号は0。
- 校正は系列に対応する6組の変形係数・焦点距離を維持。再構成前端は74,900 µm、終端177,200 µm、全段階3回。校正の前端差はglassplate offsetで明示的に調整。
- 計測・MinIPのpadding1,536、画像特徴2,048。MinIPは0,2,…,1022面と終端1023面。
- 背景除去は既存のwindow129、anchor stride32を維持。新しい原画像から全量再計算。
- 粒子計測→同じ3D追跡・平滑化→候補生成→固定済み上流モデルと衝突モデルによる推論→各系列上位80件→画像作成。
- 順位固定は座標抑制前。最終スコア、候補時刻、消失ID、継続IDによる決定的な並べ順で80件を固定。
- 新判定はcollision / non_collision / ambiguous。旧レビューは表示・転記しない。上位80件の難しい例を置換しない。

出力は`../audit/revision-experimental-20261102/`です。原画像の連続番号とサイズ、各カメラ先頭・中央・末尾の全デコードを事前確認し、各入力manifestを保存します。全画像は背景処理中にデコードされます。原画像全体のSHAを確認したという意味ではありません。

```bash
.venv/bin/python experiments/amt_experimental_20261102/inventory.py
.venv/bin/python experiments/amt_experimental_20261102/pipeline.py --scenes 12 --smoke 128
.venv/bin/python experiments/amt_experimental_20261102/pipeline.py
```

接続確認は128フレームで実行し、背景除去から80個のレビューGIFまで通過しました。この部分系列の候補は主レビューへ登録しません。全量処理は当初GPU0で開始し、現在は承認済みのGPU間差を記録して系列単位で分散します。既存のGPU1処理はそのまま完走させ、空き状態を確認してから参加させます。実行条件、照合方法、運用手順は[分散処理の記録](DISTRIBUTED.md)にまとめています。

`campaign-state.json`が実行状態、各系列`stages/*.json`が完了証跡、`inference/top80.parquet`が固定候補、`review/manifest.json`が表示画像と候補の対応です。途中までの出力は保護し、検査せず上書きしません。

`review_server.py`はローカル8770番で起動します。既存8769番の報告書サーバーが`/experimental-review/`だけを中継し、既存Quick Tunnelを再利用します。レビュー専用URLのキーはローカルの`review-access.json`に保存され、アクセス後はcookieとCSRFトークンで判定を保護します。キー、DB、生画像、モデルは配信しません。判定DBは候補集合・モデル・予測のhashと結び付けられ、変更履歴を保持します。

```bash
.venv/bin/python experiments/amt_experimental_20261102/test_review_store.py
.venv/bin/python experiments/amt_experimental_20261102/review_server.py --port 8770
```

旧レビューとのイベント対応や物理整合性・重力衝突解析は、新レビューの判定後に行います。合成学習に用いた旧実験分布は遡及的に変更しません。

処理監視ページは `/experimental-monitor/`、レビュー画面は `/experimental-review/` です。いずれもローカルで稼働し、既存のQuick Tunnelから利用します。監視の進捗定義、レビュー機能、保存・取り消し、検証結果と停止方法は[Web運用記録](WEB.md)にまとめています。
