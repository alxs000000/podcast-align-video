<p align="center">
  <img src="assets/brand-icon.png" width="176" alt="podcast-align-videoのロゴ">
</p>

<h1 align="center">podcast-align-video</h1>

<p align="center">
  <a href="../README.md" lang="en">English</a> · 日本語 · <a href="README.zh-CN.md" lang="zh-CN">简体中文</a> · <a href="README.ko.md" lang="ko">한국어</a> · <a href="README.es.md" lang="es">Español</a>
</p>

<p align="center">
  <strong>英語を、耳と目で追いかける。</strong><br>
  英語の音声やYouTube動画から、話している単語が金色に光る字幕動画を作れます。
</p>

<p align="center">
  <img src="assets/demo.gif" width="960" alt="音声に合わせて英単語を一つずつ金色で強調する字幕動画のデモ">
</p>

聞こえた単語がどこなのか、字幕を見ながら追えるようにするツールです。手元の英語音声、またはログイン不要のYouTube動画を一つ渡すと、普段の動画プレーヤーで再生できるMP4を生成します。英語のポッドキャストや長い会話を、音声と英文を対応させながら聞きたい人に向いています。

v0.1で作る字幕は英語です。日本語への翻訳、単語の意味の表示、日本語音声の文字起こしには対応していません。このページは紹介と操作手順の日本語版です。

[音声付きデモを見る](https://github.com/alxs000000/podcast-align-video/releases/download/v0.1.0/podcast-align-video-demo.mp4) · [v0.1.0をダウンロード](https://github.com/alxs000000/podcast-align-video/releases/tag/v0.1.0)

デモは[AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)のES2002a、話者A、77.0〜81.4秒の抜粋です。CC BY 4.0に基づき、切り出し・字幕追加・動画への変換を行っています。GIFには音声がありません。[素材の出典と測定結果](DEMO.md)は英語で掲載しています。

## できること

- 手元の英語音声、または公開YouTube動画一つから字幕動画を作成。
- 黒い背景の中央に英文を表示し、発話中の単語を金色で強調。
- 全編の動画に加えて、長い無音を削った動画も必要な場合に生成。
- 長尺処理の途中で止まっても、同じ入力・設定で実行すると検証済みの途中結果から再開。
- 動画のほか、元音声、文字起こし、単語ごとの時刻データを保存。

字幕は自動生成なので、聞き取りや単語のタイミングに誤りが残ることがあります。

## 動かすために必要なもの

v0.1はLinux、またはWindows上のWSL2と、CUDAが使えるNVIDIA GPUを対象にしています。操作はターミナルから行います。macOS、GPUなしのPC、Windowsだけでの実行、クリック操作だけのGUIには対応していません。

FFmpeg／FFprobe（libass・libx264対応）、NVIDIAドライバー、`curl`・`tar`・`bzip2`、Playwright Chromiumに必要な共有ライブラリを事前に用意してください。setupはPython 3.12と専用環境をユーザー領域へ構築しますが、`sudo`や`apt`でシステムの不足を補うことはありません。

文字起こし用のCohereモデルには利用申請が必要です。利用規約への同意とHugging Faceのアクセストークンは、利用者自身で用意します。

## 初回の準備

Linux／WSL2のターミナルで実行します。

```bash
git clone https://github.com/alxs000000/podcast-align-video.git
cd podcast-align-video
./scripts/setup.sh
export PATH="$HOME/.local/bin:$PATH"
cp config/default.toml config/local.toml
```

上の`export PATH`は現在のターミナルだけに有効です。新しいターミナルでコマンドが見つからない場合も実行してください。既定の設定では、モデルや実行環境は`~/.local/share/podcast-align-video`に保存されます。

[Cohereのモデルページ](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)でアクセスを申請し、承認済みアカウントの読み取り用Hugging Faceトークンを用意してから、次を実行します。トークンは画面に表示されません。

```bash
read -r -s -p 'Hugging Face token: ' HF_TOKEN
printf '\n'
export HF_TOKEN
podcast-align-video models fetch --config config/local.toml
unset HF_TOKEN
podcast-align-video doctor --config config/local.toml
```

`models fetch`はモデル名・固定revision・ライセンス／モデルページ・概算容量を表示し、ダウンロードします。利用申請や規約への同意は代行しません。`doctor`は必要な環境を確認します。実際の`run`はモデルを自動ダウンロードせず、不足があれば重い処理を始める前に終了します。

## 動画を作る

```bash
# 手元の英語音声
podcast-align-video run ./episode.flac --config config/local.toml

# ログイン不要のYouTube動画一つ
podcast-align-video run 'https://www.youtube.com/watch?v=VIDEO_ID' --config config/local.toml

# 保存先と、削る無音の長さを指定
podcast-align-video run ./episode.wav --config config/local.toml \
  --output-dir ./my-output --silence-threshold 7.5 --device cuda:0
```

`VIDEO_ID`は実際の動画IDに置き換えてください。プレイリスト、ログイン・cookieが必要な動画、非公開動画は対象外です。音声ファイルはFFmpegで読み取れる形式を使えます。

保存先を省略すると、`./outputs/<整えたタイトル>-<入力と設定の識別子12桁>/`に次のファイルができます。

```text
video.mp4                 # 全編の動画
video-speech-cut.mp4      # 長い無音を削れた場合だけ作成
source.<元の拡張子>       # 元音声
transcript.txt            # 英語の文字起こし
word-timings.json         # 単語ごとの時刻
run-manifest.json         # 処理内容と結果の記録
run.log                   # 実行ログ
```

元のローカル音声は内容を変えずにコピーします。YouTube音声は実行のたびに再取得し、取得した形式を保ったまま保存します。完成動画は既定で1920×1080・30fpsのH.264、音声は48 kHz AAC（指定ビットレート192 kbps）です。

無音の削除は既定で5秒以上が対象です。削除対象がなければ全編だけを出力します。無音削除だけに失敗した場合は、全編の動画を残して警告を返します。

## 長い音声でも描画が速い理由

ブラウザで各文の改行・文字サイズ・単語の位置を測定し、その結果をASS／libassで動画として描き直します。この方式を`hybrid`と呼んでいます。ブラウザを音声の長さだけ録画する必要がなく、字幕の配置を保ちながら描画できます。

処理の流れは、音声変換 → Sileroによる発話検出 → Cohereによる文字起こし → Qwenによる時刻合わせ → MFAによる補正 → 字幕配置の測定 → 動画生成です。MFAは必須です。正常終了したうえで一部の区間の補正が使えない場合、その区間だけQwenの時刻に戻します。

実機試験では、RTX 4070 SUPERとNVENCの`p4`設定で、5時間4分の素材を23分59秒で描画しました。これは動画描画だけの実測で、文字起こしなどを含めた全工程の時間ではありません。既定のエンコーダーは`libx264 -preset veryfast -crf 20`で、NVENCはTOML設定で選べます。

## 再開と作業ファイルの整理

同じ入力・設定・モデルrevisionなら、検証済みの途中結果を再利用します。指定した保存先に別の入力・設定の結果がある場合は、上書きせず終了します。

```bash
# 削除予定の作業ファイル容量を確認するだけ
podcast-align-video clean JOB_ID

# 完了したjobの作業ファイルを削除
podcast-align-video clean JOB_ID --yes
```

`JOB_ID`は処理結果のjob IDに置き換えます。完成した成果物とモデルは削除しません。同じjobや一つのGPUでの同時実行を防ぐ仕組みはありません。

## ライセンスと詳しい技術情報

コードは[Apache-2.0](../LICENSE)、同梱のGeistフォントは[OFL-1.1](../LICENSES/OFL-1.1.txt)です。モデルとデモ素材にはそれぞれの利用条件があります。[第三者素材の表示](../THIRD_PARTY_NOTICES.md)も参照してください。入力する音声やYouTube動画を取得・加工する権利は、利用者自身で確認してください。

Python API、開発者向けテスト、処理の詳細は[英語版README](../README.md#python-api)にあります。
