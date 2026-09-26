<p align="center">
  <img src="assets/brand-icon.png" width="176" alt="podcast-align-video 标志">
</p>

<h1 align="center">podcast-align-video</h1>

<p align="center">
  <a href="../README.md" lang="en">English</a> · <a href="README.ja.md" lang="ja">日本語</a> · 简体中文 · <a href="README.ko.md" lang="ko">한국어</a> · <a href="README.es.md" lang="es">Español</a>
</p>

<p align="center">
  <strong>听英语时，看清正在说哪个词。</strong><br>
  将英语音频或一个公开的 YouTube 视频，转换成逐词高亮的字幕视频。
</p>

<p align="center">
  <img src="assets/demo.gif" width="960" alt="随音频逐个高亮英语单词的字幕视频演示">
</p>

说到的单词会变成金色，让你跟着字幕追踪英语播客或对话。输入本地英语音频，或一个无需登录的 YouTube 视频链接，即可生成能在普通播放器中播放的 MP4。

v0.1 只生成英语字幕，不提供中文翻译、词义解释或中文语音识别。本页是中文介绍与使用指南，软件本身仍以命令行为主。

[观看带声音的演示](https://github.com/alxs000000/podcast-align-video/releases/download/v0.1.0/podcast-align-video-demo.mp4) · [下载 v0.1.0](https://github.com/alxs000000/podcast-align-video/releases/tag/v0.1.0)

演示素材来自 [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)，ES2002a，发言者 A，77.0–81.4 秒，以 CC BY 4.0 授权。修改包括截取片段、添加字幕和转码。上面的 GIF 没有声音。[来源与测量结果（英文）](DEMO.md)。

## 功能与运行条件

- 支持 FFmpeg 能解码的本地音频，或单个公开、无需登录的 YouTube 视频。
- 生成完整视频；如果检测到符合条件的长静音，还会生成去静音版本，默认阈值为 5 秒。
- 保存原始音频、英语转写文本、逐词时间戳和运行记录。
- 同样的输入、设置和模型版本可以复用通过验证的中间结果，便于继续长时间任务。

需要 Linux 或 Windows 的 WSL2、支持 CUDA 的 NVIDIA GPU，以及 FFmpeg／FFprobe（带 libass 和 libx264）、NVIDIA 驱动、`curl`、`tar`、`bzip2` 和 Playwright Chromium 所需的共享库。v0.1 不支持 macOS、纯 Windows 或无 GPU 环境，也没有图形界面。自动转写和时间戳可能有误。

## 初次安装

在 Linux／WSL2 的 Bash 终端中运行：

```bash
git clone https://github.com/alxs000000/podcast-align-video.git
cd podcast-align-video
./scripts/setup.sh
export PATH="$HOME/.local/bin:$PATH"
cp config/default.toml config/local.toml
```

setup 会在用户目录中安装 Python 3.12、专用环境和 Chromium，不会执行 `sudo` 或 `apt`。默认数据目录为 `~/.local/share/podcast-align-video`。每次打开新终端时，如果找不到命令，请重新运行上面的 `export PATH`。

在 [Cohere 模型页面](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)申请访问权限并接受条款。获批后，使用自己的 Hugging Face 读取令牌。以下输入不会在屏幕上显示令牌：

```bash
read -r -s -p 'Hugging Face token: ' HF_TOKEN
printf '\n'
export HF_TOKEN
podcast-align-video models fetch --config config/local.toml
unset HF_TOKEN
podcast-align-video doctor --config config/local.toml
```

`models fetch` 会先显示模型名称、固定版本、授权页面和大致容量，再进行下载；不能代替你同意条款。`doctor` 检查运行环境。`run` 不会下载模型，缺少必需组件时会在开始耗时处理前退出。

## 生成视频

```bash
# 本地英语音频
podcast-align-video run ./episode.flac --config config/local.toml

# 一个公开的 YouTube 视频；请替换 VIDEO_ID
podcast-align-video run 'https://www.youtube.com/watch?v=VIDEO_ID' --config config/local.toml

# 指定输出目录，删除 7.5 秒以上的静音
podcast-align-video run ./episode.wav --config config/local.toml \
  --output-dir ./my-output --silence-threshold 7.5 --device cuda:0
```

不支持播放列表、需要 cookie 或登录的视频、私有视频。YouTube 音频每次运行都会重新下载，保留下载时的编码格式。本地音频会逐字节复制。

默认输出在 `./outputs/<处理后的标题>-<输入及设置指纹的前12位>/`。`video.mp4` 是完整视频；只有实际删除了静音，才生成 `video-speech-cut.mp4`。另外保存 `source.<原扩展名>`、`transcript.txt`、`word-timings.json`、`run-manifest.json` 和 `run.log`。去静音处理失败不会影响已经完成的完整视频。

## 为什么长音频也能较快生成视频？

浏览器只测量字幕的字体大小、换行和每个词的位置，之后由 ASS／libass 绘制视频，无需实时录制浏览器。默认输出为 1920×1080、30 fps、H.264，音频为 48 kHz AAC，指定比特率为 192 kbps。

处理流程为 Silero 语音检测 → Cohere 转写 → Qwen 对齐 → MFA 校正 → 浏览器测量 → 视频绘制。MFA 是必需组件；正常运行后，只有无法有效校正的局部区段才回退到 Qwen 时间戳。

同样的输入与设置可自动恢复任务；显式输出目录已有不同任务的结果时不会覆盖。`podcast-align-video clean JOB_ID` 仅显示已完成任务的工作文件清理计划，添加 `--yes` 才执行，成果和模型会保留。没有同一任务或同一 GPU 的并发保护。

## 许可与技术说明

代码采用 [Apache-2.0](../LICENSE)，Geist 字体采用 [OFL-1.1](../LICENSES/OFL-1.1.txt)。模型和素材各有使用条件，详见[第三方声明](../THIRD_PARTY_NOTICES.md)。请确认自己有权下载和处理输入素材。

Python API、测试和更多技术说明见[英文 README](../README.md#python-api)。
