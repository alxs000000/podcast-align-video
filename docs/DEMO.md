# Demo provenance

The README demo is a real `podcast-align-video` output, not a design mockup.

## Source

- Corpus: [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)
- Meeting/channel: `ES2002a`, headset speaker A
- Excerpt: 77.0–81.4 seconds
- Duration: 4.4 seconds
- Reference transcript: “Hi, I'm David and I'm supposed to be an industrial designer.”
- License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- Changes: the excerpt was clipped from the source recording, synchronized with generated word-highlight subtitles, and transcoded to the published demo formats

The tracked `docs/assets/demo.gif` is silent. The v0.1.0 GitHub Release includes `podcast-align-video-demo.mp4` with the excerpt audio and generated subtitles.

## Generation and checks

The clip was processed by the default Cohere → Qwen → MFA pipeline and the hybrid Chromium-measurement/ASS-rendering path at 1920×1080, 30 fps, H.264/AAC.

- Predicted/reference words: 11/11
- WER: 0.0
- MFA application rate: 100%
- Absolute word-boundary error: median 0.130 sec, p95 0.313 sec across 22 boundaries

These numbers document this small licensed example; they are not claimed as general benchmark results.
