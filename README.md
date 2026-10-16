<div align="center">

# MIDI-REPA

### Improving Text-to-Music Generation with Symbolic Representation Alignment

<span style="white-space:nowrap;">Song-Ze Yu</span>&nbsp;·
<span style="white-space:nowrap;">Fang-Duo Tsai</span>&nbsp;·
<span style="white-space:nowrap;">Yi-Hsuan Yang</span>

<sub>National Taiwan University</sub>

<br/>

[![arXiv](https://img.shields.io/badge/arXiv-2610.20967-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.20967)
[![Demo](https://img.shields.io/badge/Demo-midi--repa.vaclis.net-862633.svg)](https://midi-repa.vaclis.net/)
[![AMTFlow](https://img.shields.io/badge/AMTFlow-GitHub-181717.svg?logo=github&logoColor=white)](https://github.com/vaclisinc/AMTFlow)
[![JamendoMIDI](https://img.shields.io/badge/JamendoMIDI-Hugging_Face-ffcc4d.svg?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/vaclis/JamendoMIDI)

</div>

<p align="center">
  <a href="https://midi-repa.vaclis.net/">
    <img src="assets/paper-figure.webp" alt="MIDI-REPA overview: an AMT system turns training audio into symbolic targets that are aligned with an early FluxAudio-S DiT layer, with panels on what, where, and when to align" width="100%">
  </a>
  <br/>
  <sub>Click the figure to open the demo page with audio examples.</sub>
</p>

## Code & Resources

<table>
  <tr>
    <td width="50%" valign="top">
      <a href="https://github.com/vaclisinc/AMTFlow"><img src="assets/amtflow-pipeline.webp" alt="AMTFlow pipeline diagram" width="100%"></a>
      <br/><sub>PIPELINE</sub>
      <br/><a href="https://github.com/vaclisinc/AMTFlow"><b>AMTFlow ↗</b></a>
      <br/>Transcription pipeline used for the paper. To reproduce, run with <code>--config paper</code>.
    </td>
    <td width="50%" valign="top">
      <a href="https://huggingface.co/datasets/vaclis/JamendoMIDI"><img src="assets/jamendomidi-viewer.webp" alt="JamendoMIDI dataset viewer" width="100%"></a>
      <br/><sub>DATASET</sub>
      <br/><a href="https://huggingface.co/datasets/vaclis/JamendoMIDI"><b>JamendoMIDI ↗</b></a>
      <br/>Multi-instrument MIDI transcriptions of 51,077 instrumental Jamendo tracks. CC BY-NC-SA 4.0.
    </td>
  </tr>
</table>
