# Third-party data

## EmoBank

EQ-Layer can train its learned affect regressor from EmoBank.

- Upstream: JULIELab/EmoBank
- Pinned commit: `248ce2a43e165a66d31aeaed83cff9641d6654e0`
- File used: `corpus/emobank.csv`
- License: CC-BY-SA-4.0
- Authors: Sven Buechel and Udo Hahn
- Recommended citation:
  - Buechel, S. and Hahn, U. (2017). *EmoBank: Studying the Impact of Annotation Perspective and Representation Format on Dimensional Emotion Analysis*. EACL 2017.
  - Buechel, S. and Hahn, U. (2017). *Readers vs. Writers vs. Texts: Coping with Different Perspectives of Text Understanding in Emotion Annotation*. LAW 2017.

The EmoBank dataset is **not vendored** into this MIT-licensed repository. Training and evaluation scripts download the pinned upstream CSV at runtime. Users of the dataset or trained artifacts remain responsible for complying with the upstream CC-BY-SA-4.0 terms.

## XDailyDialog

EQ-Layer can train its learned dialogue-act/emotion signal tracker from the English XDailyDialog files.

- Upstream: liuzeming01/XDailyDialog
- Pinned commit: `6e7ecf54c9f169215b4b8c18995c7aac74117127`
- Files used:
  - `data/en_train_human.txt`
  - `data/en_dev_human.txt`
  - `data/en_test_human.txt`
- XDailyDialog repository license: Apache-2.0
- XDailyDialog authors: Zeming Liu, Ping Nie, Jie Cai, Haifeng Wang, Zheng-Yu Niu, Peng Zhang, Mrinmaya Sachan, and Kaiping Peng.
- The English dialogues originate from DailyDialog.

**License-chain caution:** the XDailyDialog repository carries Apache-2.0, while common distributions of the original DailyDialog English corpus identify it as CC-BY-NC-SA-4.0. EQ-Layer therefore does not vendor the English corpus or a trained subtext artifact and does not assert that a trained artifact is covered by EQ-Layer's MIT license. Users must verify and follow the applicable upstream data terms for their use case.

Recommended citations:
- Liu, Z. et al. (2023). *XDailyDialog: A Multilingual Parallel Dialogue Corpus*. ACL 2023.
- Li, Y. et al. (2017). *DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset*. IJCNLP 2017.
