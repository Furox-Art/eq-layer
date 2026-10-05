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

## Coarse Discourse Corpus

EQ-Layer audits direct disagreement supervision using the Coarse Discourse corpus.

- Original dataset/code repository: `google-research-datasets/coarse-discourse`
- Original repository license statement: CC-by
- Runtime archive: ConvoKit Reddit Coarse Discourse package
- Pinned archive SHA-256: `33cc25e906e677881c0031c1e134460b1e389fac707f2e52fe59b484dc6d0b61`
- Direct disagreement examples observed: 3,394 / 101,219 labelled utterances

The corpus is **not vendored**. The direct-disagreement classifier is retained only as an experimental research artifact because the verified test result is not strong enough for policy routing. See `eval/results/coarse_disagreement_no_go.json`.

Recommended citation:
- Zhang, A. X., Culbertson, B., & Paritosh, P. (2017). *Characterizing Online Discussion Using Coarse Discourse Sequences*. ICWSM 2017.

## DialogBank

EQ-Layer uses the public English DialogBank DiAML annotations only for a label-availability audit.

- English dialogues successfully parsed: 15
- Dialogue acts parsed: 2,680
- Direct target counts: correction=4, disagreement=1, agreement=55, selfCorrection=98
- Parse failures in the verified audit: 0

No classifier is trained from these counts. In particular, `selfCorrection` is not relabelled as a user correcting the assistant. The source files are not vendored, and this repository does not make a new licensing claim over DialogBank. See `eval/results/dialogbank_label_inventory.json`.

## DBDC3

EQ-Layer audits text-only dialogue-breakdown detection using the English DBDC3 release.

- Runtime archive: `https://dbd-challenge.github.io/dbdc3/data/DBDC3.zip`
- Pinned archive SHA-256: `736229795dc3732f8e6bb421f094dc820ef944fef9d9d320d4110ef992b60e85`
- Effective source used: `dbdc3_revised` only, so original/revised duplicate dialogues are not mixed
- Official revised dev: 4,124 annotated system turns / 414 dialogues
- Official revised eval: 1,998 annotated system turns / 200 dialogues

The upstream GitHub repository does not expose a repository license file that EQ-Layer can safely reinterpret for the dataset. The archive is therefore **not vendored**, and users must verify upstream terms independently.

The verified text-only breakdown model is a **no-go** for routing: on the untouched revised eval split it reached ROC-AUC 0.5057. See `eval/results/dbdc3_breakdown_no_go.json`.


## EmpatheticDialogues

EQ-Layer uses the upstream EmpatheticDialogues **test split** only as one
external held-out response-level evaluation stratum.

- Upstream repository: `facebookresearch/EmpatheticDialogues`
- Repository commit recorded by the builder:
  `9649114c71e1af32189a3973b3598dc311297560`
- Dataset archive:
  `https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz`
- License: Creative Commons Attribution-NonCommercial 4.0 International
  (CC BY-NC 4.0)
- Planned held-out contribution: 60 conversation prefixes

The dataset is **not vendored** into the MIT-licensed repository. The held-out
builder downloads it at runtime, records the downloaded archive SHA-256, and
uses only conversation prefixes ending in a user turn. The original human
reference reply is excluded from model input.

Recommended citation:
- Rashkin, H. et al. (2019). *Towards Empathetic Open-domain Conversation
  Models: A New Benchmark and Dataset*. ACL 2019.

## Taskmaster-3

EQ-Layer uses Taskmaster-3 as the task-oriented/repair portion of the external
held-out response-level benchmark.

- Upstream repository: `google-research-datasets/Taskmaster`
- Pinned commit:
  `d92cb6af3005f1dc09c39e75e7daf4a04905e00b`
- Files used by the builder:
  - `TM-3-2020/data/data_00.json`
  - `TM-3-2020/data/data_01.json`
- License statement in the upstream Taskmaster-3 README: CC BY 4.0
- Planned held-out contribution:
  - 30 repair/clarification user turns
  - 30 general task user turns

Taskmaster-3 explicitly contains repair and clarification phenomena such as
users correcting previously understood entities. EQ-Layer does not use these
examples to train its policy or subtext components; they are sampled only for
the frozen external A/B evaluation set.

The Taskmaster source text is not copied into the repository. The builder
downloads pinned upstream files at runtime, records each file SHA-256, and the
workflow stores the generated 120-case evaluation set as a GitHub Actions
artifact rather than source code.

Recommended citation:
- Byrne, B. et al. Taskmaster series; see the Taskmaster-3 upstream README for
  the dataset-specific citation and collection details.
