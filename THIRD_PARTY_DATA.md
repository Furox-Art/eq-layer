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
