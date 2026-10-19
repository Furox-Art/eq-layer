<!--
Thanks for contributing. Read CONTRIBUTING.md first.

Cases are the main entry point. Policy changes need an RFC in `.github/rfcs/`
before a pull request. Do not weaken a score to make a case pass, and do not
include credentials or private transcript data.
-->

## What this changes

<!-- One paragraph. What is different after this merges, and for whom. -->

Closes #

## Type

- [ ] Case or eval addition
- [ ] Policy change (RFC linked below)
- [ ] Bug fix in the layer or CLI
- [ ] npm package or Agent Skill packaging
- [ ] Evaluation harness or metric
- [ ] Documentation
- [ ] CI, build, or release process

RFC: <!-- link to .github/rfcs/NNN-short-title.md, or "not applicable" -->

## Cases and evidence

<!--
What proves the change is correct? Paste commands and real output. For policy
changes, name at least one case where the new policy is the wrong move. For
metric changes, include the before/after result and state what it does and does
not establish.
-->

```
```

## Score and claim integrity

- [ ] I did not weaken a score or adjust a case to make it pass.
- [ ] A policy addition includes at least one case where the policy loses.
- [ ] Evidence levels (`verified`, `direct_learned`, `derived`, `structural`, `fallback`) remain accurate.
- [ ] Documentation claims stay within the measured result; no EQ or benchmark claim is implied beyond the data.
- [ ] Not applicable.

## Checks run

Paste results, including exit codes.

- [ ] `python -m pip install -e ".[dev]" "pytest-cov>=5,<8"`
- [ ] `python eval/run.py`
- [ ] `node tools/crosscheck.js`
- [ ] `python -m pytest -q -rs tests/ --cov=eq_layer --cov-branch --cov-report=term-missing --cov-fail-under=100`
- [ ] `python -m compileall -q eq_layer eval`
- [ ] `npm test` (if `npm/`, `package.json`, or `skills/` changed)
- [ ] `python -m build` and `python -m twine check dist/*` (if packaging changed)
- [ ] Not applicable

```
```

## Review notes

<!--
Anything a reviewer should look at first, anything you deliberately left out,
and anything you are unsure about. Saying "I am not sure this is right" is more
useful here than a confident guess.
-->
