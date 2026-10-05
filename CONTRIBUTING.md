# Contributing

## Start with a case

Not with a policy. If you can describe a transcript where the current behaviour is wrong, or where two policies seem equally right, file it as an issue using the Case template.

Ambiguous cases are the most useful ones. "This is why I filed it" is a valid answer to the policy dropdown.

## Policy changes go through an RFC

Copy `.github/RFC_TEMPLATE.md` into `.github/rfcs/NNN-short-title.md` and open a pull request. An RFC is expected to include cases where the proposal is the *wrong* move — a policy that cannot lose cannot be evaluated.

## Adding to the eval set

Append one JSON object per line to `eval/cases.jsonl`:

Append one line of JSON per case to `eval/cases.jsonl` — one line, not
pretty-printed, since it is parsed line by line:

```json
{
  "id": "case-0NN",
  "transcript": [{"role": "user", "content": "..."}],
  "expected_policy": "mirror_specific",
  "expected_concepts": ["word"],
  "annotated": {"stance": "user_right"},
  "notes": "why this case is hard",
  "author": "your github handle",
  "status": "seed"
}
```

Keep `expected_concepts` short. Three specific words beat a vague list; the
specificity metric is only as good as the concepts behind it.

`annotated` is optional. Use `stance` (`user_right` / `user_wrong` /
`unknown`) when the transcript makes it clear who is right, and `subtext`
(`exhaustion`, `resignation`) for signals no keyword can recover. Omitting
`stance` means `unknown`, which is a legitimate answer — do not guess to make
a case pass.

## Running the harness

```bash
python eval/run.py
```

It exits non-zero if any case mismatches or if any policy is unreachable from
the case set. An unreachable policy is not a neutral result: it means the
taxonomy contains something no case justifies.

## Verifying without Python

If you have no working Python interpreter, `tools/crosscheck.js` translates
the layer to JS and diffs the selection table against the cases. It parses the
policy table and keyword sets straight out of the Python source, so the two
cannot drift on the values that matter. It exits non-zero on any mismatch or
unreachable policy.

```bash
node tools/crosscheck.js
```

It does not check that the Python itself compiles. Run `python eval/run.py`
before opening a PR if you have an interpreter.

## Before you open a PR

- `python eval/run.py` still runs (or `node tools/crosscheck.js` passes)
- You have not weakened a score to make a case pass
- Policy additions include at least one case where the policy loses

That last point is the one that gets skipped. It is the one that matters.