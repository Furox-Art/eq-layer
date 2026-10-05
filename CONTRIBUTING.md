# Contributing

## Start with a case

Not with a policy. If you can describe a transcript where the current behaviour is wrong, or where two policies seem equally right, file it as an issue using the Case template.

Ambiguous cases are the most useful ones. "This is why I filed it" is a valid answer to the policy dropdown.

## Policy changes go through an RFC

Copy `.github/RFC_TEMPLATE.md` into `.github/rfcs/NNN-short-title.md` and open a pull request. An RFC is expected to include cases where the proposal is the *wrong* move — a policy that cannot lose cannot be evaluated.

## Adding to the eval set

Append one JSON object per line to `eval/cases.jsonl`:

```json
{
  "id": "case-0NN",
  "transcript": [{"role": "user", "content": "..."}],
  "expected_policy": "mirror_specific",
  "expected_concepts": ["word"],
  "notes": "why this case is hard",
  "author": "your github handle",
  "status": "seed"
}
```

Keep `expected_concepts` short. Three specific words beat a vague list; the specificity metric is only as good as the concepts behind it.

## Before you open a PR

- `python eval/run.py` still runs
- You have not weakened a score to make a case pass
- Policy additions include at least one case where the policy loses

That last point is the one that gets skipped. It is the one that matters.