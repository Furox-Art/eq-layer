"""Atheris harness for eq_layer.json_contract.loads.

The strict decoder is the single ingress for externally supplied JSON
transcripts and cases; it must reject duplicate keys, non-finite numbers and
non-UTF-8 text by raising ValueError, never by crashing.
"""

import sys

import atheris

from eq_layer.json_contract import loads


def TestOneInput(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    text = fdp.ConsumeUnicodeNoSurrogates(4096)
    try:
        loads(text)
    except ValueError:
        pass


atheris.Setup(sys.argv, TestOneInput)
atheris.Fuzz()
