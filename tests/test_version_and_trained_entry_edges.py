
import pytest

from eq_layer import _version
from eq_layer import trained_affect as ta
from eq_layer import trained_intent as ti
from eq_layer import trained_subtext as ts
from eq_layer.tracker import AnnotationStanceResolver


def test_source_version_fallback_rejects_wrong_metadata_table(monkeypatch, tmp_path):
    assert _version._project_table('version = "bad"') == ""
    assert (
        'version = "3.2.1"'
        == _version._project_table('[project]\nversion = "3.2.1"\n[tool.ruff]\nx=1').strip()
    )
    fake = tmp_path / "eq_layer" / "_version.py"
    fake.parent.mkdir()
    fake.write_text("")
    (tmp_path / "pyproject.toml").write_text('[tool.other]\nversion = "wrong"\n')
    monkeypatch.setattr(_version, "__file__", str(fake))
    assert _version._source_checkout_version() is None
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "7.0.1"\n')
    assert _version._source_checkout_version() == "7.0.1"


def test_detect_version_recovers_if_distribution_is_missing(monkeypatch):
    def missing(package):
        raise _version.PackageNotFoundError("eq-layer")

    monkeypatch.setattr(_version, "version", missing)
    monkeypatch.setattr(_version, "_source_checkout_version", lambda: None)
    assert _version.detect_version() == "0.0.0+unknown"
    monkeypatch.setattr(_version, "_source_checkout_version", lambda: "8.0.0")
    assert _version.detect_version() == "8.0.0"


def test_annotation_stance_requires_external_verification():
    resolver = AnnotationStanceResolver()
    assert resolver.resolve([]).label == "unknown"
    assert resolver.resolve([], {"stance": "user_right"}).confidence == 1.0
    assert resolver.resolve([], {"stance": "user_wrong"}).confidence == 1.0
    with pytest.raises(ValueError, match="Unsupported stance"):
        resolver.resolve([], {"stance": "invented"})


def test_training_cache_key_falls_back_for_archive_distributions(monkeypatch):
    monkeypatch.setattr(
        ti, "bundled_records_path", lambda: (_ for _ in ()).throw(FileNotFoundError())
    )
    assert ti._bundled_cache_key(0.4, 0.08) == (":resources:", 0, 0.4, 0.08)


def test_trained_intent_cache_tracks_bundled_file_mtime(monkeypatch, tmp_path):
    file = tmp_path / "intent_train.jsonl"
    file.write_text("{}\n")
    monkeypatch.setattr(ti, "bundled_records_path", lambda: file)
    key = ti._bundled_cache_key(0.4, 0.08)
    assert key[0] == str(file)
    assert key[1] == file.stat().st_mtime_ns


def test_trained_intent_refuses_missing_artifact_path(monkeypatch, tmp_path):
    fake = tmp_path / "outside" / "trained_intent.py"
    fake.parent.mkdir()
    fake.write_text("")
    monkeypatch.setattr(ti, "__file__", str(fake))
    with pytest.raises(FileNotFoundError, match="no filesystem path"):
        ti.bundled_records_path()


def test_from_emobank_and_from_xdailydialog_load_real_local_training(monkeypatch, tmp_path):
    csv = tmp_path / "samples.csv"
    csv.write_text(
        "split,V,A,D,text\n"
        + "".join(f"train,{5 if i % 2 else 1},3,3,Sample example {i}\n" for i in range(6))
    )
    trained = ta.TrainedAffect.from_emobank(csv)
    assert trained.training_examples == 6

    text = tmp_path / "dialogs.txt"
    text.write_text(
        "".join(
            f"Sample message {i} __eou__\tDaily\t{1 if i % 2 else 2}\t{0 if i % 2 else 4}\n"
            for i in range(10)
        )
    )
    model = ts.TrainedSubtext.from_xdailydialog(text)
    assert model.training_examples == 10
    assert model.predict_signals("Sample message 3").act in {"inform", "question"}


def test_trained_intent_reconcile_uses_real_posterior_and_no_raw_guess():
    model = ti.TrainedIntent()
    no_belief = ti.HeuristicIntent().infer([{"role": "user", "content": "Hello"}])
    assert model.reconcile_with_dialogue_evidence(no_belief, [], None) == no_belief
