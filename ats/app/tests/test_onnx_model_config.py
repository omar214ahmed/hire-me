from pathlib import Path

from pipeline.models import ensure_model_config_file


def test_ensure_model_config_file_creates_config_from_gliner_config(tmp_path: Path) -> None:
    model_dir = tmp_path / "gliner"
    model_dir.mkdir()

    fallback_config = model_dir / "gliner_config.json"
    fallback_config.write_text('{"model_type": "test"}', encoding="utf-8")

    assert not (model_dir / "config.json").exists()

    created = ensure_model_config_file(model_dir)

    assert created is True
    assert (model_dir / "config.json").read_text(encoding="utf-8") == '{"model_type": "test"}'
