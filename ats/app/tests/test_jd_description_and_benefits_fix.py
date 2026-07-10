"""
Regression tests for a real-world JD format (bare "Description" /
"Requirements" / "Benefits" headers, bullet-only body, no colons) that
previously broke the section splitter in two ways:

Bug 1 (content silently dropped): "Description" was not in
JD_SECTION_PATTERNS at all, so it fell back to the generic "header"
bucket. Lines under an unrecognized header are reclassified individually
by keyword cues (see jd_chunker.classify_chunk) - but any line with *no*
cue keyword at all (e.g. "Work closely with Data Scientists and ML
Engineers to translate research models into production-ready services")
had nowhere else to go and stayed in "header", which is only ever
scanned for a "job title" NER label and is excluded from
`build_jd_query`'s embedding text - i.e. that sentence vanished from the
pipeline entirely.

Bug 2 (no benefits field): "Benefits" headers were folded into the same
`job_meta` bucket as salary/location/employment-type, and there was no
"benefits" key anywhere in the extraction output at all - a JD's entire
perks/benefits list (health plan, PTO, stipends, ...) was extracted and
then thrown away.

Fix: "Description"/"Job Description"/"Role Description" are now
recognized headers that route to the "experience" bucket (so every line
under them keeps real section context instead of being reclassified
line-by-line), and "Benefits"/"Perks" now route to their own "benefits"
bucket, deterministically split into one list entry per bullet and
exposed as `extracted["benefits"]` / `JDExtraction.benefits`.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.jd_chunker import split_jd_sections

REAL_WORLD_JD = """Description
Collaborate in the design, development and maintenance of robust backend applications and services to serve ML inferences (FastAPI/Flask or Node.js)
Build and optimize pipelines for real-time or batch inference processing
Work closely with Data Scientists and ML Engineers to translate research models into production-ready services
Support the identification and integration of emerging technologies to improve system performance and the end-user experience.
Requirements
Advanced English for conversation
Experience with PyTorch, ONNX and OpenVINO for model optimization and execution
Curiosity
Collaborative mindset
Benefits
CLT employment contract: your security and full labor rights guaranteed from day one
SulAmerica health plan
SulAmerica dental plan
Birthday day off: a special day off during your birthday month
"""


def test_description_header_is_recognized_and_not_leaked_as_content():
    sections = split_jd_sections(REAL_WORLD_JD)
    all_text = "\n".join(sections.values())
    assert "Description" not in all_text


def test_no_content_lost_from_a_bare_description_section():
    """The two sentences with no strong keyword cue (management-speak,
    not "Docker/Python"-style content) must still land somewhere real -
    not the inert 'header' catch-all bucket."""
    sections = split_jd_sections(REAL_WORLD_JD)
    assert "header" not in sections or not sections["header"].strip()

    experience_text = sections.get("experience", "")
    assert "Work closely with Data Scientists" in experience_text
    assert "Support the identification and integration" in experience_text
    assert "Collaborate in the design" in experience_text


def test_benefits_get_their_own_section_not_merged_into_job_meta():
    sections = split_jd_sections(REAL_WORLD_JD)
    benefits_text = sections.get("benefits", "")

    assert "CLT employment contract" in benefits_text
    assert "SulAmerica health plan" in benefits_text
    assert "Birthday day off" in benefits_text
    # Must not bleed into (or be sourced only from) job_meta.
    assert "SulAmerica health plan" not in sections.get("job_meta", "")


def test_requirements_still_split_correctly_alongside_new_sections():
    sections = split_jd_sections(REAL_WORLD_JD)
    assert "Advanced English for conversation" in sections["requirements"]
    assert "PyTorch" in sections["requirements"]


def _import_jd_processor_with_stubbed_models():
    """pipeline.jd_processor imports pipeline.models at module scope,
    which pulls in torch/gliner - stub it out the same way
    test_jd_optional_marker_and_job_type_fix.py's monkeypatches do,
    just one level earlier (before import) so this file has zero heavy
    dependencies at collection time."""
    if "pipeline.models" not in sys.modules:
        fake_models = types.ModuleType("pipeline.models")

        class _StubRegistry:
            @classmethod
            def ner(cls):
                raise RuntimeError("must be monkeypatched by the test")

        fake_models.ModelRegistry = _StubRegistry
        sys.modules["pipeline.models"] = fake_models

    import pipeline.jd_processor as jd_processor
    return jd_processor


class _FakeNerNoEntities:
    def predict_entities(self, text, labels, threshold=0.3, flat_ner=True):
        return []


def test_extract_from_jd_with_sections_exposes_benefits_list(monkeypatch):
    jd_processor = _import_jd_processor_with_stubbed_models()
    monkeypatch.setattr(
        jd_processor,
        "ModelRegistry",
        type("FakeRegistry", (), {"ner": classmethod(lambda cls: _FakeNerNoEntities())}),
    )

    out = jd_processor.extract_from_jd_with_sections(REAL_WORLD_JD)
    benefits = out["extracted"]["benefits"]

    assert any("CLT employment contract" in b for b in benefits)
    assert any("SulAmerica health plan" in b for b in benefits)
    assert any("Birthday day off" in b for b in benefits)
    # Benefits must never leak into requirements/skills.
    assert not any("SulAmerica" in v for v in out["extracted"]["required hard skill"])


def test_legacy_dict_to_jd_extraction_carries_benefits_through():
    jd_processor = _import_jd_processor_with_stubbed_models()
    legacy_result = {
        "required job title": [],
        "benefits": ["Health plan", "Birthday day off"],
    }
    result = jd_processor._legacy_dict_to_jd_extraction(legacy_result, routing_reason="test")
    assert set(result.benefits.value) == {"Health plan", "Birthday day off"}
    assert result.to_legacy_dict()["benefits"] == ["Birthday day off", "Health plan"]
