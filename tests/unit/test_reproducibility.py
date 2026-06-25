from pathlib import Path
from tempfile import TemporaryDirectory

import analysis
from analysis.export import anonymize
from analysis.render import __main__ as build
from analysis.render import checksums, site, sources
from analysis.render.derive import IncompleteDataError

ANALYSIS = Path(analysis.__file__).resolve().parent
OFFLINE = [ANALYSIS / "render", ANALYSIS / "datasets"]


def test_datasets_match_schema():
    sources.baselines()
    sources.sweeps()


def test_build_does_not_import_online_deps():
    for root in OFFLINE:
        for path in root.rglob("*.py"):
            assert "import wandb" not in path.read_text(), path


def test_build_matches_committed_checksums():
    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        try:
            build.build(out)
        except IncompleteDataError:
            pass
        checksums.verify(out)


def test_site_assets_match_frozen_datasets():
    df = sources.baselines()
    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        data = site.write_data(df, out / "data.js")
        assert data.read_bytes() == (site.WEBSITE / "js" / "data.js").read_bytes()
        for figure in site.render_figures(df, out / "figures"):
            committed = site.WEBSITE / "assets" / "figures" / figure.name
            assert figure.read_bytes() == committed.read_bytes(), figure.name


def test_anonymize_drops_identity():
    lock = {
        "metrics": {},
        "baselines": [{"id": "x", "algo": "a", "env": "e", "seed": 0, "group": "g"}],
        "sweeps": [],
    }
    blind = anonymize.blind(lock)
    assert blind["baselines"] == [{"algo": "a", "env": "e", "seed": 0}]
    assert "project" not in blind
