"""Guards on the three teaching notebooks: their sources, their wiring and their baked tables.

Every check reads the notebook JSON; nothing is executed. Source checks read cell sources
only, never outputs, so a stale output from an earlier bake cannot hide or fake a change.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "01_pretrained_ai": "01_pretrained_ai",
    "02_unet_chemical_maps": "02_unet_chemical_maps",
    "03_spectral_vs_spatial": "03_spectral_vs_spatial",
}
REMOVED_NAMES = ("SpatialMean", "kiwi_sam3_spectral_match")

# The course's own numbers, literals in each notebook's reproduction table: they never move.
COURSE_VALUES = {
    "01_pretrained_ai": {
        "kiwi masks after the size filter": 30,
        "kiwi masks after the spectral filter": 30,
        "YOLO labels, yolov8n (the course's model)": "2 oven; 11 carrot, 1 person",
        "YOLO labels, YOLO26 (the plugin's default)": "2 oven; 11 carrot, 1 person",
        "PLS spatial, latent variables": 5,
        "PLS spatial RMSE": 0.447,
        "PLS spatial R2": 0.61,
        "PLS spectral, latent variables": 17,
        "PLS spectral RMSE": 0.321,
        "PLS spectral R2": 0.799,
        "PLS stacked RMSE": 0.402,
        "PLS stacked R2": 0.685,
        "average fusion RMSE": 0.333,
        "average fusion R2": 0.783,
    },
    "02_unet_chemical_maps": {
        "U-Net belly 10 (%)": 29.85,
        "U-Net belly 70 (%)": 62.25,
        "U-Net map, share outside 0 to 100 (%)": 0.0,
        "U-Net map, mean 1 - C0/var": 0.9983,
        "PLS components": 18,
        "PLS CV RMSE": 5.48,
        "PLS map of the first slice, share outside 0 to 100 (%)": 87.76,
        "PLS map of the first slice, 1 - C0/var": 0.0211,
    },
    "03_spectral_vs_spatial": {
        "standardisation mean": 0.031223,
        "standardisation std": 0.04972,
        "PCA explained variance, component 1": 0.9126,
        "PCA explained variance, component 2": 0.0567,
        "PCA explained variance, component 3": 0.0245,
        "PLS components": 6,
        "PLS validation RMSE": 4.234,
        "PLS test RMSE": 5.904,
        "PLS test R2": 0.799,
        "MLP test RMSE": 5.544,
        "MLP test R2": 0.823,
        "1D ResNet test RMSE": 6.222,
        "1D ResNet test R2": 0.776,
        "own MLP trainrun, best validation MSE": 29.04,
        "own 1D ResNet trainrun, best validation MSE": "34.30 / 78.58",
    },
}
# Rows allowed to read "open deviation". The 1D ResNet trainrun row is reported, not judged:
# the course's two published runs printed 34.30 and 78.58.
ALLOWED_OPEN: set[tuple[str, str]] = set()
# The course commit every notebook, the README and the vendored course code name.
COURSE_COMMIT = "c65ba36"


def _cells(stem: str) -> list[dict]:
    path = REPO / "notebooks" / f"{stem}.ipynb"
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def _code(cell: dict) -> str:
    """A code cell's source with IPython lines (``!``, ``%``) dropped so ``ast`` can parse it."""
    lines = "".join(cell["source"]).splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith(("!", "%")))


def _code_cells(stem: str) -> list[str]:
    return [_code(cell) for cell in _cells(stem) if cell["cell_type"] == "code"]


def _constructed_nodes(stem: str) -> set[tuple[str, str]]:
    """Every ``Class(name="...")`` call in the notebook's code cells, as (name, class)."""
    found = set()
    for source in _code_cells(stem):
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            cls = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            for keyword in node.keywords:
                if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                    found.add((keyword.value.value, cls))
    return found


def _built_pipeline_names(stem: str) -> set[str]:
    names = set()
    for source in _code_cells(stem):
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "CuvisPipeline"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                names.add(node.args[0].value)
    return names


@pytest.mark.parametrize("stem", NOTEBOOKS)
def test_every_pipeline_cell_draws_its_graph_with_the_node_names(stem: str) -> None:
    for source in _code_cells(stem):
        if "CuvisPipeline(" not in source:
            continue
        last = [line for line in source.splitlines() if line.strip()][-1].strip()
        assert last.startswith("utils.show_pipeline("), (
            f"{stem}: a cell that builds a pipeline ends with {last!r}, not utils.show_pipeline(...)"
        )


@pytest.mark.parametrize("stem", NOTEBOOKS)
def test_no_notebook_source_names_a_removed_node_or_pipeline(stem: str) -> None:
    sources = "\n".join("".join(cell["source"]) for cell in _cells(stem))
    for name in REMOVED_NAMES:
        assert name not in sources, f"{stem} still names {name}"


def test_no_pipeline_yaml_or_generator_names_a_removed_node_or_pipeline() -> None:
    texts = [p.read_text(encoding="utf-8") for p in (REPO / "pipelines").rglob("*.yaml")]
    texts += [(REPO / "tools" / "build_pipelines.py").read_text(encoding="utf-8")]
    for text in texts:
        for name in REMOVED_NAMES:
            assert name not in text


@pytest.mark.parametrize("stem", NOTEBOOKS)
def test_the_course_values_in_the_reproduction_table_never_move(stem: str) -> None:
    rows = {}
    tree = ast.parse(_code_cells(stem)[-1])
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "rows" for t in node.targets
        ):
            for element in node.value.elts:
                rows[ast.literal_eval(element.elts[0])] = ast.literal_eval(element.elts[1])
    assert rows == COURSE_VALUES[stem]


@pytest.mark.parametrize("stem", NOTEBOOKS)
def test_the_notebook_builds_the_nodes_of_its_pipeline_yamls(stem: str) -> None:
    """The cells and tools/build_pipelines.py carry the same wiring; names and classes agree."""
    constructed = _constructed_nodes(stem)
    built = _built_pipeline_names(stem)
    folder = REPO / "pipelines" / NOTEBOOKS[stem]
    compared = 0
    for path in sorted(folder.glob("*.yaml")):
        if path.stem not in built:
            continue
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        for node in doc["nodes"]:
            pair = (node["name"], node["class_name"].rsplit(".", 1)[1])
            assert pair in constructed, f"{stem}: {path.name} has {pair}, the notebook does not"
        compared += 1
    assert compared, f"{stem}: no pipeline YAML matches a CuvisPipeline the notebook builds"


@pytest.mark.parametrize("stem", NOTEBOOKS)
def test_the_baked_reproduction_table_has_no_unexpected_open_deviation(stem: str) -> None:
    outputs = _cells(stem)[-1].get("outputs", [])
    html = "".join("".join(output.get("data", {}).get("text/html", "")) for output in outputs)
    rows = re.findall(
        r"<tr>\s*<th>\d+</th>\s*<td>(.*?)</td>.*?<td>([^<]*)</td>\s*</tr>", html, re.S
    )
    assert rows, f"{stem}: no baked reproduction table found"
    for name, status in rows:
        if status.strip() == "open deviation":
            assert (stem, name.strip()) in ALLOWED_OPEN, (
                f"{stem}: {name.strip()} is an open deviation"
            )


@pytest.mark.parametrize("stem", NOTEBOOKS)
def test_the_notebook_names_the_course_commit_it_reproduces(stem: str) -> None:
    intro = "".join(_cells(stem)[0]["source"])
    assert f"`{COURSE_COMMIT}`" in intro


def test_the_readme_and_the_vendored_course_code_name_the_same_course_commit() -> None:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    vendored = (REPO / "cuvis_ai_wur_course" / "upstream" / "course_models.py").read_text(
        encoding="utf-8"
    )
    assert f"`{COURSE_COMMIT}`" in readme
    assert f"``{COURSE_COMMIT}``" in vendored


def test_the_own_resnet_trainrun_row_is_reported_not_judged() -> None:
    outputs = _cells("03_spectral_vs_spatial")[-1].get("outputs", [])
    html = "".join("".join(output.get("data", {}).get("text/html", "")) for output in outputs)
    row = re.search(r"own 1D ResNet trainrun, best validation MSE</td>(.*?)</tr>", html, re.S)
    assert row, "the baked table has no own 1D ResNet trainrun row"
    assert re.findall(r"<td>([^<]*)</td>", row.group(1))[-1].strip() == "reported"
