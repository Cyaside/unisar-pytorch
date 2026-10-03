"""Notebook syntax, manual gates and nonpersistent clone credentials."""
import ast
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def notebook():
    return json.loads((ROOT / "notebooks/train_teacher.ipynb").read_text(encoding="utf-8"))


def source(cell):
    value = cell["source"]
    return "".join(value) if isinstance(value, list) else value


def test_cells_syntax_outputs_and_manual_gates():
    book = notebook()
    assert book["nbformat"] == 4
    codes = [source(c) for c in book["cells"] if c["cell_type"] == "code"]
    for i, code in enumerate(codes):
        compile(code, f"notebook-cell-{i}", "exec")
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
    for gate in ("RUN_TEACHER_TRAINING", "RUN_BASELINE_TRAINING"):
        code = next(c for c in codes if f"{gate} =" in c)
        tree = ast.parse(code)
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == gate for t in n.targets))
        assert isinstance(assignment.value, ast.Constant) and assignment.value.value is False
        assert any(isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == gate for n in tree.body)
    assert not any("--split" in c and "test" in c for c in codes)


def test_clone_helper_keeps_token_out_of_argv_url_and_disk(tmp_path, monkeypatch):
    setup = next(source(c) for c in notebook()["cells"] if "def clone_private" in source(c))
    tree = ast.parse(setup)
    safe_nodes = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    namespace = {}
    exec(compile(ast.Module(body=safe_nodes, type_ignores=[]), "clone-helper", "exec"), namespace)
    token = "synthetic-secret-not-a-real-token"
    revision = "a" * 40
    captured = []
    def run(argv, **kwargs):
        assert token not in repr(argv)
        if "clone" in argv:
            environment = kwargs["env"]
            helper = Path(environment["GIT_ASKPASS"])
            script = helper.read_text()
            assert script.startswith("#!/bin/sh\n") and "\nesac\n" in script
            assert token not in script
            assert environment["UNISAR_GH_TOKEN"] == token
            captured.append(helper)
            Path(argv[-1]).mkdir()
        if "checkout" in argv:
            assert kwargs["env"]["GIT_LFS_SKIP_SMUDGE"] == "1"
        class Result:
            returncode = 0
        return Result()
    monkeypatch.setattr(namespace["subprocess"], "run", run)
    monkeypatch.setattr(namespace["subprocess"], "check_output", lambda argv, **k: "" if "status" in argv else revision + "\n")
    assert namespace["clone_private"]("owner/repository", tmp_path / "clone", revision, token) == revision
    assert captured and not captured[0].exists()
    assert "UNISAR_GH_TOKEN" not in namespace["os"].environ


def test_delivered_notebook_pins_immutable_source():
    # Enabled once the bootstrap source commit exists; no floating branch clone.
    setup = next(source(c) for c in notebook()["cells"] if "PACKAGE_REVISION =" in source(c))
    value = re.search(r'PACKAGE_REVISION = "([^"]+)"', setup).group(1)
    assert re.fullmatch(r"[0-9a-f]{40}", value)
