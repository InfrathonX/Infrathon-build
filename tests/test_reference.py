import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_reference", ROOT / "reference/make_reference.py")
make_reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_reference)


def test_drop_absent_rows_keeps_prose_and_present_tasks():
    text = "Intro naming gone-task.\n| a | kept-task |\n| b | gone-task |\n"
    out, removed = make_reference.drop_absent_rows(text, {"gone-task"})
    assert out == "Intro naming gone-task.\n| a | kept-task |\n"
    assert removed == 1


def test_task_list_test_keeps_its_subset_safe_half():
    text = '''def test_release_task_list():
    tasks = load()
    assert set(document["tasks"]) == current
    assert tasks.table(document) in readme
    gkd = "some-task"
    assert tasks.check(document, gkd, "hw", allow_untested=False) is None


def test_next():
    assert True
'''
    out, dropped = make_reference.keep_subset_safe_asserts(text)
    assert dropped == 2
    assert "gkd" not in out
    assert 'assert tasks.table(document) in readme\n\n\ndef test_next():' in out
    compile(out, "t.py", "exec")


def test_task_list_test_at_the_end_of_the_file():
    text = "def t():\n    assert tasks.table(document) in readme\n    assert other\n"
    out, dropped = make_reference.keep_subset_safe_asserts(text)
    assert out == "def t():\n    assert tasks.table(document) in readme\n"
    assert dropped == 1
