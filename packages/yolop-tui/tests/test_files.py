import subprocess
from pathlib import Path

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from pydantic_ai.messages import TextContent
from yolop_tui.files import FileReferenceCompleter, FileReferenceError, prepare_prompt


def test_file_reference_adds_project_text_to_native_prompt(tmp_path: Path) -> None:
    source = tmp_path / "src" / "answer.py"
    source.parent.mkdir()
    source.write_text("ANSWER = 42\n")

    prompt = prepare_prompt("Explain @src/answer.py", cwd=tmp_path)

    assert isinstance(prompt, list)
    assert prompt[0] == "Explain @src/answer.py"
    assert isinstance(prompt[1], TextContent)
    assert prompt[1].content == ('<yolop-file path="src/answer.py">\nANSWER = 42\n\n</yolop-file>')


def test_normal_prompt_quotes_and_backticks_are_not_shell_parsed(tmp_path: Path) -> None:
    text = "Explain `value` because it doesn't close \"either"

    assert prepare_prompt(text, cwd=tmp_path) == text


def test_incomplete_quoted_file_reference_has_a_specific_error(tmp_path: Path) -> None:
    with pytest.raises(FileReferenceError, match="no closing double quote"):
        prepare_prompt('Read @"project notes.md', cwd=tmp_path)


def test_file_reference_rejects_project_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("secret")

    with pytest.raises(FileReferenceError, match="escapes the project"):
        prepare_prompt("Read @../secret.txt", cwd=tmp_path)


def test_missing_file_reference_has_a_safe_error(tmp_path: Path) -> None:
    with pytest.raises(FileReferenceError, match="does not exist"):
        prepare_prompt("Read @missing.txt", cwd=tmp_path)


def test_file_reference_completion_fuzzy_matches_project_paths(tmp_path: Path) -> None:
    source = tmp_path / "src" / "answer.py"
    source.parent.mkdir()
    source.write_text("ANSWER = 42\n")
    completer = FileReferenceCompleter(tmp_path)

    completions = list(
        completer.get_completions(
            Document("Explain @sra"), CompleteEvent(completion_requested=True)
        )
    )

    assert [completion.text for completion in completions] == ["@src/answer.py"]


def test_quoted_file_reference_supports_spaces(tmp_path: Path) -> None:
    document = tmp_path / "project notes.md"
    document.write_text("Keep this context.\n")

    prompt = prepare_prompt('Use @"project notes.md"', cwd=tmp_path)

    assert isinstance(prompt, list)
    attachment = prompt[1]
    assert isinstance(attachment, TextContent)
    assert 'path="project notes.md"' in attachment.content


def test_file_reference_rejects_a_file_over_256_kib(tmp_path: Path) -> None:
    large = tmp_path / "large.txt"
    large.write_bytes(b"x" * (256 * 1024 + 1))

    with pytest.raises(FileReferenceError, match="exceeds 256 KiB"):
        prepare_prompt("Read @large.txt", cwd=tmp_path)


def test_file_reference_rejects_a_symlink_outside_the_project(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")
    (tmp_path / "linked.txt").symlink_to(outside)

    with pytest.raises(FileReferenceError, match="escapes the project"):
        prepare_prompt("Read @linked.txt", cwd=tmp_path)


def test_file_reference_rejects_non_utf8_content(tmp_path: Path) -> None:
    (tmp_path / "binary.dat").write_bytes(b"\xff\xfe")

    with pytest.raises(FileReferenceError, match="not UTF-8 text"):
        prepare_prompt("Read @binary.dat", cwd=tmp_path)


def test_file_reference_rejects_utf8_with_null_bytes(tmp_path: Path) -> None:
    (tmp_path / "binary.txt").write_bytes(b"text\0content")

    with pytest.raises(FileReferenceError, match="is binary"):
        prepare_prompt("Read @binary.txt", cwd=tmp_path)


def test_multiple_file_references_preserve_prompt_order(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("one")
    (tmp_path / "two.txt").write_text("two")

    prompt = prepare_prompt("Compare @two.txt with @one.txt", cwd=tmp_path)

    assert isinstance(prompt, list)
    first, second = prompt[1:]
    assert isinstance(first, TextContent)
    assert isinstance(second, TextContent)
    assert 'path="two.txt"' in first.content
    assert 'path="one.txt"' in second.content


def test_multiple_file_references_have_a_one_mib_total_limit(tmp_path: Path) -> None:
    names = []
    for index in range(5):
        name = f"part-{index}.txt"
        names.append(name)
        (tmp_path / name).write_bytes(b"x" * (220 * 1024))

    with pytest.raises(FileReferenceError, match="1 MiB total limit"):
        prepare_prompt("Review " + " ".join(f"@{name}" for name in names), cwd=tmp_path)


def test_file_completion_uses_git_ignored_file_policy(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("ignored.txt\n")
    (tmp_path / "visible.txt").write_text("visible")
    (tmp_path / "ignored.txt").write_text("ignored")
    completer = FileReferenceCompleter(tmp_path)

    completions = list(
        completer.get_completions(Document("@"), CompleteEvent(completion_requested=True))
    )

    assert "@visible.txt" in [completion.text for completion in completions]
    assert "@ignored.txt" not in [completion.text for completion in completions]
