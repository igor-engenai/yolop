from yolop_tui.suggestions import PromptCompleter


def test_prompt_completer_suggests_commands_and_project_files(tmp_path) -> None:
    source = tmp_path / "src" / "answer.py"
    source.parent.mkdir()
    source.write_text("ANSWER = 42\n")
    completer = PromptCompleter(tmp_path)

    commands = completer.complete("/r")
    auth_commands = completer.complete("/l")
    files = completer.complete("Explain @sra")

    assert [candidate.value for candidate in commands] == ["/resume"]
    assert [candidate.value for candidate in auth_commands] == ["/login", "/logout"]
    assert [candidate.value for candidate in files] == ["@src/answer.py"]
    assert files[0].start == len("Explain ")
