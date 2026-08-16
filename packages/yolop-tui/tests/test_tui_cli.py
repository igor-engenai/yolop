import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from yolop_tui.cli import main


def test_cli_starts_bundled_chat_with_project_sqlite_default(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with create_pipe_input() as pipe_input:
        pipe_input.send_text("/quit\r")
        with create_app_session(input=pipe_input, output=DummyOutput()):
            main([])

    assert (tmp_path / ".yolop" / "runtime.db").is_file()


def test_external_agentspec_fully_replaces_the_bundled_default(tmp_path) -> None:
    spec_path = tmp_path / "agent.yaml"
    spec_path.write_text("name: custom\ninstructions: Custom only.\n")

    with pytest.raises(SystemExit, match="must contain a string model"):
        main(["--agent-spec", str(spec_path)])
