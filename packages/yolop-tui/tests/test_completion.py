from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from yolop_tui.completion import TuiCompleter


def test_slash_completion_exposes_only_the_fixed_command_set(tmp_path) -> None:
    completer = TuiCompleter(tmp_path)

    completions = list(
        completer.get_completions(Document("/r"), CompleteEvent(completion_requested=True))
    )

    assert [completion.text for completion in completions] == ["/resume"]
