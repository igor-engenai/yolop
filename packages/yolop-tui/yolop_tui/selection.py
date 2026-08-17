from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionOption:
    value: str
    label: str


@dataclass(frozen=True)
class HistoryOption:
    value: str
    label: str
    selected: bool = False
