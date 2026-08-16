from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionOption:
    value: str
    label: str
