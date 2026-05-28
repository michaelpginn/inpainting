from typing import TypedDict


class DialogTurn(TypedDict):
    question: str
    response: str


class Dialog(TypedDict):
    id: str
    turns: list[DialogTurn]
