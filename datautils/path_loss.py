from dataclasses import dataclass, field


@dataclass(order=True)
class PathLoss:
    sort_index: int = field(init=False, repr=False)

    path: str
    loss: int
    label: str = ""

    def __post_init__(self):
        self.sort_index = self.loss