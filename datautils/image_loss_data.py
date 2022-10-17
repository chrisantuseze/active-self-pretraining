from dataclasses import dataclass, field


@dataclass(order=True)
class Image_Loss:
    sort_index: int = field(init=False, repr=False)

    image1: str
    image2: str
    loss: int

    def __post_init__(self):
        self.sort_index = self.loss


# members = [
#     Person(name='John', age=25),
#     Person(name='Bob', age=35),
#     Person(name='Alice', age=30)
# ]

# sorted_members = sorted(members)
# for member in sorted_members:
#     print(f'{member.name}(age={member.age})')