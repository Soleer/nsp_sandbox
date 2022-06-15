from torch import classes


class NSPbuilder:
    print_info: bool

    def __init__(self, print_info=True) -> None:
        self.print_info = print_info
    