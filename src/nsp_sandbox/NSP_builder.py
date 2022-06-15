from tomlkit import integer, string


class NSPbuilder:
    print_info: bool
    version: str
    n_nurses: int
    n_jobs: int
    n_days: int
    
    def __init__(self, print_info=True) -> None:
        self.print_info = print_info
        self.version = "0.0.1"
    
    def set_n_nurses(self, n:int) -> None:
        self.n_nurses = n
    
    def set_n_jobs(self, n:int) -> None:
        self.n_jobs = n
    
    def set_n_days(self, n:int) -> None:
        self.n_days = n