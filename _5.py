from abc import ABCMeta, abstractmethod


class IBase(metaclass=ABCMeta):
    def __init__(self):
        self.Name = "Base"

    @abstractmethod
    def ReadData(self):
        print(self.Name)



class ConcreteA(IBase):
    def __init__(self):
        super().__init__()
        self.Name = "ConcreteA"

    def ReadData(self):
        super().ReadData()

a = ConcreteA()
print(a.Name)