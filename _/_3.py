class MyMeta(type):
    def __new__(cls, name, bases, dct, kwarg1=1):
        print(f"Creating class {name}")
        return super().__new__(cls, name, bases, dct)

    def __init__(cls, name, bases, dct,  kwarg1=1):
        print(f"Initializing class {name}")
        super().__init__(name, bases, dct)


class MyMetaTR(MyMeta):
    def __new__(cls, name, bases, dct, kwarg1=1):
        print(f"Yaratılan sınıf: {name}")
        return super().__new__(cls, name, bases, dct)

    def __init__(cls, name, bases, dct,  kwarg1=1):
        print(f"Başlatılan Sınıf {name}")
        super().__init__(name, bases, dct)
        print(Base.__name__)


class Base(metaclass=MyMeta, kwarg1="BasE"):
    def __init__(self):
        print(Base.__name__)


class Derived(Base, metaclass=MyMetaTR):
    def __len__(self):
        return 100


class ConreteA(Derived):
    pass


print(len(ConreteA()))