class MyMeta(type):
    def __new__(cls, name, bases, dct):
        return super().__new__(cls, name, bases, dct)

    def __init__(cls, name, bases, dct):
        cls.TEMP = 0
        return super().__init__(name, bases, dct)


class MyMetaTR(MyMeta):
    def __new__(cls, name, bases, dct):
        return super().__new__(cls, name, bases, dct)

    def __init__(cls, name, bases, dct):
        cls.TEMP = 0
        return super().__init__(name, bases, dct)
    


class Base(metaclass=MyMetaTR):
    def __new__(cls):
        return super().__new__(cls)

    def __init__(self):
        super().__init__()

    def Check(self):
        self.__class__.TEMP += 1


class ConreteA(Base):
    def __new__(cls):
        return super().__new__(cls)


    def __init__(self):
        super().__init__()
        self.Check()


class ConreteB(Base):
    def __init__(self):
        super().__init__()
        self.Check()


concrete_a = ConreteA()
concrete_b = ConreteB()

concrete_a.TEMP += 5
print(concrete_b.TEMP)




# class Base():
#     def __new__(cls):
#         cls.TMP = 0
#         return super().__new__(cls)
    
#     def __init__(self) -> None:
#         ...

#     def Check(self):
#         self.__class__.TMP += 1
    

# class ConcreteA(Base):
#     def __init__(self) -> None:
#         super().__init__()
#         self.Check()


# class ConcreteB(Base):
#     def __init__(self) -> None:
#         super().__init__()
#         self.Check()


# a = ConcreteA()
# b = ConcreteB()

# a.TMP += 5

# print(a.TMP)