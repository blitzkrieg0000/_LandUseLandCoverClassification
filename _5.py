from typing import NamedTuple



class TrackableIteratorState(NamedTuple):
	Id: int
	Index: int
	Available: int
	Expired: bool
	

state = TrackableIteratorState(Id=0, Index=0, Available=4, Expired=False)
tp = tuple(state)
x = TrackableIteratorState(*tp)
print(x)