from typing import NamedTuple


class TrackableIteratorState(NamedTuple):
	Id: int
	Index: int
	Available: int
	Expired: bool


class TrackableGeoIterator():
	"""GeoSlider veya Random GeoIterator için indis takibi yapan bir sınıftır."""
	def __init__(self, iterator, id, cycle=False, margin=None):
		self.Id = id
		self.Index = -1
		self._Expired = False
		self._Cycle = cycle
		self.Iterator = iterator
		self.margin = margin


	@property
	def Margin(self):
		if 0 != self.__len__() % self.margin:
			return self.margin - (self.__len__() % self.margin)
		else:
			return 0
	

	@property
	def Available(self) -> int:
		if self.Index == -1:
			return self.__len__()
		
		return self.__len__() - ((1 + (max(self.Index, 0) % self.__len__())))


	def __iter__(self):
		return self
	

	def __next__(self):
		"""For Random GeoDataset"""
		return next(iter(self.Iterator))


	def __len__(self):
		return len(self.Iterator)


	def __getitem__(self, id):
		"""For Sliding GeoDataset"""
		self.Index += 1
		self.CheckIndex()
		# print(f"Trackable Iterator:-> Patch Index: {self.Index}-%-{self.Index % len(self.Iterator)}")
		return self.Iterator[self.Index % self.__len__()]


	def CheckIndex(self):
		if (self._Expired and not self._Cycle):
			raise IndexError
		
		self._Expired = self.Index >= self.__len__() + self.Margin - 1    # TODO Eğer alınamayacak kadar window varsa drop_last uygula veya 1 kerelik cycle yap


	def GetIndex(self):
		return self.Index


	def IsExpired(self):
		return self._Expired


	def Reset(self):
		self.Index = -1
		self._Expired = False


	def GetState(self):
		state = TrackableIteratorState(Id=self.Id, Index=self.Index, Available=self.Available, Expired=self._Expired)
		return state


	def SetState(self, state: TrackableIteratorState):
		self.Index, self._Expired = state.Index, state.Expired


