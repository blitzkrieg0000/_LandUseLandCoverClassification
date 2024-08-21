from multiprocessing import Manager

class SharedQueueSet():
    def __init__(self):
        self.manager = Manager()
        self.queue = self.manager.Queue()
        self.items_set = self.manager.dict()  # Thread-safe dict to keep track of unique items

    def Add(self, item):
        """Add an item to the set-like queue if it's not already present."""
        if item not in self.items_set:
            self.queue.put(item)
            self.items_set[item] = True

    def Get(self):
        """Get an item from the queue."""
        item = self.queue.get()
        self.items_set.pop(item, None)
        return item

    def __contains__(self, item):
        """Check if the item is in the set-like queue."""
        return item in self.items_set

    def __len__(self):
        """Return the number of unique items in the queue."""
        return len(self.items_set)

    def Empty(self):
        """Check if the queue is empty."""
        return self.queue.empty()

    def Clear(self):
        """Clear the queue and set."""
        while not self.queue.empty():
            self.queue.get_nowait()
        self.items_set.clear()



# Usage example:
queue_set = QueueSet()

# Adding items
queue_set.Add('apple')
queue_set.Add('banana')
queue_set.Add('apple')  # This won't be added again

# Checking if an item is in the queue
print('apple' in queue_set)  # True
print('orange' in queue_set)  # False

# Getting items
print(queue_set.Get())  # 'apple'
print(queue_set.Get())  # 'banana'

# Checking length
print(len(queue_set))  # 0, since both items were retrieved
