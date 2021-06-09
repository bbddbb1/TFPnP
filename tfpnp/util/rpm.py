import random

class ReplayMemory:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0
        
    def store(self, obj):
        if self.size() > self.buffer_size:
            print('buffer size larger than set value, trimming...')
            self.buffer = self.buffer[(self.size() - self.buffer_size):]
        elif self.size() == self.buffer_size:
            self.buffer[self.index] = obj
            self.index += 1
            self.index %= self.buffer_size
        else:
            self.buffer.append(obj)

    def size(self):
        return len(self.buffer)

    def sample_batch(self, env_batch):
        if self.size() < env_batch:
            batch = random.sample(self.buffer, self.size())
        else:
            batch = random.sample(self.buffer, env_batch)
            
        return batch
