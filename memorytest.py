import sys
import numpy as np
from pympler.asizeof import asizeof


class Foo:
    def __init__(self):
        self.size = (1000, 1000)
        self.x = []

    def bar(self):
        self.x = np.random.uniform(-10, 10, self.size)
        return self.x


foo = Foo()

for k in range(100):
    print(asizeof(foo))
    y = foo.bar()

