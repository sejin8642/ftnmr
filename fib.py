# Simple Fibonacci generator
class fib():
    def __init__(self, n=100):
        self.a = 0
        self.b = 1
        self.c = True
        self.n = n
        self.output = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.n < 1:
            raise StopIteration
        elif self.c == True:
            self.c = False
            self.output = self.a
            self.a = self.a + self.b
        else:
            self.c = True
            self.output = self.b
            self.b = self.a + self.b
        
        self.n -= 1
        return self.output

for i in fib(101):
    print(i)
