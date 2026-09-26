# Local gradients stored as data at forward time.
# Backward is now a fully generic chain-rule engine

import math

class Value:
    """stores a single scalar value and its gradient"""

    def __init__(self, data, _children=(), _op='', _local_grads=()):
        self.data = data
        self.grad = 0
        self._prev = list(_children)      # list for deterministic ordering
        self._op = _op                    # kept for repr/debugging only
        self._local_grads = _local_grads  # ∂out/∂child_i per child, computed at forward time

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+', (1.0, 1.0))
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*', (other.data, self.data))
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}', (other * self.data**(other - 1),))
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU',
                    (1.0 if self.data > 0 else 0.0,))
        return out

    def sin(self):
        out = Value(math.sin(self.data), (self,), 'sin', (math.cos(self.data),))
        return out

    def exp(self):
        # d/dx e^x = e^x
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp', (e,))
        return out

    def log(self):
        # d/dx ln(x) = 1/x ; floor the argument so log(0) -> a finite value
        x = max(self.data, 1e-12)
        out = Value(math.log(x), (self,), 'log', (1.0 / x,))
        return out

    def sigmoid(self):
        # 1 / (1 + e^-x), computed in a numerically stable way; grad = s * (1 - s)
        if self.data >= 0:
            z = math.exp(-self.data)
            s = 1.0 / (1.0 + z)
        else:
            z = math.exp(self.data)
            s = z / (1.0 + z)
        out = Value(s, (self,), 'sigmoid', (s * (1.0 - s),))
        return out

    def backward(self):
        # children before parents in topological order
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # the entire backward pass, generic over all ops
        self.grad = 1.0
        for v in reversed(topo):
            for child, local in zip(v._prev, v._local_grads):
                child.grad += local * v.grad

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __repr__(self):
        # return f"Value(data={self.data}, grad={self.grad}, op={self._op}, prev={self._prev})"
        return f"Value(data={self.data}, grad={self.grad}, op={self._op})"
