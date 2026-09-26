from engine import Value

x1 = Value(2.0, (), 'x1')
x2 = Value(3.0, (), 'x2')
x3 = Value(4.0, (), 'x3')

f1 = x1 + x2
f2 = x3 * f1
Loss = f2.sin()

Loss.backward()

print(f"x1 = {x1}")
print(f"x2 = {x2}")
print(f"x3 = {x3}")
print(f"f1 = {f1}")
print(f"f2 = {f2}")
print(f"Loss = {Loss}")
