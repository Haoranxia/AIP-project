import numpy as np
import torch

grid_size = 3
grid = torch.arange(grid_size)
a,b = np.meshgrid(grid, grid)

x_offset = torch.FloatTensor(a).view(-1,1)
y_offset = torch.FloatTensor(b).view(-1,1)


x_y_offset = torch.cat((x_offset, y_offset), 1)
print(x_y_offset)
# .repeat(1,num_anchors).view(-1,2).unsqueeze(0)

test_data = [[b, a] for a in grid for b in grid]
print("grid size: ", len(grid))
print("test data size: ", len(test_data))
print(test_data)

test_data_tensor = torch.FloatTensor(test_data)
print(test_data_tensor)

print("repeat")
test_data_tensor = test_data_tensor.repeat(1, 3)
print(test_data_tensor)
print("repeat len: ", len(test_data_tensor))

print("view")
test_data_tensor = test_data_tensor.view(-1,2)
print(test_data_tensor)
print("view len: ", len(test_data_tensor))

print("unsqueeze")
test_data_tensor = test_data_tensor.unsqueeze(0)
print(test_data_tensor)
print("unsqueeze len: ", test_data_tensor)

print("anchors")
anchors = torch.FloatTensor([(1, 2), (4, 5)])
print(anchors)

print("repeat")
anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
print(anchors)

print("0 test")
tens = torch.ones(5)
print("tens: ", tens * 0)

print(7.00 == 7)
