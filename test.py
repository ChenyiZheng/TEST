import torch
import numpy as np
# print(torch.cuda.is_available())
# reg_vals = []
# reg_vals.append(torch.einsum(
#                             'ij,ij->i', grad_H, R_grad_H).reshape(in_shape[:-1]))

# a = torch.tensor([[1, 2, 3, 4, 5],
#               [1, 2, 3, 4, 5]])
a = torch.tensor([[0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0]])
b = torch.tensor([[2, 2, 2, 2, 2],
              [3, 3, 3, 3, 3]])
step1 = a*b
print(step1)
out = torch.einsum('ij,ij->i', a, b)
print(out)