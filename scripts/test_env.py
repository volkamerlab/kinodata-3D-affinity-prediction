import torch
import torch_geometric
import wandb


print(f"torch version: {torch.__version__}")
print(f"torch_geometric version: {torch_geometric.__version__}")
print(f"cuda version: {torch.version.cuda}")
print(f"cuda available: {torch.cuda.is_available()}")

print("Simple CUDA test")
x = torch.randn(42).cuda()
x.requires_grad_(True)
y = torch.rand(1).cuda()
pred = x.mean() * x.std()
loss = (pred - y)
loss.backward()
print(x.grad.data)


print("Torch cluster test")
from torch_cluster import knn_graph, radius_graph

x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]).cuda()
batch = torch.tensor([0, 0, 0, 0]).cuda()
edge_index = knn_graph(x, k=2, batch=batch, loop=False)
edge_index = radius_graph(x, r=2.5, batch=batch, loop=False)
