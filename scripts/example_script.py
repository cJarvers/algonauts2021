from pathlib import Path
import torch

rng_seed = 42

print("seeding with {}".format(rng_seed))

torch.manual_seed(rng_seed)
torch.cuda.manual_seed_all(rng_seed)

file_path = Path(__file__)

a = torch.rand(2,3).cuda()
b = torch.rand(3,2).cuda()
c = b.matmul(a).cpu()
out_path = str(file_path.parent / "tensor.pt")
print("saving to {}".format(out_path))
torch.save(c, out_path)
