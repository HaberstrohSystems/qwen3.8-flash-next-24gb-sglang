"""Per-layer expert routing-mass histogram from the routing probe -> perf/expert_freq.pt

  python3 expert_freq.py [routing_dump]

Output: {"mass": FloatTensor[48, 512] (sum of gate weights), "count": LongTensor[48, 512]}.
Used by the frequency-based placement: the hottest experts of every layer go to the GPU.
"""
import glob, os, sys
import torch

src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "routing_dump")
mass = torch.zeros(48, 512); count = torch.zeros(48, 512, dtype=torch.long)
n = 0
for f in sorted(glob.glob(os.path.join(src, "routing_*.pt"))):
    for ids, w in torch.load(f):                       # ids [48,10] int16, w [48,10] f32
        ids = ids.long()
        for l in range(ids.shape[0]):
            mass[l].index_add_(0, ids[l], w[l])
            count[l].index_add_(0, ids[l], torch.ones(ids.shape[1], dtype=torch.long))
        n += 1
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expert_freq.pt")
torch.save({"mass": mass, "count": count, "tokens": n}, out)
top = mass.sort(dim=1, descending=True).values
for S in (128, 181, 256):
    print(f"  top {S:>3} per layer cover {100 * top[:, :S].sum() / mass.sum():5.1f} % of routing mass")
print(f"  {n} tokens -> {out}")
