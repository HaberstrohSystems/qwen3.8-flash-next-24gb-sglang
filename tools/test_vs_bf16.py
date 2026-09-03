"""Isolate dequantization: identical weights, two paths.

reference : fused_moe with the DEQUANTIZED weights as bf16
candidate : the same weights packed as 2 bit + scales

Routing, activation and reduction are identical in both cases, so what remains
is exactly the path the patch touches.

Set SGLANG_SRC to your SGLang checkout if it is not at ~/quant/sglang.
"""
import os, torch, sys
sys.path.insert(0, os.path.join(
    os.environ.get("SGLANG_SRC", os.path.expanduser("~/quant/sglang")),
    "test", "manual"))
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
import test_triton_moe_wna16 as T
set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
dev="cuda"

def case(m,n,k,e,topk,gs,has_zp,dtype=torch.bfloat16,seed=0):
    torch.manual_seed(seed)
    a=torch.randn((m,k),device=dev,dtype=dtype)/10
    w1=torch.randn((e,2*n,k),device=dev,dtype=dtype)/10
    w2=torch.randn((e,k,n),device=dev,dtype=dtype)/10
    score=torch.randn((m,e),device=dev,dtype=dtype)
    pf=4; qt="w2a16" if has_zp else "w2a16b2"
    w1r,w2r=w1.clone(),w2.clone()
    w1q=torch.empty((e,2*n,k//pf),device=dev,dtype=torch.uint8)
    w2q=torch.empty((e,k,n//pf),device=dev,dtype=torch.uint8)
    w1s=torch.empty((e,2*n,k//gs),device=dev,dtype=dtype)
    w2s=torch.empty((e,k,n//gs),device=dev,dtype=dtype)
    w1z=torch.empty((e,2*n//pf,k//gs),device=dev,dtype=torch.uint8)
    w2z=torch.empty((e,k//pf,n//gs),device=dev,dtype=torch.uint8)
    for i in range(e*2):
        eid=i%e
        w,wr,wq,ws,wz=(w1,w1r,w1q,w1s,w1z) if i//e==0 else (w2,w2r,w2q,w2s,w2z)
        weight,qw,sc,qz=T.quantize_weights(w[eid].T,qt,gs,has_zp,False)
        weight=weight.T; qw=qw.T.contiguous().to(torch.uint8); sc=sc.T
        if has_zp: qz=qz.T.contiguous().to(torch.uint8)
        qw=qw[:,3::4]*64+qw[:,2::4]*16+qw[:,1::4]*4+qw[:,::4]
        if has_zp: qz=qz[3::4,:]*64+qz[2::4,:]*16+qz[1::4,:]*4+qz[::4,:]
        wr[eid]=weight; wq[eid]=qw; ws[eid]=sc
        if has_zp: wz[eid]=qz
    tk=select_experts(hidden_states=a,router_logits=score,topk_config=TopKConfig(top_k=topk))
    ref=fused_moe(a.clone(),w1r,w2r,tk)                                  # bf16-Weg
    got=fused_moe(a.clone(),w1q,w2q,tk,use_int2_w2a16=True,
                  w1_scale=w1s,w2_scale=w2s,
                  w1_zp=w1z if has_zp else None,w2_zp=w2z if has_zp else None,
                  block_shape=[0,gs])
    r,g=ref.float(),got.float()
    rel=(g-r).norm()/r.norm().clamp(min=1e-12)
    return rel.item(),(g-r).abs().max().item(),r.abs().max().item()

print("     m     n     k    e topk   gs  zp   rel.Fehler   maxAbw   |ref|max")
ok=True
for (m,n,k,e,topk,gs,hz) in [
    (1,128,128,8,2,128,False),(32,1024,128,8,6,128,False),
    (222,2048,1024,64,6,128,False),(32,1024,1024,8,2,64,False),
    (1,128,128,8,2,128,True),(32,1024,128,8,6,128,True),
    (222,2048,1024,64,6,128,True),(32,640,2560,64,6,128,False),   # echte Expertenmasse
]:
    rel,mx,rm=case(m,n,k,e,topk,gs,hz)
    good = rel < 2e-2
    ok &= good
    print(f"  {m:4d} {n:5d} {k:5d} {e:4d} {topk:4d} {gs:4d} {str(hz)[0]:>3}   "
          f"{rel:9.2e} {mx:8.4f} {rm:9.4f}   {'ok' if good else 'ABWEICHUNG'}")
print("\n  " + ("ALLE BESTANDEN" if ok else "FEHLER"))
