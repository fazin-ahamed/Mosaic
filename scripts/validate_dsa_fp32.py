import sys, types, torch, json
torch.set_default_dtype(torch.float32)
torch.set_default_device("cuda")
sys.path.insert(0, "."); sys.path.insert(0, ".mosaicvm/source")
import mosaicvm.model.dsa as mdsa
fht = types.ModuleType("fast_hadamard_transform")
fht.hadamard_transform = lambda x, scale=None: mdsa.hadamard(x)
sys.modules["fast_hadamard_transform"] = fht
import inference_kernel
sys.modules["kernel"] = inference_kernel
import inference_model as SM
SM.rotate_activation = lambda x: mdsa.hadamard(x)
def aq(x, *a, **k): return x
SM.act_quant = aq; SM.fp4_act_quant = aq
def sp(q, kv, sink, idxs, scale):
    b, s, h, d = q.size()
    valid = idxs >= 0; idxc = idxs.clamp(min=0)
    g = kv.unsqueeze(1).expand(b, s, *kv.shape[1:]).gather(2, idxc.unsqueeze(-1).expand(-1,-1,-1,d))
    sc_ = torch.einsum("bshd,bstd->bsht", q, g) * scale
    sc_ = torch.where(valid.unsqueeze(2), sc_, torch.full_like(sc_, float("-inf")))
    m = sc_.max(-1, keepdim=True).values
    p = torch.exp(sc_ - m)
    den = p.sum(-1) + torch.exp(sink.view(1,1,h).float() - m.squeeze(-1))
    return torch.einsum("bsht,bstd->bshd", p, g) / den.unsqueeze(-1)
SM.sparse_attn = sp
SM.default_dtype = torch.float32
src_cfg = json.load(open(".mosaicvm/source/inference_config.json"))
args = SM.ModelArgs(); args.__dict__.update({
    "dim": src_cfg["dim"], "n_heads": src_cfg["n_heads"], "head_dim": src_cfg["head_dim"],
    "rope_head_dim": src_cfg["rope_head_dim"], "q_lora_rank": src_cfg["q_lora_rank"],
    "o_groups": src_cfg["o_groups"], "o_lora_rank": src_cfg["o_lora_rank"],
    "window_size": src_cfg["window_size"], "original_seq_len": src_cfg["original_seq_len"],
    "rope_theta": src_cfg["rope_theta"], "rope_factor": src_cfg["rope_factor"],
    "beta_fast": src_cfg["beta_fast"], "beta_slow": src_cfg["beta_slow"],
    "index_n_heads": src_cfg["index_n_heads"], "index_head_dim": src_cfg["index_head_dim"],
    "index_topk": src_cfg["index_topk"], "norm_eps": 1e-6, "max_batch_size": 1,
    "max_seq_len": 1024, "compress_rope_theta": src_cfg["compress_rope_theta"],
    "compress_ratios": src_cfg["compress_ratios"]})
import mosaicvm.model.dsa as D
import mosaicvm.model.dsa_layers as DL
D.act_quant_sim = lambda x, block=64: x
D.fp4_act_quant_sim = lambda x, block=32: x
DL.act_quant_sim, DL.fp4_act_quant_sim = D.act_quant_sim, D.fp4_act_quant_sim
from mosaicvm.model.config import V4Config
from mosaicvm.model.dsa_layers import ExactAttention
cfg = V4Config()
for LAYER, S in [(0, 256), (2, 256), (3, 256)]:
    torch.manual_seed(7)
    a = SM.Attention(LAYER, args).cuda()
    a.wo_a.float()
    if getattr(a, "indexer", None) is not None:
        a.indexer.weights_proj.float()
    o = ExactAttention(cfg, layer_id=LAYER, max_seq=1024, device="cuda").float()
    o.rebuild_freqs()
    if o.compressor is not None:
        o.compressor.rebuild_freqs()
        if o.indexer is not None:
            o.indexer.compressor.rebuild_freqs()
    has_comp = getattr(a, "compressor", None) is not None
    with torch.no_grad():
        def gen(*sh): return torch.randn(*sh, device="cuda") * 0.02
        for s_nm, o_nm, inn, out in [("wq_a","q_a",args.dim,args.q_lora_rank),
                                     ("wq_b","q_b",args.q_lora_rank,64*512),
                                     ("wkv","kv_proj",args.dim,512),
                                     ("wo_b","wo_b",8*1024,args.dim),
                                     ("wo_a","wo_a",4096,8*1024)]:
            w = gen(inn, out).T.contiguous()
            getattr(a, s_nm).weight.copy_(w); getattr(o, o_nm).weight.copy_(w)
        for nm in ("q_norm", "kv_norm"):
            v = torch.rand(getattr(a, nm).weight.shape, device="cuda") + 0.5
            getattr(a, nm).weight.copy_(v); getattr(o, nm).weight.copy_(v)
        sink = torch.randn(64, device="cuda")
        a.attn_sink.copy_(sink); o.attn_sink.data.copy_(sink)
        def lc(sc_, oc_):
            ape = torch.randn(sc_.ape.shape, device="cuda") * 0.1
            sc_.ape.copy_(ape); oc_.ape.copy_(ape)
            for nm in ("wkv", "wgate"):
                w = torch.randn(getattr(sc_, nm).weight.shape, device="cuda") * 0.02
                getattr(sc_, nm).weight.copy_(w); getattr(oc_, nm).weight.copy_(w)
            v = torch.rand(sc_.norm.weight.shape, device="cuda") + 0.5
            sc_.norm.weight.copy_(v); oc_.norm.weight.copy_(v)
        if has_comp:
            lc(a.compressor, o.compressor)
            if getattr(a, "indexer", None) is not None and o.indexer is not None:
                ix, oix = a.indexer, o.indexer
                wq_ix = gen(args.q_lora_rank, 64*128).T.contiguous()
                ix.wq_b.weight.copy_(wq_ix); oix.wq_b.weight.copy_(wq_ix)
                wp = gen(args.dim, 64).T.contiguous()
                ix.weights_proj.weight.copy_(wp); oix.weights_proj.weight.copy_(wp)
                lc(ix.compressor, oix.compressor)
    x = torch.randn(1, S, args.dim, device="cuda") * 0.5
    with torch.no_grad():
        ys = a(x, 0).float()
        yo = o(x.squeeze(0)).float().unsqueeze(0)
    d = (ys - yo).abs(); rel = d / yo.abs().clamp_min(1e-3)
    print(f"L{LAYER} (ratio {cfg.compress_ratios[LAYER]}) fp32 s={S}: "
          f"max_abs={d.max().item():.2e} max_rel={rel.max().item():.2e} "
          f"rtol1e-4={(rel<=1e-4).float().mean().item():.4f} "
          f"rtol1e-2={(rel<=1e-2).float().mean().item():.4f}")
