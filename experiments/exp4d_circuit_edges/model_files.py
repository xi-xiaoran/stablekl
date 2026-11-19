import hashlib, json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edge_scores2 as E
TEXT = 'When Mary and John went to the store, John gave a drink to'

def digest(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).numpy().tobytes() if t.dtype != torch.bfloat16 else t.reshape(-1).float().numpy().tobytes())
    return h.hexdigest()[:16]

def same_state(a, b):
    return a.keys() == b.keys() and all((a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and torch.equal(a[k], b[k]) for k in a))

def save_and_check(name, ref, ref16_state, d):
    torch.save(dict(cfg=ref.cfg, state=ref.state_dict()), os.path.join(d, name + '.pt'))
    ref.tokenizer.save_pretrained(os.path.join(d, 'tok_' + name))
    loc = E.load_local(name, d, torch.float32, torch.device('cpu'))
    toks = ref.to_tokens(TEXT)
    a = ref.state_dict()
    s32 = same_state(a, loc.state_dict())
    with torch.no_grad():
        so = bool(torch.equal(ref(toks), loc(toks)))
    st = bool(torch.equal(toks, loc.to_tokens(TEXT)))
    del loc
    s16 = same_state(ref16_state, E.load_local(name, d, torch.float16, torch.device('cpu')).state_dict())
    return (s32 and so and st and s16, dict(fp32=digest(a), fp16=digest(ref16_state)), f'state dict FP32 {s32}, logits {so}, tokens {st}, state dict FP16 {s16}')

def main():
    mode, d = (sys.argv[1], sys.argv[2])
    os.makedirs(d, exist_ok=True)
    if mode == 'online':
        import transformer_lens as tl
        out = {}
        for name in sys.argv[3:]:
            ref = tl.HookedTransformer.from_pretrained_no_processing(name, device='cpu', dtype=torch.float32)
            r16 = tl.HookedTransformer.from_pretrained_no_processing(name, device='cpu', dtype=torch.float16).state_dict()
            ok, dg, msg = save_and_check(name, ref, r16, d)
            out[name] = dg
            print(f"  {('ok  ' if ok else 'FAIL')} {name}: {ref.cfg.n_layers} layers, {ref.cfg.n_heads} heads, d_model {ref.cfg.d_model}, vocab {ref.cfg.d_vocab}, attn_only {ref.cfg.attn_only} | local copy == online model: {msg}", flush=True)
            if not ok:
                sys.exit(1)
            del ref, r16
        json.dump(out, open(os.path.join(d, 'digests.json'), 'w'), indent=1)
    else:
        ref = json.load(open(os.path.join(d, 'digests.json')))
        bad = 0
        for name in ref:
            for dn, dt in (('fp32', torch.float32), ('fp16', torch.float16)):
                g = digest(E.load_local(name, d, dt, torch.device('cpu')).state_dict())
                ok = g == ref[name][dn]
                bad += not ok
                print(f"  {('ok  ' if ok else 'FAIL')} offline, empty cache: {name} {dn} {g}", flush=True)
        sys.exit(1 if bad else 0)
if __name__ == '__main__':
    main()
