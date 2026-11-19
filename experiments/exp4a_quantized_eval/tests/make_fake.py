import json, os, sys, shutil, random, torch
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..'))
from dequant import pack_awq, pack_gptq
from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
root = sys.argv[1]
shutil.rmtree(root, ignore_errors=True)
torch.manual_seed(0)
words = [f'w{i}' for i in range(3000)]
text = ' '.join((random.Random(0).choice(words) for _ in range(40000)))
os.makedirs(os.path.dirname(root), exist_ok=True)
open(os.path.join(os.path.dirname(root), 'fake_text.txt'), 'w').write(text)
tk = Tokenizer(models.WordLevel(vocab={'[UNK]': 0, **{w: i + 1 for i, w in enumerate(words)}}, unk_token='[UNK]'))
tk.pre_tokenizer = pre_tokenizers.Whitespace()
tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token='[UNK]')
cfg = Qwen2Config(vocab_size=len(tok), hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512, tie_word_embeddings=False)
m = Qwen2ForCausalLM(cfg).half()

def snap(repo):
    d = os.path.join(root, 'models--' + repo.replace('/', '--'), 'snapshots', 'abc')
    os.makedirs(d)
    return d
b = snap('Fake/base')
m.save_pretrained(b, safe_serialization=True)
tok.save_pretrained(b)
from safetensors.torch import save_file
for kind in ('awq', 'gptq4', 'gptq8'):
    bits = 8 if kind == 'gptq8' else 4
    g = 128
    sd = {}
    for n, mod in m.named_modules():
        if isinstance(mod, torch.nn.Linear) and n != 'lm_head':
            W = mod.weight.data.float().t()
            inn, out = W.shape
            Wg = W.reshape(inn // g, g, out)
            lo, hi = (Wg.amin(1), Wg.amax(1))
            qmax = (1 << bits) - 1
            s = ((hi - lo) / qmax).clamp(min=1e-06).half()
            z = torch.round(-lo / s.float()).clamp(0, qmax)
            q = torch.clamp(torch.round(Wg / s.float()[:, None]) + z[:, None], 0, qmax).reshape(inn, out)
            if kind == 'awq':
                qw, qz = pack_awq(q.long(), z.long())
                sd.update({n + '.qweight': qw, n + '.qzeros': qz, n + '.scales': s})
            else:
                qw, qz = pack_gptq(q.long(), z.long(), bits)
                sd.update({n + '.qweight': qw, n + '.qzeros': qz, n + '.scales': s, n + '.g_idx': torch.arange(inn, dtype=torch.int32) // g})
            if mod.bias is not None:
                sd[n + '.bias'] = mod.bias.data.clone()
        elif n == '':
            pass
    for k, v in m.state_dict().items():
        if not any((k.startswith(x) for x in [n for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear) and n != 'lm_head'])):
            sd[k] = v.clone()
    d = snap(f'Fake/base-{kind}')
    save_file(sd, os.path.join(d, 'model.safetensors'))
    c = json.load(open(os.path.join(b, 'config.json')))
    c['quantization_config'] = dict(bits=bits, group_size=g, quant_method='awq' if kind == 'awq' else 'gptq', desc_act=False, zero_point=True, version='gemm')
    json.dump(c, open(os.path.join(d, 'config.json'), 'w'))
print('fake hub ready', root)
