import os, re, sys, json
roots = sys.argv[1:-1]
out = {}

def is_test(p):
    q = '/' + p.lower().replace('\\', '/')
    b = os.path.basename(q)
    return any((x in q for x in ('/tests/', '/test/', '/testing/', '/build/lib/', '/docs/', '/examples/'))) or b.startswith('test_') or b.endswith('_test.py') or (b == 'conftest.py')

def count_A(s):
    n = len(re.findall('(?:torch|th|t)\\.distributions(?:\\.kl)?\\.kl_divergence\\s*\\(', s))
    al = set(re.findall('import\\s+torch\\.distributions(?:\\.kl)?\\s+as\\s+(\\w+)', s)) | set(re.findall('from\\s+torch\\s+import\\s+(?:[\\w ,]*?\\b)?distributions\\s+as\\s+(\\w+)', s))
    al |= set(re.findall('from\\s+torch\\.distributions\\s+import\\s+(?:[\\w ,\\(\\)\\n]*?\\b)?kl\\s+as\\s+(\\w+)', s))
    if re.search('from\\s+torch\\.distributions\\s+import\\s+(?:\\(?[\\w ,\\n]*?\\b)?kl\\b(?!_)(?!\\s+as)', s):
        al.add('kl')
    if re.search('from\\s+torch\\s+import\\s+(?:\\(?[\\w ,\\n]*?\\b)?distributions\\b(?!\\s+as)', s):
        al.add('distributions')
    if re.search('from\\s+torch\\.distributions\\s+import\\s+(?:\\(?[\\w ,\\n]*?\\b)?kl\\b(?!_)(?!\\s+as)', s):
        al.add('kl')
    for a in al:
        n += len(re.findall('(?<![\\w.])' + re.escape(a) + '(?:\\.kl)?\\.kl_divergence\\s*\\(', s))
    names = set()
    for m in re.finditer('from\\s+torch\\.distributions(?:\\.kl)?\\s+import\\s+(\\([^)]*\\)|[^\\n]*)', s):
        for part in m.group(1).strip('()').split(','):
            t = part.split('#')[0].split()
            if t and t[0] == 'kl_divergence':
                names.add(t[2] if len(t) == 3 and t[1] == 'as' else 'kl_divergence')
    for nm in names:
        if re.search('def\\s+' + re.escape(nm) + '\\s*\\(', s):
            continue
        n += len(re.findall('(?<![\\w.])' + re.escape(nm) + '\\s*\\(', s))
    return n

def count_B(s):
    n = len(re.findall('(?<![\\w])(?:F|functional|torch|th|t|nnf|TF)\\.kl_div\\s*\\(', s)) + len(re.findall('KLDivLoss\\s*\\(', s))
    if re.search('from\\s+torch\\.nn\\.functional\\s+import\\s+(?:\\(?[\\w ,\\n]*?\\b)?kl_div\\b', s) and (not re.search('def\\s+kl_div\\s*\\(', s)):
        n += len(re.findall('(?<![\\w.])kl_div\\s*\\(', s))
    return n
for root in roots:
    for pkg in sorted(os.listdir(root)):
        a, b = ([], [])
        for dp, dn, fn in os.walk(os.path.join(root, pkg)):
            for f in fn:
                if not f.endswith('.py'):
                    continue
                p = os.path.join(dp, f)
                rel = os.path.relpath(p, os.path.join(root, pkg))
                if is_test(rel):
                    continue
                try:
                    s = open(p, encoding='utf-8', errors='ignore').read()
                except Exception:
                    continue
                if 'kl_div' not in s and 'KLDivLoss' not in s:
                    continue
                s2 = re.sub('(?m)^\\s*#.*$', '', s)
                na, nb = (count_A(s2), count_B(s2))
                if na:
                    a.append((rel, na))
                if nb:
                    b.append((rel, nb))
        d = out.setdefault(pkg, dict(A=[], B=[]))
        d['A'] += a
        d['B'] += b
json.dump(out, open(sys.argv[-1], 'w'), indent=1)
A = [k for k, v in out.items() if v['A']]
B = [k for k, v in out.items() if v['B']]
print(len(out), 'packages; A:', len(A), 'B:', len(B), 'either:', len(set(A) | set(B)))
print('A:', ', '.join((f"{k} ({len(out[k]['A'])}/{sum((n for _, n in out[k]['A']))})" for k in A)))
print('B:', ', '.join((f"{k} ({len(out[k]['B'])}/{sum((n for _, n in out[k]['B']))})" for k in B)))
