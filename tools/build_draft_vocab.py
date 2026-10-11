"""Write a DSV41_DRAFT_VOCAB ranking file (engine/draft_vocab.py) from text corpora.

    python3 tools/build_draft_vocab.py --out engine/draft_vocab_ranking_v1.txt \
        --source chat:3:ultrachat_test_sft.parquet --source chat:3:results/ \
        --source wiki:1:wikitext103_train0.parquet --source code:1:engine/ \
        --source cjk:1:docs_ja/ --source euro:1:docs_en/ \
        --exclude-replies-of results/accept-ab-20261008 --eval ab:results/accept-ab-20261008

Each group's token counts are normalized to frequencies before the weighted sum, so a 20M-token
wiki dump does not drown a 0.7M-token CJK group: a language's common tokens rank by how common they
are *in that language*. The output lists ids most frequent first; ids no source produced are left
out (draft_vocab.select fills from the lowest ids, which on this tokenizer is a poor proxy for
frequency: 87.5% coverage of held-out English docs at 24,576 a rank, against 99.5% ranked by counts).

`--exclude-replies-of DIR` drops every reply whose first 80 characters match a reply under DIR:
the A/B driver's prompts recur across results/ (prefill-trace held 35 of its 42 replies), and a
ranking that has seen the measuring set's replies understates the acceptance cost.

Sources: .parquet (`messages` assistant turns, else `text`; needs pyarrow), .jsonl (`messages`
contents, else `text`), .json (every reply-like string over 200 characters, as results/ stores
them), any other file as text, or a directory of those. Needs `tokenizers` (the engine image has it).
"""
import argparse
import collections
import glob
import json
import os

EXTS = ('.md', '.txt', '.py', '.cpp', '.cu', '.h', '.rs', '.go', '.js', '.ts', '.html', '.json', '.jsonl', '.parquet')
REPLY_KEYS = ('text', 'output', 'completion', 'reply', 'content', 'generated')


def _json_replies(obj):
    out, stack = [], [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            for k, v in o.items():
                if k in REPLY_KEYS and isinstance(v, str) and len(v) > 200:
                    out.append(v)
                else:
                    stack.append(v)
        elif isinstance(o, list):
            stack.extend(o)
    return out


def read(path):
    """Texts of one file or directory."""
    if os.path.isdir(path):
        out = []
        for f in sorted(glob.glob(os.path.join(path, '**', '*'), recursive=True)):
            if f.endswith(EXTS) and os.path.isfile(f):
                out += read(f)
        return out
    if path.endswith('.parquet'):
        import pyarrow.parquet as pq
        t = pq.read_table(path)
        if 'messages' in t.column_names:
            return [m['content'] for conv in t.column('messages').to_pylist() for m in conv
                    if m.get('role') == 'assistant' and isinstance(m.get('content'), str)]
        return [s for s in t.column('text').to_pylist() if s]
    if path.endswith('.jsonl'):
        out = []
        for line in open(path, encoding='utf-8', errors='ignore'):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            msgs = d.get('messages')
            if isinstance(msgs, list):
                out += [m['content'] for m in msgs if isinstance(m.get('content'), str)]
            elif isinstance(d.get('text'), str):
                out.append(d['text'])
        return out
    if path.endswith('.json'):
        try:
            return _json_replies(json.load(open(path, encoding='utf-8')))
        except ValueError:
            return []
    return [open(path, encoding='utf-8', errors='ignore').read()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tokenizer', default=os.path.join(os.environ.get('MODEL_DIR', ''), 'tokenizer.json'))
    ap.add_argument('--source', action='append', required=True, metavar='GROUP:WEIGHT:PATH')
    ap.add_argument('--exclude-replies-of', action='append', default=[], metavar='DIR')
    ap.add_argument('--eval', action='append', default=[], metavar='NAME:PATH')
    ap.add_argument('--rows', default='32768,40960', help='rows a rank to report coverage at')
    ap.add_argument('--top', type=int, default=98304, help='ids written (rows x 2 at most is ever read)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)

    def encode(texts):
        ids = []
        for i in range(0, len(texts), 512):
            for e in tok.encode_batch(texts[i:i + 512], add_special_tokens=False):
                ids.extend(e.ids)
        return ids

    banned = {t[:80] for d in args.exclude_replies_of for t in read(d)}
    groups, weights = collections.defaultdict(list), {}
    for spec in args.source:
        group, weight, path = spec.split(':', 2)
        if weights.setdefault(group, float(weight)) != float(weight):
            raise SystemExit(f'group {group} given two weights')
        texts = [t for t in read(path) if t[:80] not in banned]
        groups[group] += texts
        print(f'{group:6s} {path}: {len(texts)} texts', flush=True)
    score = collections.defaultdict(float)
    for group, texts in groups.items():
        counts = collections.Counter(encode(texts))
        total = sum(counts.values())
        for i, c in counts.items():
            score[i] += weights[group] * c / total
        print(f'{group:6s} {total} tokens, {len(counts)} distinct ids', flush=True)
    order = sorted(score, key=lambda i: (-score[i], i))[:args.top]

    lines = [f'# DSpark draft vocabulary ranking (tools/build_draft_vocab.py), {len(order)} ids, most frequent first.',
             '# groups: ' + ', '.join(f'{g} x{weights[g]:g}' for g in groups),
             '# sources: ' + ' '.join(args.source)]
    if args.exclude_replies_of:
        lines.append('# replies excluded: ' + ' '.join(args.exclude_replies_of))
    for spec in args.eval:
        name, path = spec.split(':', 1)
        ids = encode(read(path))
        for rows in map(int, args.rows.split(',')):
            # what draft_vocab.select loads: the ranking's first ids, then the lowest unlisted ones
            listed = set(order[:2 * rows])
            fill = (i for i in range(tok.get_vocab_size(with_added_tokens=True)) if i not in listed)
            while len(listed) < 2 * rows:
                listed.add(next(fill))
            miss = sum(i not in listed for i in ids) / max(1, len(ids))
            lines.append(f'# held-out {name}: {100 * miss:.2f}% of {len(ids)} tokens unlisted at {rows} a rank')
            print(lines[-1], flush=True)
    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
        for i in range(0, len(order), 16):
            f.write(' '.join(map(str, order[i:i + 16])) + '\n')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
