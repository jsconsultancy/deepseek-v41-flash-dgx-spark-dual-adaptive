"""DSV41_DRAFT_VOCAB: the drafter's two heads over a listed vocabulary instead of all 129,280 ids.

The draft only proposes; the verifier accepts exactly the target's argmax, so greedy output is
identical under any drafter and a token the draft cannot propose costs acceptance, never a token
(docs/decode-dynamic-depth.md, "How"). Both draft heads are bandwidth reads whose time is their
row count: the fp8 vocabulary head draws 331 MB a pass at 225 GB/s (1.47 ms) and the Markov bias
head draws its 33 MB half five times (0.79 ms) (docs/exl3-draft.md, 2026-10-11 trace). A smaller
*format* did not make them faster -- the fp4 kernel's cost per byte at M = 5/6 ate the saving --
but fewer *rows* through the same kernels is a different lever: same kernel, same bytes per row.
At 40,960 listed ids a rank the Markov rows are 21 MB of bf16, just under the 24 MiB L2 that its five
back-to-back 33 MB reads cannot use today (whether they then hit L2 is for the trace to show).

The list is a ranking file (ids, most frequent first, `#` comments; `tools/build_draft_vocab.py`
writes one). Both ranks read the same file, take its first `world * rows` distinct ids, sort them
and split them into contiguous runs: rank r drafts ids[r * rows:(r + 1) * rows]. Contiguous runs
keep every rank-0 id below every rank-1 id, so the split chain's "rank 0 wins a tie" and each
rank's "lower column wins a tie" are still "lower id wins", as a full-vocabulary argmax is.

Unlike TensorFold's per-half lists (its `draft_head.per_rank`), the runs are not tied to the
verifier's vocabulary halves: the common tokens of this tokenizer sit mostly below 64,640, and an
equal quota per half wasted rank 1's rows (held-out English miss 2.1% vs 1.2% at 24,576 a rank).
The heads are built from the full `head.weight` and `markov_head`, which every rank loads anyway.

Both ranks must list the same ids or they emit different drafts with nothing in the log: the
digest of the whole list is in the boot guard (`V41Engine` cfg, `draft_vocab`).
"""
from __future__ import annotations

import hashlib
import os

import torch

VERSION = 1
DEFAULT_ROWS = 40960  # 0.28% of the A/B set's tokens unlisted; 32,768 left 1.46% (the ranking file's table)
# fp8 head scales are 32x32 blocks; keep each rank's rows whole blocks and the GEMV grid even
ROW_ALIGN = 128


def path() -> str:
    return os.environ.get("DSV41_DRAFT_VOCAB", "").strip()


def enabled() -> bool:
    return bool(path())


def rows() -> int:
    raw = os.environ.get("DSV41_DRAFT_VOCAB_ROWS", "").strip()
    n = int(raw) if raw else DEFAULT_ROWS
    if n <= 0 or n % ROW_ALIGN:
        raise ValueError(f"DSV41_DRAFT_VOCAB_ROWS must be a positive multiple of {ROW_ALIGN}, got {n}")
    return n


def ranking(file: str, vocab: int) -> list[int]:
    """A ranking file's ids, most frequent first, duplicates dropped; any id outside the vocabulary
    is an error rather than skipped, because a list built for another tokenizer must not load."""
    seen, out = set(), []
    with open(file, encoding="utf-8") as fh:
        for line in fh:
            for word in line.split("#", 1)[0].split():
                i = int(word)
                if not 0 <= i < vocab:
                    raise ValueError(f"{file}: id {i} outside the vocabulary (0..{vocab - 1})")
                if i not in seen:
                    seen.add(i)
                    out.append(i)
    return out


def select(order: list[int], rows_per_rank: int, world: int, vocab: int) -> torch.Tensor:
    """The listed ids, int64 ascending [world * rows_per_rank]: the ranking's first ids, then the
    lowest unlisted ids if the ranking is shorter. Rank r reads its contiguous run of the result."""
    want = world * rows_per_rank
    if want > vocab:
        raise ValueError(f"{world} x {rows_per_rank} listed ids exceed the vocabulary ({vocab})")
    picked = list(dict.fromkeys(order))[:want]
    if len(picked) < want:
        taken = set(picked)
        picked += [i for i in range(vocab) if i not in taken][:want - len(picked)]
    return torch.tensor(sorted(picked), dtype=torch.int64)


def rank_ids(ids_all: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    n = ids_all.numel() // world
    return ids_all[rank * n:(rank + 1) * n]


def digest(ids_all: torch.Tensor) -> str:
    return hashlib.sha256(ids_all.to(torch.int64).cpu().numpy().tobytes()).hexdigest()[:16]


def load(vocab: int, world: int) -> torch.Tensor | None:
    """The listed ids for this configuration, or None when DSV41_DRAFT_VOCAB is unset."""
    if not enabled():
        return None
    return select(ranking(path(), vocab), rows(), world, vocab)
