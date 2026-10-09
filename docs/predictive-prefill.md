# Predicting the first adaptation pass

`DSV41_PROBE_PREFILL=1` enables computed discovery in **apply mode**: the normal
planner loads the selected experts before real prefill. `0` disables discovery.
This single switch selects the computed method; no backend selector, shadow-mode
setting, dictionary or training bank is needed in `.env` or the example profiles.
It takes precedence over all legacy predictor mode/backend settings.

The dual-node example profiles use a 64-token excerpt cap, a `.1` token fraction,
and full discovery below 500 uncached tokens. Both discovery paths reuse the same
global ranking and fixed resident budget. Missing experts are not streamed during
token computation. Ordinary residency learning and post-prefill correction remain.

## Computed small probe

`probe` forms a bounded shorter sequence from the instruction head, two windows
of the uncached suffix, and its tail. Half the token budget goes to the tail,
usually containing the latest question. Excerpts stay in original order. The
excerpt path does not run a complete second prefill.

`DSV41_PROBE_TWO_PASS_BELOW=500` enables full discovery for **1–499 uncached
tokens**, using the actual prefix-cache match rather than total context length.
Zero new tokens skips discovery; 500 or more uses the excerpt path. This short
path bypasses the ordinary 32-token adaptation minimum and excerpt fraction/cap.
Both passes reuse the cached prefix. The first runs the complete new suffix with
current residents and the ordinary bounded decoder replay, collects routing into
separate counters, restores the valid prefix boundary, and loads the planner's
replacements. The second computes the real suffix and final logits. Discovery
adds no history votes or prefix snapshots. Normal history accounting and later
correction remain enabled. `shadow` does the discovery and rerun without loading.

Rollback preserves bounded window rings, pending compressor inputs and replay
containers. Compressed/index rows beyond the restored boundary are unreachable
and are overwritten by the second pass, so the entire KV cache is not copied.
Original-source Engram hashes avoid mutating the live hash cache. The normal
decoder replay can include cached tail rows; full discovery records them because
the real prefill executes them too. Cached prefix states retain their historical
weights. Discovery uses the current resident model; loading different experts
can change the second pass's demand.

Both ranks run the ordinary resident model on these excerpts, recording unmasked
expert preferences. With decoder SWA replay enabled, the probe uses that same
encoder/replay split. Cached head rows provide approximate context but do not
vote in the demand histogram. Original Engram n-grams are preserved separately
at excerpt boundaries; attention and positional context are still approximations.

The resulting per-layer counts and score mass are passed provisionally to the
normal global planner. `shadow` evaluates the plan against actual prefill demand;
`apply` loads the planned replacements before real prefill. Post-prefill correction
remains enabled. The probe uses no prompt-history lookup, needs no training bank,
and neither reads nor updates `DSV41_PREDICTIVE_DB`. Probe passes add no history
votes; actual prefill and decode retain the ordinary adaptive accounting.

The probe has separate KV caches, replay state, hash scratch and demand counters.
It does not overwrite the live prefix state or captured demand-buffer addresses.
Prefill FFN graphs are temporarily bypassed in the probe and restored afterward;
normal prefill and decode keep their original graphs. Expert weights, static
resident routes and Engram row caches are shared. Probe reads can warm those row
caches, which must be accounted for when interpreting timing comparisons.
Scratch KV reserves roughly 0.18 GB per rank with the default 4,096-row rings,
plus probe activations and small counters. It is retained for later requests.

| Variable | Default | Meaning |
| --- | --- | --- |
| `DSV41_PROBE_PREFILL` | unset (disabled) | `1` enables computed apply; `0` disables; overrides legacy predictor settings |
| `DSV41_PROBE_TOKENS` | `256` | Maximum excerpt tokens, at most 1,024 |
| `DSV41_PROBE_FRACTION` | `.05` | Token budget as a fraction of the uncached suffix, at most `.1` |
| `DSV41_PROBE_MIN_TOKENS` | `32` | Skip when the token budget is smaller |
| `DSV41_PROBE_TARGET_RATIO` | `.1` | Measured prediction-time/prefill-time target |
| `DSV41_PROBE_TWO_PASS_BELOW` | `0` | Full discovery when 0 < uncached tokens < this threshold; 0 disables, maximum 1,024 |

The excerpt time target is **telemetry, not a hard execution limit**. A token fraction is
not a time fraction: small batches still read weights and issue collectives. At
finish, logs report prediction/planning time, probe time, observed ordinary
prefill time, `prediction_prefill_ratio`, and `probe_within_target`. Weight-loading
time is reported separately. Full short-suffix discovery intentionally spends
another prefill: its ratio is still reported, with `probe_within_target=null`.
It is exempt from the excerpt 10% target. First-use allocation/JIT can be slower.
Fully cached requests and vision skip. Without short discovery, requests below
the normal adaptation threshold or minimum probe budget also skip. All enabled
requests still reach the same rank-authoritative
decision broadcasts. Both ranks agree on scratch preparation before model
collectives; a failure during a distributed forward is fatal rather than retried.
Probe policy fields join the boot configuration guard. Concurrency must be 1,
and layer ablation must be disabled.
Full short-suffix discovery additionally requires SWA replay.

Qualification uses the disposable two-node driver
`tools/bench_probe_prefill_tp.py`. It freezes the history and residency for matched
off/shadow comparisons, checks generated tokens and actual routing demand for
exact equality, includes cached-extension/full-hit checks, grades retrieval and
executable Python answers, and optionally exercises early apply. It suppresses
production DB writes and must run with serving stopped. See `RESULTS.md` for
measured cost and selection accuracy before enabling this backend in production.

`tools/bench_two_pass_tp.py` qualifies the short path with >98% prefix reuse,
odd/even compressor boundaries, 1/31/128/499/500 new tokens, a full hit, a cold
short prompt, and a serialized medical conversation turn. Off/shadow must match
outputs, real demand and history exactly. Shadow's full discovery must match
actual real-pass demand and the incoming expert plan, and valid cached-prefix
rows, pending state and replay containers must survive discovery unchanged.
The final apply turn keeps ordinary post-prefill correction and grades its answer.

The intended use is the **first request after a sudden topic change**. For that
evaluation, `tools/bench_topic_shift_tp.py` starts from a copied demand history,
trains residency on CS, Python and fictional news, and switches to synthetic
medical records. It compares off, shadow and early apply on the same inputs. Each
arm restores exact warmed expert-to-slot placement, masks, decode LUT and adaptive
counters; ordinary post-prefill/decode correction remains enabled. It measures
first-prefill count/score misses, compute and loading cost, total prefill latency,
and fixed-answer chart/trial tasks. A Python continuation is the same-topic
control. Off/shadow outputs and demand must match, and both ranks must agree.

Fixed-demand coverage also applies each early-loaded mask to the off arm's actual
routing observations. This separates changes in residency from changes in hidden
states after loading different experts. It is still resident-model routing, not
an all-expert quality reference. These fixtures test document extraction and
invented eligibility rules, not clinical competence. The current prototype has
no automatic topic-change trigger; that requires a separate qualification.

## Original nearest-neighbor backend

The legacy controls below remain for compatibility and diagnostic qualification.
They are absent from the serving profiles. To use them explicitly, first unset
`DSV41_PROBE_PREFILL`; an explicit `0` keeps both predictors disabled.

`DSV41_PREDICTIVE_PREFILL=shadow` learns a bounded bank of prompt features and
observed prefill routing distributions. It estimates demand before prefill and
compares its proposed promotions with those selected from actual prefill demand.
`apply` additionally executes the predicted plan before prefill. The engine's own default
is `off`; select `DSV41_PREDICTIVE_BACKEND=nearest` explicitly to use this bank. This is experimental: similar vocabulary does
not guarantee similar routing, and CPU speed is not evidence of selection quality.

## Request flow

1. Restore any reusable prefix cache and identify the uncached suffix.
2. On rank 0, hash the full context, suffix vocabulary, and ordered suffix bigrams.
   Query up to four sufficiently similar prior examples. Cold starts and unfamiliar
   prompts proceed without a prediction.
3. Preview one normalized request vote alongside the existing aged demand history
   and trace prior. Use the normal global planner, layer floor, swap cap and minimum
   gain. In apply mode, broadcast and load that plan on both TP ranks.
4. Execute ordinary resident prefill. Missing experts may still be missed. There is
   no discovery forward pass, temporary expert streaming, or load-every-miss policy.
5. Evaluate the estimate against observed demand before adding the current example
   to the bank. Keep the existing post-prefill adaptation: actual observations are
   folded once and can correct the prediction before decode. This adaptation does
   not rerun prefill. Normal decode and between-request adaptation also remain.

The predicted vote is never written into demand history. Existing cached states
retain their original computation; predictions cannot improve a prefix already
cached. Better early selection could improve newly computed states, but that
quality benefit has not yet been established.

## Configuration and telemetry

Requires dynamic native FP4 TP2, recorded request-unit demand, and post-prefill
adaptation enabled. Streaming and calibrated rescue must be disabled. Prefix RAM
cache is supported. Prediction is skipped for vision and suffixes smaller than the
existing prefill adaptation threshold, including full cache hits.

| Variable | Default | Meaning |
| --- | --- | --- |
| `DSV41_PREDICTIVE_PREFILL` | `off` | `off`, `shadow`, or `apply` |
| `DSV41_PREDICTIVE_DB` | `results/predictive-prefill.npz` | Rank-0 authoritative bank |
| `DSV41_PREDICTIVE_CAPACITY` | `128` | Maximum distinct prompt/prefix examples; limit 256 |
| `DSV41_PREDICTIVE_MIN_SAMPLES` | `8` | Minimum bank size before predicting |
| `DSV41_PREDICTIVE_MIN_SIMILARITY` | `.85` | Cosine threshold, a heuristic rather than calibrated confidence |

Prediction-related control fields join the boot-time rank configuration guard.
Every enabled request reaches the same plan broadcast, even when skipped or when
prediction fails. Errors before broadcast skip prediction; failure while applying
a distributed plan terminates the rank, as with existing adaptation.

The health engine configuration exposes `predictive_prefill`. Logs include status,
similarity, estimated miss rate, planned/applied swaps, and prediction/load/observation
time. Shadow evaluation additionally reports mean total variation of the demand
distribution, promotion precision/recall, and exact swap overlap against the normal
planner using actual prefill demand. Empty promotion sets report null ratios.
Apply mode cannot provide that same counterfactual comparison after changing the
residents, so it reports demand error only.

The bank stores signed hashed token features, prompt hashes and normalized expert
counts/score mass. It stores no raw prompt text or token sequences. These derived
features are still private user data, not a guarantee of anonymization. Keep the
bank local and out of version control. Aggregate historical expert rankings cannot
initialize it: they lack the corresponding prompt features. Repeated identical
prompt/prefix pairs replace their previous entry. A full bank occupies about
16.5 MiB for 40 layers × 384 experts, plus temporary working memory.

The model configuration, tokenizer and guarded numerical/serving policy identify
the bank. A bank from another configuration, or one that cannot be read, is moved to
`<bank>.stale` at boot (replacing any earlier `.stale` copy) and an empty bank starts in
its place; the boot log line `predictive prefill: ... reset: ...` says why. Until
2026-10-09 such a bank was kept read-only instead, which silently stopped learning
after any engine update or profile edit, since every new guard field changes the identity.

## Validation

Run the focused CPU suite:

```sh
.venv/bin/python -m unittest engine.test_probe_prefill engine.test_predictive_prefill engine.test_decode_adapt \
  engine.test_global_residency engine.test_expert_seed engine.test_adapt_config \
  engine.test_prune_unit
```

Tests cover matching the normal post-observation planner, no duplicate history
votes, rank agreement, skip/error paths, bounded persistence, and long-context
feature discrimination. The live probe uses isolated copies of demand history and
a separate predictor bank so synthetic traffic does not seed normal use.

On the development Spark, feature extraction plus a 128-example nearest-neighbor
query took median 1.715 ms for 10k new tokens, 16.151 ms for 200k context/2k new,
and 22.424 ms for 524,288 context/5,243 new. These synthetic CPU measurements
exclude ranking, weight loading and model execution. No GPU kernel is needed for
this measured portion. See `RESULTS.md` for live validation and limitations.
