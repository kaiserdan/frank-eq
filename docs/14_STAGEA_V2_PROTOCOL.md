# Stage-A v2 protocol registration: native chat-template competence

Status: frozen implementation protocol, proposed for ratification before any
outcome-bearing run.  
Date: 2026-08-10  
Config: `configs/stage0/real_lumi_v2.yaml`

## 1. Motivation and decision-tree branch

The v1 diagnostic (`docs/13_STAGEA_V1_CORRECTION_LOG.md`, machine code
`FIX_NATIVE_COMPETENCE_BEFORE_LATENT_REVISION`) found that the frozen source
models' own future branches lose to an operation-wise training prior on
validation worlds (founder Brier gain −0.060, negative in every family) while
facts, oracle signatures, and self-future signatures are all linearly readable
from the existing capture. Per the decision tree in
`docs/12_STAGEA_V1_AUDIT_AND_V2_PROTOCOL.md`, branch A applies: freeze a
model/task competence prerequisite before latent analysis.

## 2. Frozen hypothesis (v2-1)

> Under the same frozen checkpoints and panel geometry, replacing the raw
> world prefix with each model's native chat template (system contract +
> world rendering in the user turn) raises the source models' own future
> branch competence above operation-wise priors, and the shared-quotient
> pipeline trained on chat-templated capture passes the frozen real Stage-A
> representation gates.

Scope discipline: exactly one variable changes relative to v1 — the capture
prompt surface. Checkpoints, panel geometry, depths, chart architecture, head
scope, losses, and the frozen operation registry are unchanged. This isolates
the competence prerequisite per branch A; capture expansion (branch B) is not
in this registration.

## 3. Changes relative to v1

| Item | v1 | v2-1 |
|---|---|---|
| Capture prompt | raw non-chat text | native chat template per model (system contract + world statement in user turn) |
| World seed | 1729 | 20260810 (fresh worlds, untouched test role) |
| Revision pins | omitted | required, exact hashes in every model entry |
| KV-vs-replay parity | branch-mode counts only | registered 32-branch audit per model; kv_reuse exclusive, no fallback; tolerance 0.33 (3× measured max) |
| Native-competence gate | absent | frozen, validation-worlds, founders, held-out operations |
| Residual gate | decision-gating | declared-global control, reported but not gating |
| Renderer invariance | cosine only | conjunctive: cosine AND retrieval AND low leakage (unchanged, restated) |
| Decision schema | real v2 schema (PR) | unchanged |
| Compiler scope | shared heads | shared heads (declared; `model.public_head_scope: shared`) |

## 4. Ten registration prerequisites (docs/12 section 5)

1. **Fresh world seed and untouched test role** — `panel.seed: 20260810`;
   v1 test worlds are never scored in v2. `train/validation/test` split
   rebuilt from the new seed.
2. **Exact checkpoint revision pins** — every model entry carries
   `revision:`; `require_revision_pins: true` fails config validation
   otherwise. Pins match the staged LUMI HF cache exactly.
3. **Native-competence gate defined before test execution** —
   `gates.min_native_competence_brier_gain: 0.0`: mean over founder models,
   validation worlds, held-out operations, of
   `brier(model_signatures vs oracle) − brier(coordinate prior)`. Computed
   from the frozen cache (no trained weights), so it cannot be gamed by
   chart selection. Frozen in this document and in the config before the
   run.
4. **Exact capture stream definitions** — final-token residual at normalized
   depths 0.35/0.60/0.85 of the chat-templated prefix; prefix tokenized with
   `add_special_tokens=False` (templates carry their own opening markers —
   Llama emits `<|begin_of_text|>`, Qwen3/SmolLM open with `<|im_start|>` —
   and injecting special tokens would double tags or append a trailing EOS);
   operation text appended raw after the assistant-turn boundary; capture
   step strictly before reveal; per-model hidden dims recorded in metadata.
   Dimensionality per model: qwen3-0.6b 1536, smollm-1.7b-instruct 2048,
   llama-3.2-1b-instruct-held 2048.
5. **KV-versus-replay numerical parity on a frozen sample** —
   measured noise floor on a 32-branch sample per model (dev-g, bf16/ROCm):
   qwen3-0.6b max 0.0757, smollm-1.7b max 0.1089, llama-3.2-1b-held max
   0.0517. The two branch modes are therefore NOT interchangeable at gate
   precision on this stack; `branch_mode: kv_reuse` is the exclusive mode
   (`allow_exact_replay_fallback: false`), so no cache can mix modes. The
   32-branch dual-mode sample remains registered per model in
   `cache/metadata.json` as a stack-property audit; tolerance
   `parity_max_abs_diff: 0.33` = max(0.05, 3 × measured max 0.1089),
   guarding any future re-enablement of fallback.
6. **Compiler scope declared in advance** — `model.public_head_scope:
   shared` (v1-compatible). Local compilers remain a dormant option only.
7. **Self-future and oracle namespaces** — `model_signatures` (self-future)
   and `signatures` (oracle) remain separate matrices and separate metric
   namespaces (`source_branch_*`, `quotient_*` diagnostic vs
   `heldout_signature_*` gating). Only oracle targets gate the decision.
8. **Prior-relative metrics** — every decision metric keeps its CI; fact and
   signature metrics are reported against coordinate priors and the
   reconstructed-majority baseline; native competence is prior-relative by
   construction.
9. **Renderer invariance conditioned on non-collapse** — the invariance
   check passes only conjunctively: `renderer_cosine >= 0.85` AND
   `cross_model_retrieval >= 0.30` AND `model_leakage_over_chance <= 0.25`.
   Renderer cosine alone cannot pass.
10. **Real-specific reducer schema** — `frank_eq_real_stagea_decision_v2`
    with `authorizes_receiver_protocol_design` on pass and
    `authorizes_scientific_claim=false` always.

## 5. Frozen gate table (v2-1)

Same thresholds as the adopted v1 config except where noted.

| Check | Threshold | Basis |
|---|---|---|
| heldout_signature_brier | upper95 ≤ 0.18 | test worlds, held-out ops |
| fact_accuracy | lower95 ≥ 0.70 | test worlds |
| renderer_invariance | mean ≥ 0.85 | test worlds |
| cross_model_retrieval | lower95 ≥ 0.30 | test worlds |
| wrong_world_margin | lower95 ≥ 0.03 | test worlds |
| quantization_retention | ≥ 0.90 | test worlds |
| held_model_retention | ≥ 0.70 | test worlds |
| model_identity_leakage | over chance ≤ 0.25 | test worlds |
| **native_competence** | **≥ 0.0** | **validation worlds, founders, held-out ops (new)** |
| operational_residual | reported, not gating | declared-global control (density/reciprocity tags are printed in the prefix) |

## 6. Access and authorization rules

- The v1 test role is never scored in v2. No v1 artifact is reused as a
  label source.
- Cache build enforces capture-before-reveal, exact prefix continuity
  (never re-tokenize prefix jointly with the query), split-by-world, and
  full registry coverage.
- Telemetry is fail-open and never gates. W&B project `frank-eq-stagea`,
  tags `[real-stagea, lumi, v2, chat-template]`.
- A negative decision under this frozen registration is a valid terminal
  negative for v2-1; it authorizes no tuning of gates, layers, panel, or
  prompts, and no receiver execution.
- This registration authorizes one outcome-bearing run on the frozen
  `real_lumi_v2.yaml` after ratification. Any deviation (chat template
  variant, checkpoint set, capture expansion, local head scope) is a new
  versioned registration with a fresh test role.

## 7. Review gates before ratification

- [ ] diagnostic recommendation recorded (docs/13) and this document read;
- [ ] config validation passes with `require_revision_pins: true`;
- [ ] parity audit implemented and unit-tested;
- [ ] native-competence gate implemented and unit-tested;
- [ ] repo validation suite green;
- [ ] dry-run submission succeeds with the frozen config.
