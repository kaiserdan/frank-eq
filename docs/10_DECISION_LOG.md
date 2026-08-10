# Append-only decision log

## 2026-08-10 — initialize Frank-EQ

- **Decision:** replace direct cross-model hidden-state reconstruction with future-defined operational equivalence quotients.
- **Scientific reason:** historical work repeatedly found shared descriptive structure without legal realization, continuation, semantic specificity, or sender-independent establishment.
- **Primary object:** a state cached before operation reveal and identified by its distribution of outcomes under a frozen family of future operations.
- **Architecture:** model-local private chart; gauge-fixed grounded facts plus operational residual; frozen public decoder; query-conditioned typed packet; receiver-native execution only after representation qualification.
- **Excluded:** pair-specific primary translators, target-hidden reconstruction, joint sender-receiver training, operation-conditioned state formation.
- **Initial gate:** synthetic Stage 0 under `configs/stage0/synthetic_full.yaml`.

## 2026-08-10 — adopt gauge-fixed public coordinates

- **Decision:** the chart's private bottleneck is not the cross-model code. The public code is constructed explicitly from grounded fact probabilities and bounded residual coordinates.
- **Reason:** an arbitrary shared bottleneck retained model identity and reproduced the same gauge ambiguity the project is intended to remove.
- **Consequence:** invariance, retrieval, packetization, leakage, and quantization are evaluated on the public state only.

## 2026-08-10 — adopt frozen public operation execution

- **Decision:** replace the learned operation decoder in the primary synthetic path with a parameter-free decoder over frozen public operation coefficients.
- **Reason:** held-out future operations should test state sufficiency, not whether a learned decoder inferred a new operation from a few examples.
- **Consequence:** operation instances can be held out while retaining exact public semantics. A learned operation decoder remains a potential baseline, not the primary executor.

## 2026-08-10 — synthetic Stage-0 pass

- **Decision:** the full reference run passes and authorizes implementation of a real-model representation canary.
- **Result:** `evidence/reference_stage0/decision.json` returns `PROMOTE_REAL_MODEL_CANARY` and `authorizes_scientific_claim=false`.
- **Next action:** implement real state capture and future-branch records without interpreting the synthetic result as LLM evidence.

## 2026-08-10 — freeze the first real Stage-A vertical slice

- **Decision:** use controlled closed-world relational graphs as the first real frozen-checkpoint task family.
- **Operation registry:** lookup, inverse, mutual relation, two-hop composition, out-degree comparison, counterfactual edge addition, graph-density class, and reciprocity class; two operation instances per family, with family-stratified holdout.
- **Public semantics:** exact graph facts plus two declared global coordinates, executed by a parameter-free differentiable graph interrogator. No target hidden state, target logit, learned receiver, or pair-specific decoder enters the primary path.
- **Source behavior:** record each source model's actual post-reveal A/B distribution, but keep it diagnostically separate from the formal oracle signature used to define the public quotient.
- **Causal boundary:** capture three normalized-depth hidden states from a prefix containing no operation. Attempt physical cloned-KV branches; allow exact-prefix replay fallback only with explicit counts.
- **Model roles:** two founder families and one final held-sender entry. Held onboarding freezes the public interrogator and all founder charts.
- **Gate scope:** a pass authorizes design of a receiver-native execution experiment only. It does not authorize a universal-latent, communication, safety, or natural-task claim.

## 2026-08-10 — make scientific failure scheduler-valid

- **Decision:** `run-real-stagea` exits successfully when the workflow and artifacts are complete even if `eval/decision.json` is negative.
- **Reason:** a preregistered scientific failure is a valid experiment and must not be conflated with an engineering or scheduler failure.
- **Authority:** `workflow_status.json` records execution integrity; `eval/decision.json` exclusively records scientific promotion.

## 2026-08-10 — adopt content-addressed Olivia/LUMI execution

- **Decision:** every cluster submission packages a deterministic source tarball, hashes it, records the exact config/stages/remote root, and writes local submission/status/fetch/verify state under `.agents/state/<cluster>/<job>`.
- **Reason:** mutable working-directory deployment and prose-only job identity are insufficient for a branch-heavy research program.
- **First authorized launch:** `cache,validate` using `configs/stage0/real_olivia.yaml` or `configs/stage0/real_lumi.yaml`, after checkpoint-cache preflight.
