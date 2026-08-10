# Independent review of Stage-A v2-1

Run: `frank-eq-stagea-lumi-v2` (LUMI dev-g, Slurm 20952565).

## Preserved outcome

The run is an engineering-valid negative for the exact registered pipeline. Cache
validation, training, evaluation, artifact verification, revision pins, and the
exclusive KV-reuse execution path completed. `STOP_OR_REVISE_STAGE0` remains
authoritative. No receiver experiment or scientific claim is authorized.

The measured failures are substantial: native-competence Brier gain versus the
training-world operation prior was -0.0521; held-out signature Brier was 0.2065
(upper 95% 0.2421); fact accuracy was 0.5509 (lower 95% 0.5120);
wrong-world margin was -0.0607; held-sender retention was -0.3445; and model-ID
leakage over chance was 0.6389.

## Interpretation corrections

The exact v2-1 pipeline failed, but the repository's broader statement that
"native chat-template competence" or the prompt surface was falsified is not
identified by this run.

1. `prompt_format: chat` cached a prefix ending at an assistant-generation
   header. The registered operation text was then appended directly to that
   cache, so it was processed as assistant content rather than as a new user
   turn. This is a reproducible model-input contract, but not a normal native
   multi-turn chat interaction.
2. The v1/v2 comparison is unpaired. Changing `panel.seed` changed the sampled
   worlds, operation registry, and split/holdout assignment, in addition to the
   prompt wrapper. The difference -0.0603 versus -0.0521 therefore cannot isolate
   a prompt effect or be described as unchanged "within noise."
3. The native-competence gate used a point estimate. It had no grouped confidence
   interval even though the protocol stated that decision metrics retain
   intervals.
4. The competence prerequisite was evaluated during final evaluation, after
   quotient training and test-world scoring. It therefore did not operate as a
   prerequisite that could stop the campaign before test consumption.
5. KV-reuse is internally consistent as the exclusive v2 branch path. The
   observed replay differences up to 0.1089 show that replay and KV reuse are not
   interchangeable on this stack; the amended 0.33 threshold is a stack alarm,
   not evidence of numerical equivalence.

These corrections do not turn the run positive. They narrow what was falsified
to the exact legacy chat-assistant-continuation pipeline.

## Scientific indication

Across v1 and v2-1, the shared-head oracle-grounded quotient remains strongly
negative. The only encouraging observation is development-only readability of
some models' own future branch distributions. That supports separating two
questions:

- behavioral operational state: what the frozen model will do under future
  operations;
- semantic grounding: whether that state corresponds to the correct external
  world.

The next experiment should not train another quotient or use another test split
until source competence is qualified under a correct conversation contract.

## Authorized continuation: Stage Q

Implement and run a development-only qualification:

1. cache a complete conversation containing system, user world, and a fixed
   assistant acknowledgement;
2. reveal each operation as a new user turn followed by the assistant-generation
   marker;
3. verify exact token-prefix continuity before KV branching;
4. use the same worlds, operations, split, checkpoints, and branch path for any
   raw-versus-chat comparison;
5. compute prior-relative founder competence on training/validation worlds with
   a world-grouped 95% interval;
6. stop before quotient training or test-world evaluation unless the lower
   interval bound is non-negative.

A pass only permits registration of one fresh Stage-A hypothesis. It does not
authorize receiver execution or a scientific claim.
