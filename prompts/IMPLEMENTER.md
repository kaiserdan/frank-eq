# Frank-EQ implementer prompt

You are implementing a bounded Frank-EQ work package.

Read `AGENTS.md`, `docs/02_ARCHITECTURE.md`, `docs/03_INFORMATION_ACCESS_CONTRACT.md`, `docs/05_GATES_AND_STOP_RULES.md`, and `HANDOFF.md` before editing.

Requirements:

- preserve state-before-operation timing;
- preserve world-grouped splits;
- do not introduce target-private runtime inputs;
- do not make the public decoder sender specific;
- add focused tests for every new invariant;
- write machine-readable artifacts and fail closed on missing fields;
- update documentation and the append-only decision log;
- run compile, tests, repository validation, and the relevant smoke;
- report exact files changed, commands run, outcomes, and unresolved limitations.

Do not optimize to make a frozen gate pass. Diagnose a miss and stop unless a separately versioned protocol authorizes a new hypothesis.
