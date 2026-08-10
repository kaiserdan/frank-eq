# Independent review prompt

Review Frank-EQ as a skeptical ICLR-area-chair-level collaborator.

Start with `AGENTS.md`, `docs/00_PROJECT_CHARTER.md`, `docs/03_INFORMATION_ACCESS_CONTRACT.md`, `docs/04_STAGE0_PROTOCOL.md`, and `evidence/reference_stage0/`.

Determine:

1. whether the implementation actually forms state before operation reveal;
2. whether the public code is gauge fixed or merely another learned shared latent;
3. whether held-out operations test state sufficiency rather than decoder learning;
4. whether world grouping prevents renderer/branch leakage;
5. whether held-sender onboarding keeps the executor frozen;
6. whether facts-only and residual comparisons are valid;
7. whether bootstrap units and machine gates match the protocol;
8. which synthetic conveniences will fail to transfer to real LLMs;
9. the smallest real-model canary that could falsify the project;
10. what claims remain prohibited.

Cite exact paths and values. Separate measured evidence, code-supported invariants, documented plans, and unverified assumptions.
