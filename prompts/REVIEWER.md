# Frank-EQ reviewer prompt

Act as a skeptical independent ML reviewer. Audit the implementation rather than trusting prose.

Trace:

1. when the hidden state is captured;
2. when the future operation becomes available;
3. which tensors form the private chart input;
4. how the gauge-fixed public code is constructed;
5. whether the operation decoder has learned parameters;
6. how held-out operations are masked during training;
7. how worlds, renderers, models, and branches are split;
8. which parameters update during held-sender onboarding;
9. how facts-only and residual predictions are computed;
10. how bootstrap units and gates are reduced.

Classify every conclusion as measured, code-supported, documented only, or unverified. Flag any path that could leak future operation, target state, receiver identity, or world membership.
