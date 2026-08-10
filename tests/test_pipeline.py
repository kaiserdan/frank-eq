from pathlib import Path

from frank_eq.config import (
    DataConfig,
    EvaluationConfig,
    GateConfig,
    LossConfig,
    ModelConfig,
    RunConfig,
    TrainingConfig,
)
from frank_eq.data.synthetic import generate_synthetic_bundle
from frank_eq.evaluation import Stage0Evaluator
from frank_eq.training import Stage0Trainer


def test_tiny_pipeline_writes_authoritative_artifacts(tmp_path: Path) -> None:
    config = RunConfig(
        run_name="test",
        output_dir=str(tmp_path),
        data=DataConfig(
            n_worlds=30,
            n_founder_models=2,
            include_held_model=True,
            n_renderers=2,
            n_layers=3,
            model_hidden_dims=[16, 20, 18],
            n_facts=6,
            n_residual=2,
            n_operations=24,
            seed=11,
        ),
        model=ModelConfig(
            code_dim=12,
            chart_hidden_dim=32,
            operation_hidden_dim=24,
            dropout=0.0,
        ),
        losses=LossConfig(
            facts=0.8,
            residual=0.4,
            renderer_invariance=0.1,
            cross_model_invariance=0.1,
            world_contrastive=0.1,
        ),
        training=TrainingConfig(
            epochs=3,
            onboarding_epochs=3,
            worlds_per_batch=8,
            patience=3,
            device="cpu",
            seed=11,
        ),
        evaluation=EvaluationConfig(bootstrap_replicates=20, bootstrap_seed=11),
        gates=GateConfig(
            max_heldout_signature_brier=1.0,
            min_fact_accuracy=0.0,
            min_renderer_cosine=-1.0,
            min_cross_model_retrieval_top1=0.0,
            min_wrong_world_margin=-2.0,
            min_residual_brier_gain=-1.0,
            min_quantization_retention=-2.0,
            min_held_model_retention=-2.0,
            max_model_leakage_over_chance=1.0,
        ),
    )
    config.validate()
    bundle = generate_synthetic_bundle(config.data)
    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    Stage0Trainer(config, bundle, train_dir).train()
    _, decision = Stage0Evaluator(
        config,
        bundle,
        checkpoint_path=train_dir / "final.pt",
        output_dir=eval_dir,
    ).evaluate()
    assert decision["status"] == "pass"
    assert (eval_dir / "metrics.json").is_file()
    assert (eval_dir / "decision.json").is_file()
    assert (eval_dir / "artifact_manifest.json").is_file()
