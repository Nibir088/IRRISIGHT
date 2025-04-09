from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import optuna


def run_hyperparameter_optimization(cfg: DictConfig) -> Dict[str, Any]:
    """Run hyperparameter optimization using Optuna."""
    
    # Create base directory for all trials
    base_save_dir = Path(cfg.train.save_dir) / "hparam_tuning"
    base_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{cfg.logging.project_name}_hparam_tuning",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=20,
            interval_steps=10
        )
    )
    
    # Run optimization
    optimization_fnc = OptunaObjective(cfg, base_save_dir)
    study.optimize(
        optimization_fnc,
        n_trials=cfg.hparam_tuning.n_trials,
        timeout=cfg.hparam_tuning.timeout_hours * 3600 if hasattr(cfg.hparam_tuning, 'timeout_hours') else None,
        gc_after_trial=True,
        # callbacks=PyTorchLightningPruningCallback(
        #     trial=self.trial,
        #     monitor=monitor
        # ),
        show_progress_bar=True
    )
    
    # Save study results
    results_dir = base_save_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Save best parameters and results
    best_params = study.best_params
    best_value = study.best_value
    
    results = {
        "best_params": best_params,
        "best_value": float(best_value),
        "n_trials": len(study.trials),
        "study_name": study.study_name,
        "direction": study.direction.name
    }
    
    # Save results to YAML
    with open(results_dir / "optimization_results.yaml", "w") as f:
        yaml.safe_dump(results, f, default_flow_style=False)
    
    # Save study statistics
    stats = {
        "best_trial": {
            "number": study.best_trial.number,
            "value": float(study.best_trial.value),
            "params": study.best_trial.params
        },
        "n_trials": len(study.trials),
        "n_complete_trials": len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE])),
        "n_pruned_trials": len(study.get_trials(states=[optuna.trial.TrialState.PRUNED])),
        "n_failed_trials": len(study.get_trials(states=[optuna.trial.TrialState.FAIL]))
    }
    
    # Save statistics to YAML
    with open(results_dir / "study_statistics.yaml", "w") as f:
        yaml.safe_dump(stats, f, default_flow_style=False)
    
    print("\nHyperparameter Optimization Results:")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best value: {best_value:.4f}")
    print("\nBest parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    return best_params