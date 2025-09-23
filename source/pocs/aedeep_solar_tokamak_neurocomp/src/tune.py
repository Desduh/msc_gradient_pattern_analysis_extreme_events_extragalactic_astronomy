import optuna
from train import objective

if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner()
    )
    study.optimize(objective, n_trials=50)
    # study.optimize(objective, n_trials=30, timeout=3600)  # até 30 tentativas ou 1h

    print("Melhores hiperparâmetros encontrados:")
    print(study.best_trial.params)

Melhores hiperparâmetros encontrados:
{'lr': 0.004533235897845981, 'num_layers': 1, 'dropout': 0.14070177241232235}