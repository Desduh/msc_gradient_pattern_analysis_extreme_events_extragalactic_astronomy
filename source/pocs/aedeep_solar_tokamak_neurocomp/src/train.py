# train.py
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

from model import PeakPredictorLSTM
from pmodel_generation import generate_multiple_series, create_sequences, plot_generated_series

def prepare_dataloader(df, window_size, lookahead, batch_size):
    X, y = create_sequences(df, window_size, lookahead)
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_model(model, criterion, optimizer, train_loader, val_loader, epochs, device):
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}  # <--- adicionar histórico

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - Training"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze(-1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val).squeeze(-1)
                loss = criterion(outputs, y_val)
                val_loss += loss.item() * X_val.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

        # Atualiza histórico
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

    print("\nTreinamento finalizado. Melhor modelo salvo em 'best_model.pth'.")
    return history


def main():
    with open('src/config.json', 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = PeakPredictorLSTM(config['model']).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    print("Model Architecture:")
    print(model)

    df_train = generate_multiple_series(
        length=config['p_model']['series_length_train'],
        p_value=config['p_model']['p_value'],
        num_series=config['data']['num_series_train'],
        peak_threshold=config['data']['peak_threshold'],
        norm_percentile=config['data']['normalization_percentile']
    )

    df_validation = generate_multiple_series(
        length=config['p_model']['series_length_eval'],
        p_value=config['p_model']['p_value'],
        num_series=config['data']['num_series_validation'],
        peak_threshold=config['data']['peak_threshold'],
        norm_percentile=config['data']['normalization_percentile']
    )

    df_test = generate_multiple_series(
        length=config['p_model']['series_length_eval'],
        p_value=config['p_model']['p_value'],
        num_series=config['data']['num_series_test'],
        peak_threshold=config['data']['peak_threshold'],
        norm_percentile=config['data']['normalization_percentile']
    )

    plot_generated_series(df_train, title="Exemplo de Séries p-model (Treino)")
    plot_generated_series(df_validation, title="Exemplo de Séries p-model (Validação)")
    plot_generated_series(df_test, title="Exemplo de Séries p-model (Teste)")

    train_loader = prepare_dataloader(df_train, config['data']['window_size'], config['data']['lookahead'], config['training']['batch_size'])
    val_loader = prepare_dataloader(df_validation, config['data']['window_size'], config['data']['lookahead'], config['training']['batch_size'])

    history = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        device=device
    )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8,5))
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='o')
    plt.title("Histórico de Treinamento")
    plt.xlabel("Época")
    plt.ylabel("Loss (BCELoss)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    torch.save(model.state_dict(), config['files']['model_save_path'])
    print(f"\nTraining complete. Model saved to {config['files']['model_save_path']}")


if __name__ == '__main__':
    main()


import optuna
from optuna.exceptions import TrialPruned

def prepare_dataloader_subset(df, window_size, lookahead, batch_size, frac=0.2):
    """
    Usa apenas uma fração sequencial inicial do dataset (mantendo ordem temporal).
    Garante que haja amostras suficientes para gerar sequências.
    """
    n_min = window_size + lookahead + 1  # tamanho mínimo para gerar ao menos 1 sequência
    n_frac = int(len(df) * frac)
    n = max(n_frac, n_min)  # garante que nunca fique menor que o necessário

    if n > len(df):
        n = len(df)  # não ultrapassar limite

    df_subset = df.iloc[:n]

    X, y = create_sequences(df_subset, window_size, lookahead)
    if len(X) == 0:
        raise ValueError(f"Subset de dados muito pequeno: {n} pontos não geraram sequências.")

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
import logging
from optuna.exceptions import TrialPruned

# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def objective(trial):
    with open('src/config.json', 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # 🔹 Sugerir hiperparâmetros automaticamente
    # config['training']['learning_rate'] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    # config['training']['batch_size'] = trial.suggest_categorical("batch_size", [32, 64, 128])
    # config['model']['hidden_dim'] = trial.suggest_int("hidden_dim", 32, 128, step=16)
    # config['model']['num_layers'] = trial.suggest_int("num_layers", 2, 6)
    # config['model']['dropout'] = trial.suggest_float("dropout", 0.1, 0.5)
    # {'lr': 0.001374028762900389, 'batch_size': 128, 'hidden_dim': 112, 'num_layers': 2, 'dropout': 0.1728596770684891}

    # config['training']['learning_rate'] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    # config['training']['batch_size'] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    # config['model']['hidden_dim'] = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
    # config['model']['num_layers'] = trial.suggest_int("num_layers", 1, 3)
    # config['model']['dropout'] = trial.suggest_float("dropout", 0.1, 0.2)
    # config['data']['window_size'] = trial.suggest_categorical("window_size", [128, 256, 512])
    # config['model']['bidirectional'] = trial.suggest_categorical("bidirectional", [True, False])

    # {'lr': 0.005116626777172824, 'batch_size': 128, 'hidden_dim': 128, 'num_layers': 1, 'dropout': 0.12484365601844347, 'window_size': 512, 'bidirectional': False}

    config['training']['learning_rate'] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    config['training']['batch_size'] = 128
    config['model']['hidden_dim'] = 128
    config['model']['num_layers'] = trial.suggest_int("num_layers", 1, 6)
    config['model']['dropout'] = trial.suggest_float("dropout", 0.1, 0.2)
    config['data']['window_size'] = 512
    config['model']['bidirectional'] = False



    logging.info(f"🎯 Iniciando trial {trial.number} com params: {trial.params}")

    # 🔹 Gerar datasets
    df_train = generate_multiple_series(
        length=config['p_model']['series_length_train'],
        p_value=config['p_model']['p_value'],
        num_series=config['data']['num_series_train'],
        peak_threshold=config['data']['peak_threshold'],
        norm_percentile=config['data']['normalization_percentile']
    )
    df_validation = generate_multiple_series(
        length=config['p_model']['series_length_eval'],
        p_value=config['p_model']['p_value'],
        num_series=config['data']['num_series_validation'],
        peak_threshold=config['data']['peak_threshold'],
        norm_percentile=config['data']['normalization_percentile']
    )

    train_loader = prepare_dataloader_subset(
        df_train,
        config['data']['window_size'],
        config['data']['lookahead'],
        config['training']['batch_size'],
        frac=0.2
    )

    val_loader = prepare_dataloader_subset(
        df_validation,
        config['data']['window_size'],
        config['data']['lookahead'],
        config['training']['batch_size'],
        frac=0.2
    )

    # 🔹 Montar modelo
    model = PeakPredictorLSTM(config['model']).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    best_val_loss = float("inf")
    max_epochs = min(config['training']['epochs'], 10)  # nunca mais que 10 no tuning

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze(-1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # 🔹 Avaliação na validação
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val).squeeze(-1)
                loss = criterion(outputs, y_val)
                val_loss += loss.item() * X_val.size(0)
        val_loss /= len(val_loader.dataset)

        # 🔹 Logs de progresso
        logging.info(f"Trial {trial.number} | Epoch {epoch+1}/{max_epochs} | "
                     f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        # 🔹 Reporta progresso para o Optuna
        trial.report(val_loss, step=epoch)

        # 🔹 Early stopping do Optuna (pruning)
        if trial.should_prune():
            logging.warning(f"Trial {trial.number} pruneado na época {epoch+1}")
            raise TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    logging.info(f"✅ Trial {trial.number} finalizado com melhor Val loss = {best_val_loss:.4f}")
    return best_val_loss
