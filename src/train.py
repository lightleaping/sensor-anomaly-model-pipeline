import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import SensorAutoEncoder


def train_model(
    data_path: str = "outputs/preprocessed_data.npz",
    model_path: str = "models/autoencoder.pt",
    history_path: str = "outputs/train_history.csv",
    epochs: int = 80,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    random_seed: int = 42,
) -> None:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Preprocessed data not found: {data_file}")

    data = np.load(data_file)
    X_train = data["X_train"].astype(np.float32)
    X_val = data["X_val"].astype(np.float32)

    train_dataset = TensorDataset(torch.tensor(X_train))
    val_tensor = torch.tensor(X_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = SensorAutoEncoder(input_dim=X_train.shape[1], latent_dim=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for (batch_x,) in train_loader:
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_reconstructed = model(val_tensor)
            val_loss = criterion(val_reconstructed, val_tensor).item()

        avg_train_loss = float(np.mean(train_losses))

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
        })

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={avg_train_loss:.6f} | "
                f"val_loss={val_loss:.6f}"
            )

    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": X_train.shape[1],
            "latent_dim": 2,
        },
        model_file,
    )

    history_file = Path(history_path)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_file, index=False, encoding="utf-8-sig")

    print(f"Saved model: {model_file}")
    print(f"Saved train history: {history_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/preprocessed_data.npz")
    parser.add_argument("--model", default="models/autoencoder.pt")
    parser.add_argument("--history", default="outputs/train_history.csv")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train_model(
        data_path=args.data,
        model_path=args.model,
        history_path=args.history,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
