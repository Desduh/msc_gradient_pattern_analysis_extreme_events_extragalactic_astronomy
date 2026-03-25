# model.py

import torch
import torch.nn as nn
torch.backends.mkldnn.enabled = True

class PeakPredictorLSTM(nn.Module):
    """
    LSTM-based model for peak prediction in time series.

    The model applies an input projection followed by stacked LSTM layers
    and outputs a single logit representing the probability of a future peak.
    """

    def __init__(self, model_config):
        super().__init__()

        self.projection_dim = model_config["projection_dim"]
        self.hidden_dim = model_config["hidden_dim"]
        self.num_layers = model_config["num_layers"]

        # -------- Input projection --------
        self.input_projection = nn.Linear(
            model_config["input_dim"], self.projection_dim
        )
        self.silu = nn.SiLU()

        # -------- LSTM stack --------
        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(
            nn.LSTM(
                self.projection_dim,
                self.hidden_dim,
                batch_first=True
            )
        )

        for _ in range(self.num_layers - 1):
            self.lstm_layers.append(
                nn.LSTM(
                    self.hidden_dim,
                    self.hidden_dim,
                    batch_first=True
                )
            )

        self.dropout = nn.Dropout(model_config["dropout"])

        # -------- Output projection --------
        self.output_projection = nn.Linear(self.hidden_dim, 1)
        # Sigmoid is intentionally omitted (use BCEWithLogitsLoss)

    def forward(self, x):
        """
        Forward pass for full sequence inference.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            torch.Tensor: Output logits of shape (batch, 1)
        """
        x = self.input_projection(x)
        x = self.silu(x)

        for i, lstm_layer in enumerate(self.lstm_layers):
            x, _ = lstm_layer(x)
            if i < self.num_layers - 1:
                x = self.dropout(x)
                x = self.silu(x)

        # Use the last time step
        last_step = x[:, -1, :]
        out = self.output_projection(last_step)

        return out

    def init_hidden(self, batch_size=1):
        """
        Initializes hidden and cell states on the same device as the model.

        Args:
            batch_size (int): Batch size for the hidden states

        Returns:
            list[tuple]: List of (h, c) tuples for each LSTM layer
        """
        device = next(self.parameters()).device
        hidden_states = []

        for _ in range(self.num_layers):
            h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
            c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
            hidden_states.append((h, c))

        return hidden_states

    def step(self, x_t, prev_hidden_states):
        """
        Performs a single-step forward pass (online inference).

        Args:
            x_t (torch.Tensor): Input at time t, shape (1, 1, input_dim)
            prev_hidden_states (list): Previous hidden states for each LSTM layer

        Returns:
            torch.Tensor: Output logit
            list[tuple]: Updated hidden states
        """
        x_t = self.input_projection(x_t)
        x_t = self.silu(x_t)

        new_hidden_states = []
        current_input = x_t

        for i, lstm_layer in enumerate(self.lstm_layers):
            output, (h_new, c_new) = lstm_layer(
                current_input, prev_hidden_states[i]
            )

            current_input = output
            if i < self.num_layers - 1:
                current_input = self.dropout(current_input)
                current_input = self.silu(current_input)

            new_hidden_states.append((h_new, c_new))

        prediction = self.output_projection(current_input.squeeze(1))
        return prediction, new_hidden_states
