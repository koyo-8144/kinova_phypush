import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. MODEL DEFINITIONS (Must match Training)
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class PhysicsTransformerEstimator(nn.Module):
    def __init__(self, input_dim=2, d_model=32, nhead=4, num_encoder_layers=2, seq_len=20, dropout=0.4, version=4):
        super().__init__()
        self.version = version
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 10)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Force Projection MLPs
        self.net_force_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)) 
        self.fric_force_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)) 
        self.phys_net_proj = nn.Linear(d_model, 1)
        self.phys_fric_proj = nn.Linear(d_model, 1)

        # Spotlight Readouts
        self.q_mass = nn.Parameter(torch.randn(1, 1, d_model) * 2.0)
        self.mass_attn = nn.MultiheadAttention(d_model, 1, batch_first=True)
        self.mass_pred_mlp = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus())

        self.q_fric = nn.Parameter(torch.randn(1, 1, d_model) * 2.0)
        self.fric_attn = nn.MultiheadAttention(d_model, 1, batch_first=True)
        self.mu_pred_mlp = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus())

    def forward(self, x_acc, x_vel):
        # Training input logic: concat [vel, acc] -> [B, 20, 2]
        x = torch.cat([x_vel, x_acc], dim=-1) 
        z = self.input_proj(x)
        z = self.pos_encoder(z)
        h_enc = self.transformer_encoder(z)

        feat_net = self.net_force_mlp(h_enc)
        feat_fric = self.fric_force_mlp(h_enc)

        # Spotlight Mass
        q_m = self.q_mass.expand(x.size(0), -1, -1)
        mass_ctx, _ = self.mass_attn(query=q_m, key=feat_net, value=feat_net)
        mass_pred = self.mass_pred_mlp(mass_ctx.squeeze(1))

        # Spotlight Mu
        q_f = self.q_fric.expand(x.size(0), -1, -1)
        fric_ctx, _ = self.fric_attn(query=q_f, key=feat_fric, value=feat_fric)
        mu_pred = self.mu_pred_mlp(fric_ctx.squeeze(1))

        return torch.cat([mass_pred, mu_pred], dim=-1)

# ==========================================
# 2. DEPLOYMENT INSPECTION SCRIPT
# ==========================================

PATH = 'trained_models/transformer_epoch500.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"--- 🔍 Inspecting: {PATH} ---")

# Load and Check Weights
state_dict = torch.load(PATH, map_location=device)
print(f"Model Checkpoint Loaded. Number of parameter tensors: {len(state_dict)}")

# Initialize model (using d_model=32 and version=4 from your logs)
model = PhysicsTransformerEstimator(input_dim=2, d_model=32, seq_len=20, version=4)

try:
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print("✅ Model weights loaded successfully into PhysicsTransformerEstimator (V4).")
except Exception as e:
    print(f"❌ Weight mismatch: {e}")

# Create Dummy Input
# Your model expects sequence_length=20, input_dim=1 for acc and 1 for vel
dummy_acc = torch.randn(1, 20, 1).to(device)
dummy_vel = torch.randn(1, 20, 1).to(device)

print("\n--- 🚀 Running Dummy Inference ---")
with torch.no_grad():
    prediction = model(dummy_acc, dummy_vel)

mass_out = prediction[0, 0].item()
mu_out = prediction[0, 1].item()

print(f"Input Shape (Acc/Vel): {dummy_acc.shape}")
print(f"Output Raw Tensor:     {prediction}")
print(f"Parsed Prediction:")
print(f"  -> Predicted Mass: {mass_out:.4f} kg")
print(f"  -> Predicted Mu:   {mu_out:.4f}")