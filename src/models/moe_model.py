"""
Mixture of Experts (MoE) model for multi-timeframe cryptocurrency prediction.
Each expert specializes in a specific timeframe (5m, 30m, 1h, 1d, 1w).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class TimeframeExpert(nn.Module):
    """
    Individual expert network for a specific timeframe.
    Uses transformer-like architecture with attention mechanism.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 feedforward_dim: int = None):
        """
        Initialize timeframe expert.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            feedforward_dim: Feedforward dimension (default: hidden_dim * 4)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer layers with configurable feedforward dimension
        feedforward_dim = feedforward_dim or (hidden_dim * 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Multiple output heads with proper transformer structure
        # Price change prediction head (logits for probability distribution)
        self.price_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Raw logits, no activation
        )
        
        # Direction classification head (logits)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Direction classification (logits, no activation)
        )
        
        # Volatility prediction head (logits)
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Raw logits, no activation
        )
        
        # Price magnitude prediction head (logits)
        self.magnitude_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Raw logits, no activation
        )
        
        # Price percentile prediction head (logits)
        self.percentile_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Raw logits, no activation
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through expert network.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Dictionary with predictions
        """
        # Project input to hidden dimension
        x = self.input_projection(x)  # [batch, seq_len, hidden_dim]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer layers
        x = self.transformer(x)  # [batch, seq_len, hidden_dim]
        
        # Use last timestep for prediction
        x = x[:, -1, :]  # [batch, hidden_dim]
        
        # Apply layer norm and dropout
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Generate predictions as raw logits (proper transformer structure)
        price_change_logits = self.price_head(x)  # [batch, 1] - raw logits
        direction_logits = self.direction_head(x)  # [batch, 2] - raw logits
        volatility_logits = self.volatility_head(x)  # [batch, 1] - raw logits
        magnitude_logits = self.magnitude_head(x)  # [batch, 1] - raw logits
        percentile_logits = self.percentile_head(x)  # [batch, 1] - raw logits
        
        return {
            'price_change_logits': price_change_logits.squeeze(-1),  # [batch] - raw logits
            'direction_logits': direction_logits,      # [batch, 2] - raw logits
            'volatility_logits': volatility_logits.squeeze(-1),  # [batch] - raw logits
            'magnitude_logits': magnitude_logits.squeeze(-1),  # [batch] - raw logits
            'percentile_logits': percentile_logits.squeeze(-1)  # [batch] - raw logits
        }


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class GatingNetwork(nn.Module):
    """
    Gating network to determine expert weights based on input features.
    """
    
    def __init__(self, 
                 input_dim: int,
                 num_experts: int = 5,
                 hidden_dim: int = 128):
        """
        Initialize gating network.
        
        Args:
            input_dim: Input feature dimension
            num_experts: Number of experts
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_experts)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute expert weights.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            
        Returns:
            Expert weights [batch_size, num_experts]
        """
        # Use global average pooling over sequence dimension
        x_pooled = x.mean(dim=1)  # [batch_size, input_dim]
        
        # Compute logits and apply softmax
        logits = self.network(x_pooled)  # [batch_size, num_experts]
        weights = F.softmax(logits, dim=-1)
        
        return weights


class MoECryptoPredictor(nn.Module):
    """
    Main MoE model for cryptocurrency prediction across multiple timeframes.
    """
    
    def __init__(self, 
                 input_dim: int,
                 timeframes: List[str] = None,
                 expert_config: Dict = None,
                 use_gating: bool = True):
        """
        Initialize MoE model.
        
        Args:
            input_dim: Input feature dimension
            timeframes: List of timeframe identifiers
            expert_config: Configuration for expert networks
            use_gating: Whether to use gating network
        """
        super().__init__()
        
        self.timeframes = timeframes or ['5m', '30m', '1h', '1d', '1w']
        self.num_experts = len(self.timeframes)
        self.use_gating = use_gating
        
        # Default expert configuration
        default_config = {
            'hidden_dim': 256,
            'num_layers': 4,
            'num_heads': 8,
            'dropout': 0.1
        }
        self.expert_config = {**default_config, **(expert_config or {})}
        
        # Create expert networks
        self.experts = nn.ModuleDict()
        for tf in self.timeframes:
            self.experts[tf] = TimeframeExpert(
                input_dim=input_dim,
                **self.expert_config
            )
        
        # Gating network (optional)
        if self.use_gating:
            self.gating_network = GatingNetwork(
                input_dim=input_dim,
                num_experts=self.num_experts
            )
        
        # Final aggregation layers for logits
        self.final_price_layer = nn.Linear(self.num_experts, 1)
        self.final_direction_layer = nn.Linear(self.num_experts * 2, 2)
        self.final_volatility_layer = nn.Linear(self.num_experts, 1)
        self.final_magnitude_layer = nn.Linear(self.num_experts, 1)
        self.final_percentile_layer = nn.Linear(self.num_experts, 1)
    
    def forward(self, 
                inputs: Dict[str, torch.Tensor],
                target_timeframe: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MoE model.
        
        Args:
            inputs: Dictionary mapping timeframes to input tensors
            target_timeframe: Specific timeframe to predict for (optional)
            
        Returns:
            Dictionary with aggregated predictions
        """
        expert_outputs = {}
        
        # Get predictions from each expert
        for tf in self.timeframes:
            if tf in inputs:
                expert_outputs[tf] = self.experts[tf](inputs[tf])
        
        if not expert_outputs:
            raise ValueError("No valid inputs provided")
        
        # If targeting specific timeframe, return that expert's output
        if target_timeframe and target_timeframe in expert_outputs:
            return expert_outputs[target_timeframe]
        
        # Aggregate expert outputs
        batch_size = next(iter(expert_outputs.values()))['price_change_logits'].size(0)
        
        # Collect predictions (all as logits)
        price_predictions = []
        direction_predictions = []
        volatility_predictions = []
        magnitude_predictions = []
        percentile_predictions = []
        
        for tf in self.timeframes:
            if tf in expert_outputs:
                price_predictions.append(expert_outputs[tf]['price_change_logits'].unsqueeze(1))
                direction_predictions.append(expert_outputs[tf]['direction_logits'])
                volatility_predictions.append(expert_outputs[tf]['volatility_logits'].unsqueeze(1))
                magnitude_predictions.append(expert_outputs[tf]['magnitude_logits'].unsqueeze(1))
                percentile_predictions.append(expert_outputs[tf]['percentile_logits'].unsqueeze(1))
            else:
                # Fill with zeros if timeframe not available
                device = next(iter(expert_outputs.values()))['price_change_logits'].device
                price_predictions.append(torch.zeros(batch_size, 1, device=device))
                direction_predictions.append(torch.zeros(batch_size, 2, device=device))
                volatility_predictions.append(torch.zeros(batch_size, 1, device=device))
                magnitude_predictions.append(torch.zeros(batch_size, 1, device=device))
                percentile_predictions.append(torch.zeros(batch_size, 1, device=device))
        
        # Stack predictions
        price_stack = torch.cat(price_predictions, dim=1)  # [batch, num_experts]
        direction_stack = torch.cat(direction_predictions, dim=1)  # [batch, num_experts * 2]
        volatility_stack = torch.cat(volatility_predictions, dim=1)  # [batch, num_experts]
        magnitude_stack = torch.cat(magnitude_predictions, dim=1)  # [batch, num_experts]
        percentile_stack = torch.cat(percentile_predictions, dim=1)  # [batch, num_experts]
        
        # Apply gating if enabled
        if self.use_gating and len(expert_outputs) > 1:
            # Use the first available input for gating
            gating_input = next(iter(inputs.values()))
            expert_weights = self.gating_network(gating_input)  # [batch, num_experts]
            
            # Apply weights to all predictions
            price_weighted = (price_stack * expert_weights).sum(dim=1)
            volatility_weighted = (volatility_stack * expert_weights).sum(dim=1)
            magnitude_weighted = (magnitude_stack * expert_weights).sum(dim=1)
            percentile_weighted = (percentile_stack * expert_weights).sum(dim=1)
            
            # For direction, apply weights to each class separately
            direction_weights = expert_weights.repeat_interleave(2, dim=1)
            direction_weighted = self.final_direction_layer(direction_stack * direction_weights)
        else:
            # Simple aggregation
            price_weighted = self.final_price_layer(price_stack).squeeze(-1)
            direction_weighted = self.final_direction_layer(direction_stack)
            volatility_weighted = self.final_volatility_layer(volatility_stack).squeeze(-1)
            magnitude_weighted = self.final_magnitude_layer(magnitude_stack).squeeze(-1)
            percentile_weighted = self.final_percentile_layer(percentile_stack).squeeze(-1)
        
        return {
            'price_change_logits': price_weighted,
            'direction_logits': direction_weighted,
            'volatility_logits': volatility_weighted,
            'magnitude_logits': magnitude_weighted,
            'percentile_logits': percentile_weighted,
            'expert_outputs': expert_outputs,
            'expert_weights': expert_weights if self.use_gating else None
        }
    
    def get_expert_predictions(self, 
                             inputs: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get individual expert predictions without aggregation.
        
        Args:
            inputs: Dictionary mapping timeframes to input tensors
            
        Returns:
            Dictionary mapping timeframes to expert predictions
        """
        expert_outputs = {}
        
        for tf in self.timeframes:
            if tf in inputs:
                with torch.no_grad():
                    expert_outputs[tf] = self.experts[tf](inputs[tf])
        
        return expert_outputs


class MoELoss(nn.Module):
    """
    Multi-task loss function for MoE model.
    """
    
    def __init__(self, 
                 price_weight: float = 1.0,
                 direction_weight: float = 0.5,
                 volatility_weight: float = 0.3,
                 magnitude_weight: float = 0.2,
                 percentile_weight: float = 0.2,
                 diversity_weight: float = 0.1):
        """
        Initialize loss function.
        
        Args:
            price_weight: Weight for price prediction loss
            direction_weight: Weight for direction classification loss
            volatility_weight: Weight for volatility prediction loss
            magnitude_weight: Weight for magnitude prediction loss
            percentile_weight: Weight for percentile prediction loss
            diversity_weight: Weight for expert diversity regularization
        """
        super().__init__()
        
        self.price_weight = price_weight
        self.direction_weight = direction_weight
        self.volatility_weight = volatility_weight
        self.magnitude_weight = magnitude_weight
        self.percentile_weight = percentile_weight
        self.diversity_weight = diversity_weight
        
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary with loss components
        """
        losses = {}
        
        # Price prediction loss (MSE on logits)
        if 'price_change_logits' in predictions and 'price_change' in targets:
            pred_price_logits = predictions['price_change_logits']
            target_price = targets['price_change']
            
            # Ensure same shape
            if target_price.dim() > 1:
                target_price = target_price.squeeze(-1)
            if pred_price_logits.dim() > 1:
                pred_price_logits = pred_price_logits.squeeze(-1)
                
            losses['price_loss'] = self.mse_loss(pred_price_logits, target_price)
        
        # Direction classification loss (CrossEntropy)
        if 'direction_logits' in predictions and 'direction' in targets:
            pred_direction = predictions['direction_logits']
            target_direction = targets['direction']
            
            # Ensure target is 1D
            if target_direction.dim() > 1:
                target_direction = target_direction.squeeze(-1)
                
            losses['direction_loss'] = self.ce_loss(
                pred_direction,
                target_direction.long()
            )
        
        # Volatility prediction loss (MSE on logits)
        if 'volatility_logits' in predictions and 'volatility' in targets:
            pred_vol_logits = predictions['volatility_logits']
            target_vol = targets['volatility']
            
            # Ensure same shape
            if target_vol.dim() > 1:
                target_vol = target_vol.squeeze(-1)
            if pred_vol_logits.dim() > 1:
                pred_vol_logits = pred_vol_logits.squeeze(-1)
                
            losses['volatility_loss'] = self.mse_loss(pred_vol_logits, target_vol)
        
        # Magnitude prediction loss (MSE on logits)
        if 'magnitude_logits' in predictions and 'magnitude' in targets:
            pred_magnitude_logits = predictions['magnitude_logits']
            target_magnitude = targets['magnitude']
            
            # Ensure same shape
            if target_magnitude.dim() > 1:
                target_magnitude = target_magnitude.squeeze(-1)
            if pred_magnitude_logits.dim() > 1:
                pred_magnitude_logits = pred_magnitude_logits.squeeze(-1)
                
            losses['magnitude_loss'] = self.mse_loss(pred_magnitude_logits, target_magnitude)
        
        # Percentile prediction loss (MSE on logits)
        if 'percentile_logits' in predictions and 'percentile' in targets:
            pred_percentile_logits = predictions['percentile_logits']
            target_percentile = targets['percentile']
            
            # Ensure same shape
            if target_percentile.dim() > 1:
                target_percentile = target_percentile.squeeze(-1)
            if pred_percentile_logits.dim() > 1:
                pred_percentile_logits = pred_percentile_logits.squeeze(-1)
                
            losses['percentile_loss'] = self.mse_loss(pred_percentile_logits, target_percentile)
        
        # Expert diversity regularization
        if 'expert_weights' in predictions and predictions['expert_weights'] is not None:
            # Encourage diversity in expert usage
            weights = predictions['expert_weights']
            entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=1)
            losses['diversity_loss'] = -entropy.mean()  # Negative because we want to maximize entropy
        
        # Total loss
        total_loss = 0
        if 'price_loss' in losses:
            total_loss += self.price_weight * losses['price_loss']
        if 'direction_loss' in losses:
            total_loss += self.direction_weight * losses['direction_loss']
        if 'volatility_loss' in losses:
            total_loss += self.volatility_weight * losses['volatility_loss']
        if 'magnitude_loss' in losses:
            total_loss += self.magnitude_weight * losses['magnitude_loss']
        if 'percentile_loss' in losses:
            total_loss += self.percentile_weight * losses['percentile_loss']
        if 'diversity_loss' in losses:
            total_loss += self.diversity_weight * losses['diversity_loss']
        
        losses['total_loss'] = total_loss
        
        return losses


def create_moe_model(input_dim: int, 
                    timeframes: List[str] = None,
                    **kwargs) -> MoECryptoPredictor:
    """
    Factory function to create MoE model.
    
    Args:
        input_dim: Input feature dimension
        timeframes: List of timeframe identifiers
        **kwargs: Additional model configuration
        
    Returns:
        Initialized MoE model
    """
    return MoECryptoPredictor(
        input_dim=input_dim,
        timeframes=timeframes,
        **kwargs
    )


if __name__ == "__main__":
    # Test model creation and forward pass
    input_dim = 50
    seq_len = 100
    batch_size = 32
    timeframes = ['5m', '30m', '1h', '1d', '1w']
    
    # Create model
    model = create_moe_model(input_dim, timeframes)
    
    # Create dummy inputs
    inputs = {}
    for tf in timeframes:
        inputs[tf] = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    outputs = model(inputs)
    
    print("Model output shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"{key}: {len(value)} experts")
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
