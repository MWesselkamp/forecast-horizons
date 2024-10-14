import torch

def test_forecast_shapes(model, clim_feats, met_feats, states):
    """
    Tests the shapes of all tensors in the forecast function of the LSTM regressor.
    Args:
    - model: the instance of the class containing the forecast method
    - clim_feats: tensor of climate features
    - met_feats: tensor of meteorological features
    - states: tensor of states
    
    Returns: None. Prints or asserts expected vs. actual tensor shapes.
    """
    # Run forecast function
    lead_time = states.shape[0] - model.lookback
    
    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Test y_h shape
        y_h = states[:model.lookback, ...].unsqueeze(0)
        print(f"y_h shape: {y_h.shape}")
        assert y_h.shape == torch.Size([1, model.lookback] + list(states.shape[1:])), \
            f"y_h shape is incorrect: {y_h.shape}"

        # Test X_static shape and its variations
        X_static = clim_feats.expand(model.lookback + lead_time, -1, -1)
        print(f"X_static shape: {X_static.shape}")
        assert X_static.shape == torch.Size([model.lookback + lead_time] + list(clim_feats.shape[1:])), \
            f"X_static shape is incorrect: {X_static.shape}"

        X_static_h = X_static[:model.lookback, ...].unsqueeze(0)
        print(f"X_static_h shape: {X_static_h.shape}")
        assert X_static_h.shape == torch.Size([1, model.lookback] + list(clim_feats.shape[1:])), \
            f"X_static_h shape is incorrect: {X_static_h.shape}"

        X_met_h = met_feats[:model.lookback, ...].unsqueeze(0)
        print(f"X_met_h shape: {X_met_h.shape}")
        assert X_met_h.shape == torch.Size([1, model.lookback] + list(met_feats.shape[1:])), \
            f"X_met_h shape is incorrect: {X_met_h.shape}"

        # Test future inputs
        X_static_f = X_static[model.lookback:(model.lookback + lead_time), ...].unsqueeze(0)
        print(f"X_static_f shape: {X_static_f.shape}")
        assert X_static_f.shape == torch.Size([1, lead_time] + list(clim_feats.shape[1:])), \
            f"X_static_f shape is incorrect: {X_static_f.shape}"

        X_met_f = met_feats[model.lookback:(model.lookback + lead_time), ...].unsqueeze(0)
        print(f"X_met_f shape: {X_met_f.shape}")
        assert X_met_f.shape == torch.Size([1, lead_time] + list(met_feats.shape[1:])), \
            f"X_met_f shape is incorrect: {X_met_f.shape}"

        # Check the output of the model's forward method
        preds = model.forward(X_static_h, X_static_f, X_met_h, X_met_f, y_h)
        print(f"Predictions shape: {preds.shape}")


