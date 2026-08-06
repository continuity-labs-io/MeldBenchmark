import json
import torch
import numpy as np

import logging
logger = logging.getLogger(__name__)

class ThermodynamicAutopsyEngine:
    def __init__(self, model, feature_names=None):
        self.model = model
        if feature_names is None:
            self.feature_names = self._generate_feature_names()
        else:
            self.feature_names = feature_names
            
    def _generate_feature_names(self):
        names = []
        for i in range(1, 101):
            names.append(f"Sigma_PC{i:03d}")
        
        rna_anchors = ["Psi_NFE2L2", "Psi_TP53", "Psi_CDKN2A", "Psi_TREM2", 
                       "Psi_APOE", "Psi_IL6", "Psi_GFAP", "Psi_MAPT", 
                       "Psi_NANOG", "Psi_CASP3", "Psi_CAS13", "Psi_GAPDH"]
        names.extend(rna_anchors)
        
        voltage_tracks = ["Omega_VoltRed", "Omega_VoltGrn"]
        names.extend(voltage_tracks)
        return names

    def generate_autopsy(self, x_sequence, crash_time_step, confidence_score=0.98):
        # x_sequence: [1, Time, 114]
        
        # 1. Compute first-order Taylor attribution
        if hasattr(self.model, "compute_attribution"):
            attribution_matrix = self.model.compute_attribution(x_sequence, crash_time_step)
        else:
            x_req = x_sequence.clone().detach().requires_grad_(True)
            predicted_state, _ = self.model(x_req)
            target_state_sum = predicted_state[:, crash_time_step, :].sum()
            gradients = torch.autograd.grad(target_state_sum, x_req, retain_graph=True)[0]
            attribution_matrix = x_req.detach() * gradients
            
        # attribution_matrix is [1, Time, 114]
        attr_np = attribution_matrix[0].detach().cpu().numpy() # [Time, 114]
        
        # 2. Identify the primary latent driver dimension with the highest global variance or gradient magnitude.
        # Compute magnitude across the sequence leading up to the crash
        seq_attr = attr_np[:crash_time_step+1, :]
        feature_magnitudes = np.abs(seq_attr).sum(axis=0) # Sum of magnitudes over time
        primary_driver_idx = int(np.argmax(feature_magnitudes))
        primary_latent_driver = self.feature_names[primary_driver_idx]
        
        # 3. Extract the top 3 critical time steps prior to the crash.
        time_step_magnitudes = np.abs(seq_attr).sum(axis=1) # [Time]
        
        prior_steps = np.arange(crash_time_step)
        if len(prior_steps) == 0:
            top_3_steps = []
        else:
            prior_mags = time_step_magnitudes[:crash_time_step]
            # Get indices of top 3
            num_steps_to_extract = min(3, len(prior_steps))
            top_3_indices = np.argsort(prior_mags)[-num_steps_to_extract:][::-1]
            top_3_steps = top_3_indices
        
        causal_trace = []
        for t in top_3_steps:
            step_attr = np.abs(attr_np[t, :])
            flagged_idx = int(np.argmax(step_attr))
            flagged_feature = self.feature_names[flagged_idx]
            relevance_score = float(step_attr[flagged_idx])
            
            # Normalize relevance score for display purposes (optional, but requested in prompt)
            # Dividing by the sum of absolute attributions at this time step
            total_step_attr = np.sum(step_attr)
            if total_step_attr > 0:
                normalized_score = relevance_score / float(total_step_attr)
            else:
                normalized_score = 0.0
            
            # Map mechanism
            if flagged_feature.startswith("Psi"):
                mechanism = "RNA stress alarm"
            elif flagged_feature.startswith("Omega"):
                mechanism = "Electrical baseline destabilization"
            elif flagged_feature.startswith("Sigma"):
                mechanism = "Morphological shape distortion"
            else:
                mechanism = "Unknown anomaly"
                
            causal_trace.append({
                "time_step": f"T={int(t)}",
                "flagged_input": flagged_feature,
                "relevance_score": round(normalized_score, 4),
                "mechanism": mechanism
            })
            
        return {
            "status": "CRITICAL_FAILURE_PREDICTED",
            "predicted_crash_time": f"T={int(crash_time_step)}",
            "confidence_score": confidence_score,
            "anomaly_ontology": {
                "primary_latent_driver": primary_latent_driver,
                "causal_trace": causal_trace
            }
        }

if __name__ == "__main__":
    import sys
    import os
    # Ensure src is in the path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    from src.models.ssm.neocortical_engine import NeocorticalEngine
    
    # Mock initialization
    model = NeocorticalEngine(input_dim=114, d_model=256, d_state=32)
    engine = ThermodynamicAutopsyEngine(model)
    
    # Dummy sequence (Batch=1, Time=50, Features=114)
    x_sequence = torch.randn(1, 50, 114)
    
    # Injected crash at T=45
    crash_time_step = 45
    
    # Generate autopsy report
    autopsy_report = engine.generate_autopsy(x_sequence, crash_time_step)
    
    logger.info(json.dumps(autopsy_report, indent=2))
