# DeepLense: Gravitational Lensing Substructure & PINN Detection

This project explores dark matter substructure detection in strong gravitational lensing simulations using deep learning architectures and physics-informed constraints.

## Project Structure

- `substructure_cnn_classifier/`
  - `efficientnet_substructure_classifier.ipynb`: Transfer learning pipeline fine-tuning EfficientNet-B0 and custom CNNs for 3-class dark matter substructure classification (no substructure, spherical vortex, extended density).
  - `vit_attention_experiments.ipynb`: Self-attention and Vision Transformer patch embedding experiments for single-channel lensing maps.

- `physics_informed_pinn/`
  - `pinn_poisson_solver.ipynb`: Physics-Informed Neural Network (PINN) implementation. Embeds the 2D Poisson equation ($\nabla^2 I \approx 2\kappa$) directly into the loss function to enforce consistency between observed surface brightness $I$ and mass convergence $\kappa$.

- `data_simulations/`
  - Contains simulated gravitational lensing `.npy` samples generated for training and evaluation. (Large simulation archives ignored via `.gitignore`).

## Key Methodologies

1. **Substructure Classification**: Custom CNNs and EfficientNet-B0 trained on simulated single-channel lensing images with class imbalance handling.
2. **Physics Regularization**: Added spatial finite-difference approximation of the Poisson operator to the mean squared error loss, reducing unphysical mass prediction artifacts.
3. **Vision Transformer (DeiT-Tiny) Adaptation**: Modified input projections to accept 1-channel astrophysical maps.
