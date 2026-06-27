"""
Implementação simples de Grad-CAM para modelos CNN 2D em PyTorch.

Este módulo é usado para interpretar o baseline CNN 2D do projeto.

Entradas esperadas
------------------
- Modelo PyTorch treinado.
- Tensor de imagem no formato [1, C, H, W].
- Nome da camada convolucional alvo ou descoberta automática da última Conv2d.

Saídas
------
- Heatmap Grad-CAM normalizado no intervalo [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class GradCAMResult:
    """Resultado da aplicação do Grad-CAM."""

    heatmap: np.ndarray
    logit: float
    probability: float
    target_layer_name: str


def find_last_conv2d_layer(model: nn.Module) -> tuple[str, nn.Module]:
    """Encontra automaticamente a última camada Conv2d do modelo."""
    conv_layers: list[tuple[str, nn.Module]] = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))

    if not conv_layers:
        raise ValueError("Nenhuma camada Conv2d encontrada no modelo.")

    return conv_layers[-1]


class GradCAM:
    """
    Implementação de Grad-CAM para classificação binária.

    Para o baseline CNN 2D, o modelo retorna um logit. O Grad-CAM é calculado
    em relação a esse logit positivo, associado à classe Hepatic_Steatosis.
    """

    def __init__(self, model: nn.Module, target_layer_name: str | None = None) -> None:
        self.model = model.eval()
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        if target_layer_name is None:
            self.target_layer_name, self.target_layer = find_last_conv2d_layer(model)
        else:
            modules = dict(model.named_modules())
            if target_layer_name not in modules:
                raise ValueError(f"Camada não encontrada: {target_layer_name}")
            self.target_layer_name = target_layer_name
            self.target_layer = modules[target_layer_name]

        self._forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor],
        grad_output: tuple[torch.Tensor],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        """Remove hooks registrados no modelo."""
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __call__(self, image_tensor: torch.Tensor) -> GradCAMResult:
        """
        Calcula Grad-CAM para uma imagem.

        Parâmetros
        ----------
        image_tensor:
            Tensor no formato [1, C, H, W].

        Retorno
        -------
        GradCAMResult:
            Heatmap normalizado, logit, probabilidade e nome da camada usada.
        """
        if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
            raise ValueError("image_tensor deve ter formato [1, C, H, W].")

        self.model.zero_grad(set_to_none=True)

        output = self.model(image_tensor)

        if output.ndim == 0:
            logit = output
        elif output.ndim == 1:
            logit = output[0]
        else:
            logit = output.reshape(-1)[0]

        probability = torch.sigmoid(logit)

        logit.backward()

        if self.activations is None:
            raise RuntimeError("Ativações não foram capturadas.")
        if self.gradients is None:
            raise RuntimeError("Gradientes não foram capturados.")

        activations = self.activations[0]
        gradients = self.gradients[0]

        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros_like(activations[0])

        for channel_idx, weight in enumerate(weights):
            cam += weight * activations[channel_idx]

        cam = torch.relu(cam)

        cam_min = cam.min()
        cam_max = cam.max()

        if torch.isclose(cam_max, cam_min):
            heatmap = torch.zeros_like(cam)
        else:
            heatmap = (cam - cam_min) / (cam_max - cam_min)

        return GradCAMResult(
            heatmap=heatmap.cpu().numpy(),
            logit=float(logit.detach().cpu()),
            probability=float(probability.detach().cpu()),
            target_layer_name=self.target_layer_name,
        )
