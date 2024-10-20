"""Reimplementation of the ACTINN cell-type annotation method.

Reference
---------
Ma, Feiyang, and Matteo Pellegrini. "ACTINN: automated identification of cell types in single cell RNA sequencing."
Bioinformatics 36.2 (2020): 533-538.

"""
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor
import logging
import itertools
from dance import logger

from dance.models.nn import VanillaMLP
from dance.modules.base import BaseClassificationMethod
from dance.transforms import AnnDataTransform, Compose, FilterGenesPercentile, SetConfig
from dance.typing import LogLevel, Optional, Tuple

class FeatureEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, aggregation: str = "mean"):
        super().__init__()
        
        self.embedding = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.aggregation = aggregation

    def forward(self, x: Tensor):
        # num_genes = x.shape[1]
        # gene_indices = torch.arange(num_genes).to(x.device)
        # gene_indices = gene_indices.repeat(x.shape[0], 1)
        batch_dim = x.shape[0]
        gene_embedding = self.embedding.unsqueeze(0).expand(batch_dim, -1, -1)
        x = x.unsqueeze(-1) * gene_embedding # scale gene expression by gene embedding
        x = x + self.bias

        if self.aggregation == "mean":
            x = x.mean(dim=1)
        elif self.aggregation == "sum":
            x = x.sum(dim=1)
        elif self.aggregation == "max":
            x = x.max(dim=1).values
        else:
            raise ValueError(f"Invalid aggregation method: {self.aggregation}")

        return x
    

class FeatureEmbeddingMLP(nn.Module):
    """Vanilla multilayer perceptron with ReLU activation.

    Parameters
    ----------
    input_dim
        Input feature dimension.
    output_dim
        Output dimension.
    hidden_dims
        Hidden layer dimensions.
    device
        Computation device.
    random_seed
        Random seed controlling the model weights initialization.

    """

    def __init__(self, input_dim: int, output_dim: int, *, hidden_dims: Tuple[int, ...] = (100, 50, 25),
                 device: str = "cpu", random_seed: Optional[int] = None, aggregation: str = "mean"):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.device = device
        self.random_seed = random_seed

        embedding = FeatureEmbedding(input_dim, hidden_dims[0], aggregation=aggregation)
        self.aggregation = aggregation

        if len(hidden_dims) == 1:
            self.model = nn.Sequential(
                embedding,
                nn.Linear(hidden_dims[0], output_dim),
            ).to(device)
        else:
            self.model = nn.Sequential(
                embedding,
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                *itertools.chain.from_iterable(
                    zip(
                        map(nn.Linear, hidden_dims[1:-1], hidden_dims[2:]),
                        itertools.repeat(nn.ReLU()),
                    )),
                nn.Linear(hidden_dims[-1], output_dim),
            ).to(device)

        self.initialize_parameters()
        logger.debug(f"Initialized model:\n{self.model}")

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def initialize_parameters(self):
        """Initialize parameters."""
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)

        for i in range(0, len(self.model), 2):
            if isinstance(self.model[i], nn.Linear):
                nn.init.xavier_normal_(self.model[i].weight)
                self.model[i].bias[:] = 0


class ACTINN(BaseClassificationMethod):
    """The ACTINN cell-type classification model.

    Parameters
    ----------
    hidden_dims
        Hidden layer dimensions.
    lambd
        Regularization parameter
    device
        Training device

    """

    def __init__(
        self,
        *,
        hidden_dims: Tuple[int, ...] = (100, 50, 25),
        lambd: float = 0.01,
        device: str = "cpu",
        random_seed: Optional[int] = None,
        use_feature_embedding: bool = False,
        use_pretrained_gene_embedding: bool = False,
    ):
        super().__init__()

        self.hidden_dims = hidden_dims
        self.lambd = lambd
        self.device = device
        self.random_seed = random_seed

        self.model_size = len(hidden_dims) + 2

        self.use_feature_embedding = use_feature_embedding
        self.use_pretrained_gene_embedding = use_pretrained_gene_embedding
        self.gene_names = []

    @staticmethod
    def preprocessing_pipeline(normalize: bool = True, filter_genes: bool = True, log_level: LogLevel = "INFO"):
        transforms = []

        if normalize:
            transforms.append(AnnDataTransform(sc.pp.normalize_total, target_sum=1e4))
            transforms.append(AnnDataTransform(sc.pp.log1p, base=2))

        if filter_genes:
            transforms.append(AnnDataTransform(sc.pp.filter_genes, min_cells=1))
            transforms.append(FilterGenesPercentile(min_val=1, max_val=99, mode="sum"))
            transforms.append(FilterGenesPercentile(min_val=1, max_val=99, mode="cv"))

        transforms.append(SetConfig({"label_channel": "cell_type"}))

        return Compose(*transforms, log_level=log_level)

    def compute_loss(self, z: Tensor, y: Tensor):
        """Compute loss function.

        Parameters
        ----------
        z
            Output of forward propagation (cells by cell-types).
        y
            Cell labels (cells).

        Returns
        -------
        torch.Tensor
            Loss.

        """
        log_prob = F.log_softmax(z, dim=-1)
        loss = nn.NLLLoss()(log_prob, y)
        for i, p in enumerate(self.model.model):
            if isinstance(p, nn.Linear):
                loss += self.lambd * (p.weight**2).sum() / 2
        return loss

    def random_batches(self, x: Tensor, y: Tensor, batch_size: int = 32, seed: Optional[int] = None):
        """Shuffle data and split into batches.

        Parameters
        ----------
        x
            Training data (cells by genes).
        y
            True labels (cells by cell-types).

        Yields
        ------
        Tuple[torch.Tensor, torch.Tensor]
            Batch of training data (x, y).

        """
        ns = x.shape[0]
        perm = np.random.default_rng(seed).permutation(ns).tolist()
        slices = [perm[i:i + batch_size] for i in range(0, ns, batch_size)]
        yield from map(lambda slice_: (x[slice_], y[slice_]), slices)

    def fit(
        self,
        x_train: Tensor,
        y_train: Tensor,
        *,
        batch_size: int = 128,
        lr: float = 0.01,
        num_epochs: int = 50,
        print_cost: bool = False,
        seed: Optional[int] = None,
    ):
        """Fit the classifier.

        Parameters
        ----------
        x_train
            training data (cells by genes).
        y_train
            training labels (cells by cell-types).
        batch_size
            Training batch size.
        lr
            Initial learning rate.
        num_epochs
            Number of epochs to run.
        print_cost
            Print training loss if set to True.
        seed
            Random seed, if set to None, then random.

        """
        input_dim, output_dim = x_train.shape[1], y_train.shape[1]
        x_train = x_train.clone().detach().float().to(self.device)  # cells by genes
        y_train = torch.where(y_train)[1].to(self.device)  # cells

        # Initialize weights, optimizer, and scheduler
        if self.use_feature_embedding:
            self.model = FeatureEmbeddingMLP(
                input_dim, output_dim, hidden_dims=self.hidden_dims, device=self.device, aggregation="sum")
            if self.use_pretrained_gene_embedding:
                assert len(self.gene_names) == input_dim, "Gene names must be provided for pretrained gene embedding."
            
        else:
            self.model = VanillaMLP(input_dim, output_dim, hidden_dims=self.hidden_dims, device=self.device)


        print(f"Model: {self.model}")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

        # Start training loop
        global_steps = 0
        for epoch in range(num_epochs):
            epoch_seed = seed if seed is None else seed + epoch
            batches = self.random_batches(x_train, y_train, batch_size, epoch_seed)

            tot_cost = tot_size = 0
            for batch_x, batch_y in tqdm(batches, total=len(x_train) // batch_size):
                batch_cost = self.compute_loss(self.model(batch_x), batch_y)
                tot_cost += batch_cost.item()
                tot_size += 1

                optimizer.zero_grad()
                batch_cost.backward()
                optimizer.step()

                global_steps += 1
                if global_steps % 1000 == 0:
                    lr_scheduler.step()

            if print_cost and (epoch % 10 == 0):
                print(f"Epoch: {epoch:>4d} Loss: {tot_cost / tot_size:6.4f}")

    @torch.no_grad()
    def predict(self, x: Tensor):
        """Predict cell labels.

        Parameters
        ----------
        x
            Gene expression input features (cells by genes).

        Returns
        -------
        torch.Tensor
            Predicted cell-label indices.

        """
        x = x.clone().detach().to(self.device)

        batch_size = 128
        num_batches = x.shape[0] // batch_size
        if x.shape[0] % batch_size:
            num_batches += 1

        for i in range(num_batches):
            batch_x = x[i * batch_size:(i + 1) * batch_size]
            z = self.model(batch_x)
            if i == 0:
                predictions = torch.argmax(z, dim=-1)
            else:
                predictions = torch.cat((predictions, torch.argmax(z, dim=-1)))

        # z = self.model(x)
        # prediction = torch.argmax(z, dim=-1)
        
        return predictions
