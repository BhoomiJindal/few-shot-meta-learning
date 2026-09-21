import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call


class MAMLModel(nn.Module):
    """
    Combines the shared Conv4 backbone with a linear classification head
    to form the full network MAML adapts. Both the backbone and the head
    are meta-learned together; the head's output size equals n_way for
    whichever episode this model instance is used on.
    """

    def __init__(self, backbone, n_way):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.output_dim, n_way)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class MAML(nn.Module):
    """
    Model-Agnostic Meta-Learning (Finn, Abbeel, Levine, 2017).

    Wraps a MAMLModel. For each episode:
      - Inner loop (`adapt`): starting from the current meta-parameters,
        takes `inner_steps` gradient descent steps on the support set,
        producing task-adapted parameters -- WITHOUT touching the model's
        real stored parameters. Uses functional_call + autograd.grad so
        the whole process stays differentiable with respect to the
        original meta-parameters.
      - Outer loop (handled by the training loop that calls this class):
        query-set loss, computed using the adapted parameters, is
        backpropagated -- this updates the ORIGINAL meta-parameters,
        which is what actually teaches the model to adapt quickly to
        new tasks, rather than teaching it to solve one specific task.
    """

    def __init__(self, model, inner_lr=0.01, inner_steps=5, first_order=False):
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        # first_order=True gives FOMAML (Nichol et al.): cheaper, skips
        # second-order gradient terms, small accuracy cost. Reference [6]
        # in our report covers this trade-off.
        self.first_order = first_order

    def adapt(self, support_images, support_labels):
        params = {name: p for name, p in self.model.named_parameters()}

        for _ in range(self.inner_steps):
            logits = functional_call(self.model, params, (support_images,))
            loss = F.cross_entropy(logits, support_labels)

            grads = torch.autograd.grad(
                loss,
                params.values(),
                create_graph=not self.first_order,
            )

            params = {
                name: p - self.inner_lr * g
                for (name, p), g in zip(params.items(), grads)
            }

        return params

    def forward(self, support_images, support_labels, query_images):
        """
        Runs one full MAML episode: adapt on the support set, then evaluate
        (functionally, using the adapted parameters) on the query set.
        Returns query logits -- the caller computes query loss and calls
        .backward() on it to trigger the outer-loop meta-update.
        """
        adapted_params = self.adapt(support_images, support_labels)
        query_logits = functional_call(self.model, adapted_params, (query_images,))
        return query_logits