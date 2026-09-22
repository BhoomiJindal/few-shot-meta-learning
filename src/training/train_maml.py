import torch
import torch.nn.functional as F


def train_maml(maml, train_sampler, val_sampler, num_episodes=100, val_every=20, outer_lr=0.001):
    """
    Trains a MAML model. Each episode is one outer-loop step:
      1. maml(...) runs the inner-loop adaptation and returns query logits
         computed with the adapted parameters (see MAML.forward).
      2. Cross-entropy loss on those query logits is the OUTER loss.
      3. .backward() on the outer loss propagates gradients through the
         entire inner-loop adaptation, back into maml.model's real
         parameters (verified working in Step 7.5).
      4. The outer optimizer updates maml.model's real parameters --
         this is the actual "meta-learning" step.
    """
    outer_optimizer = torch.optim.Adam(maml.model.parameters(), lr=outer_lr)

    history = {"train_loss": [], "train_acc": [], "val_episode": [], "val_acc": []}

    for episode in range(1, num_episodes + 1):
        maml.model.train()
        s_img, s_lab, q_img, q_lab = train_sampler.sample_episode()

        query_logits = maml(s_img, s_lab, q_img)
        outer_loss = F.cross_entropy(query_logits, q_lab)

        outer_optimizer.zero_grad()
        outer_loss.backward()
        outer_optimizer.step()

        acc = (query_logits.argmax(dim=1) == q_lab).float().mean().item()
        history["train_loss"].append(outer_loss.item())
        history["train_acc"].append(acc)

        if episode % val_every == 0:
            val_acc = evaluate_maml(maml, val_sampler, num_episodes=20)
            history["val_episode"].append(episode)
            history["val_acc"].append(val_acc)
            print(f"Episode {episode:4d} | outer loss {outer_loss.item():.3f} | "
                  f"train acc {acc:.3f} | val acc (20 ep avg) {val_acc:.3f}")

    return history


def evaluate_maml(maml, sampler, num_episodes=50):
    """
    Average query accuracy over num_episodes freshly sampled episodes.

    Important: we do NOT wrap this in torch.no_grad() like we did for
    ProtoNet evaluation. MAML's inner-loop adaptation needs gradients
    (autograd.grad inside .adapt()) even at evaluation time, since
    "evaluating on a new task" still means adapting to it first via
    gradient steps. We just don't call outer_optimizer.step() here, so
    the model's real parameters are never updated during evaluation.
    """
    accs = []
    for _ in range(num_episodes):
        s_img, s_lab, q_img, q_lab = sampler.sample_episode()
        query_logits = maml(s_img, s_lab, q_img)
        acc = (query_logits.argmax(dim=1) == q_lab).float().mean().item()
        accs.append(acc)
    return sum(accs) / len(accs)