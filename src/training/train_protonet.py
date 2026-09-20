import torch
import torch.nn.functional as F


def train_protonet(protonet, train_sampler, val_sampler, n_way, num_episodes=100, val_every=20, lr=0.001):
    """
    Trains a PrototypicalNetwork by treating each sampled episode as one
    training step: embed support/query, compute prototype-based logits,
    backpropagate cross-entropy loss on the query predictions.

    Returns a dict of per-episode training loss/accuracy and periodic
    validation accuracy, so training progress can be plotted afterward.
    """
    optimizer = torch.optim.Adam(protonet.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_episode": [], "val_acc": []}

    for episode in range(1, num_episodes + 1):
        protonet.train()
        s_img, s_lab, q_img, q_lab = train_sampler.sample_episode()

        logits = protonet(s_img, s_lab, q_img, n_way)
        loss = F.cross_entropy(logits, q_lab)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (logits.argmax(dim=1) == q_lab).float().mean().item()
        history["train_loss"].append(loss.item())
        history["train_acc"].append(acc)

        if episode % val_every == 0:
            val_acc = evaluate_protonet(protonet, val_sampler, n_way, num_episodes=20)
            history["val_episode"].append(episode)
            history["val_acc"].append(val_acc)
            print(f"Episode {episode:4d} | train loss {loss.item():.3f} | "
                  f"train acc {acc:.3f} | val acc (20 ep avg) {val_acc:.3f}")

    return history


def evaluate_protonet(protonet, sampler, n_way, num_episodes=50):
    """Average accuracy over num_episodes freshly sampled episodes, no gradient updates."""
    protonet.eval()
    accs = []
    with torch.no_grad():
        for _ in range(num_episodes):
            s_img, s_lab, q_img, q_lab = sampler.sample_episode()
            logits = protonet(s_img, s_lab, q_img, n_way)
            acc = (logits.argmax(dim=1) == q_lab).float().mean().item()
            accs.append(acc)
    protonet.train()
    return sum(accs) / len(accs)