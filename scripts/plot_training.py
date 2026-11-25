import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_training_plot(history, output_path, ea_threshold):
    if not history:
        return

    epochs = [row["epoch"] for row in history]
    best = max(history, key=lambda row: row["hiou"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    panels = [
        ("Focal Loss", ("train_loss", "val_loss")),
        (f"Precision / Recall @ EA {ea_threshold:g}", ("precision", "recall")),
        (f"Val F1 @ EA {ea_threshold:g}", ("f1",)),
        ("Val HIoU", ("hiou",)),
    ]

    for ax, (title, keys) in zip(axes.flat, panels):
        for key in keys:
            ax.plot(epochs, [row[key] for row in history], label=key)
        ax.axvline(best["epoch"], color="gray", linestyle="--",
                   label=f"Best HIoU epoch {best['epoch']}")
        ax.set(title=title, xlabel="Epoch")
        ax.grid(alpha=0.2)
        ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)