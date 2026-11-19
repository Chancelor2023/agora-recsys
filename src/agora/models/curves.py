"""Training / validation log-loss curves for the Week 3 report (Q1)."""
from pathlib import Path


def plot_loss_curves(report, output):
    """Save a PNG of the per-epoch log loss; return a one-line summary."""
    history = report['history']
    epochs = [row['epoch'] for row in history]
    train = [row['train_log_loss'] for row in history]
    validation = [row['validation_log_loss'] for row in history if 'validation_log_loss' in row]
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return 'matplotlib is not installed: curves not plotted (history kept in the report)'
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot(epochs, train, label='train', color='#1f77b4')
    if validation:
        ax.plot(epochs, validation, label='validation', color='#ff7f0e')
        ax.axvline(report['best_epoch'], color='grey', linestyle='--', linewidth=1,
                   label=f"best epoch ({report['best_epoch']})")
    listwise = report.get('loss') == 'listwise'
    ax.axhline(report['prevalence_baseline_log_loss'], color='#1f77b4', linestyle=':', linewidth=1,
               label='uniform-ranking baseline (train)' if listwise else 'train prevalence baseline')
    ax.set_xlabel('epoch')
    ax.set_ylabel('listwise cross-entropy per feed' if listwise else 'log loss')
    title = report.get('title', 'Relevance model: training and validation log loss')
    if report.get('synthetic'):
        title += ' (synthetic data)'
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    last = history[-1]
    tail = f", validation {last['validation_log_loss']:.4f}" if validation else ''
    return f"Saved {output} ({len(history)} epochs, final train {last['train_log_loss']:.4f}{tail})"
