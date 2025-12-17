from model import *
from matplotlib import pyplot as plt

def pretty_confidence_label(r):
    # New mapping: R=0 -> Confident, R=1 -> Insecure
    return 'Confident' if r == R.CONFIDENT else 'Insecure'

def display_us(tag, us, support, side=1, err=None, kind=True, xtick_labels=None):
    plt.title(tag)
    if xtick_labels is None:
        plt.xticks(range(4), ['both', 'insuff', 'diff', 'none'])
    else:
        plt.xticks(range(4), xtick_labels)
    if kind and side == 1:
        color = 'firebrick'
    elif kind and side == -1:
        color = 'lightsalmon'
    elif not kind and side == 1:
        color = 'darkblue'
    elif not kind and side == -1:
        color = 'lightskyblue'
    plt.bar(range(4), us[support], align='edge', width=0.3 * side, yerr=err if err is None else err[support] * 2, capsize=2, color=color, label='synthetic' if side == -1 else 'model')
    plt.ylim(0, 1)
    # Apply per-panel x-axis labels if provided, else default
    if xtick_labels is None:
        plt.xticks(range(4), ['both', 'insuff', 'diff', 'none'])
    else:
        plt.xticks(range(4), xtick_labels)

def add_badge(i, kind=True):
    c = plt.Rectangle((2.6, 0.65), 0.8, 0.3, color='peachpuff' if kind else 'deepskyblue')
    plt.gca().add_patch(c)
    plt.text(3.0, 0.8, f'{i}', fontsize=20, ha='center', va='center')
    if i == 12: plt.legend()

def make_graph_model_only(us_model, title, kind):
    """Plot only the model distributions, without human data overlay.

    Mirrors the panel layout of make_graph but renders the model bars alone.
    """
    plt.figure(figsize=(8, 9))
    i = 0
    for r in R:
        for f in F:
            i += 1; plt.subplot(4, 3, i)
            support = us_model[f, r, 1, 0, 1] > 0
            display_us(f"{pretty_confidence_label(r)}/{f.name}\nsufficient, difficult", us_model[f, r, 1, 0, 1], support, kind=kind, xtick_labels=['both', '> avg', 'diff', 'none'])
            add_badge(i, kind=kind)
            i += 1; plt.subplot(4, 3, i)
            support = us_model[f, r, 1, 1, 1] > 0
            display_us(f"{pretty_confidence_label(r)}/{f.name}\ninsufficient, difficult", us_model[f, r, 1, 1, 1], support, kind=kind, xtick_labels=['both', '< avg', 'diff', 'none'])
            add_badge(i, kind=kind)
            i += 1; plt.subplot(4, 3, i)
            support = us_model[f, r, 1, 1, 0] > 0
            display_us(f"{pretty_confidence_label(r)}/{f.name}\ninsufficient, easy", us_model[f, r, 1, 1, 0], support, kind=kind, xtick_labels=['both', '< avg', 'easy', 'none'])
            add_badge(i, kind=kind)

    plt.suptitle(title, fontsize='xx-large')
    plt.tight_layout()
# 1, 0, 1
# 1, 1, 0
# 0, 1, 0