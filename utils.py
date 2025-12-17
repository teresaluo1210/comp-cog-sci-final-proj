from model import *
import model_utils
import analysis

def main_model_only(params, kind=True, ablation=model_utils.Ablations.FULL,
                    output_dir='figs/', file_tag='', show_plot=False):
    """Run the teacher model with provided params and plot model-only bars.

    - params: dict with keys like 'uc','urc','tc','tca','sc','scc','alpha','alphac',
              optionally 'p_both','p_either'.
    - kind: True for Tactful color scheme, False for Candid.
    - ablation: choose from model_utils.Ablations.* to disable components.
    - output_dir/file_tag: where to save and how to name the figure.
    - show_plot: whether to display the figure interactively.
    """
    name = 'kind' if kind else 'unkind'
    name_ = 'Tactful' if kind else 'Candid'

    us_model = model_utils.run_model(params, ablation=ablation)
    analysis.make_graph_model_only(us_model, f"{name_} (model-only)", kind=kind)
    import matplotlib.pyplot as plt
    plt.savefig(f"{output_dir}/bar-modelonly-{name}-{file_tag}.pdf")
    if show_plot:
        plt.show()
    plt.close()
    return params

