import numpy as np


def save_endpoint_visualization(path, image, reference, predicted, footer):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), dpi=140)
    for axis, points, color, title in zip(axes, (reference, predicted), ("#2ca02c", "#1f77b4"), ("GT", "Pred")):
        axis.imshow(image)
        first_x, first_y, second_x, second_y = [float(value) for value in points.tolist()]
        axis.plot([first_x, second_x], [first_y, second_y], color=color, linewidth=2.0)
        axis.scatter([first_x, second_x], [first_y, second_y], color=color, s=30)
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    figure.text(0.02, 0.02, "\n".join(footer), fontsize=9, va="bottom", ha="left")
    figure.tight_layout(rect=[0, 0.14, 1, 1])
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def save_report_visualizations(run, validation, test):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = run / "summary"
    destination.mkdir(parents=True, exist_ok=True)
    tasks = sorted(set(validation) | set(test))
    positions = np.arange(len(tasks))
    for metric, title in (("mae", "MAE"), ("rmse", "RMSE"), ("pearson_r", "Pearson r")):
        figure, axis = plt.subplots(figsize=(max(8, len(tasks) * 1.3), 4.8), dpi=140)
        axis.bar(positions - .38 / 2, [validation[task][metric] for task in tasks], width=.38, label="Validation", color="#4C72B0")
        axis.bar(positions + .38 / 2, [test[task][metric] for task in tasks], width=.38, label="Test", color="#DD8452")
        axis.set_xticks(positions)
        axis.set_xticklabels(tasks, rotation=25, ha="right")
        axis.set_ylabel(title)
        axis.set_title(f"Fusion Multi-Task {title} by Task")
        axis.legend()
        axis.grid(axis="y", linestyle="--", alpha=.25)
        figure.tight_layout()
        figure.savefig(destination / f"{metric}_val_vs_test.png", bbox_inches="tight")
        plt.close(figure)
