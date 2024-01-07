import pandas as pd
import matplotlib.pyplot as plt

datasets = ["covertype", "pokerhand", "VarImb_ISSI"]
classifiers = [
    "arfclassifier",
    "baggingclassifier",
    "calmid",
    "hoeffdingtreeclassifier",
]

for dataset in datasets:
    for classifier in classifiers:
        globals()[f"{dataset}_{classifier}"] = pd.read_csv(
            f"{dataset}/{classifier}/metrics.csv"
        )
        globals()[f"{dataset}_{classifier}"]["model"] = classifier

    globals()[f"df_{dataset}"] = pd.concat(
        [globals()[f"{dataset}_{classifier}"] for classifier in classifiers]
    )
    globals()[f"grouped_{dataset}"] = globals()[f"df_{dataset}"].groupby(
        "model"
    )

    globals()[f"grouped_{dataset}"]["accuracy"].plot(
        rot=45, title="acc", legend=True
    )
    plt.legend(classifiers, loc="best")
    plt.grid(linestyle=":")
    plt.title("Accuracy on the %s dataset" % dataset)
    plt.tight_layout()
    plt.savefig(f"plots/{dataset}.png")
