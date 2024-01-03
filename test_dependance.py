from river import metrics
from river.tree import HoeffdingTreeClassifier
from river.forest import ARFClassifier
from river.ensemble import BaggingClassifier
from calmid import CALMID
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import os
from testing_utils.custom_import_dataset import get_iter_stream
import inspect

### Modify this
dataset_name = "VarImb_ISSI"
models = [CALMID(sizelab = 300), CALMID(sizelab = 800), CALMID(budget=0.15), CALMID(budget=0.25)]
eval_metrics = [
    metrics.Accuracy(),
    metrics.ROCAUC(),
    metrics.BalancedAccuracy(),
    metrics.F1(),
    metrics.Recall(),
    ## metrics.ConfusionMatrix(),
    metrics.GeometricMean()
]
###

save_every_n_steps = 1000
stream = get_iter_stream(dataset_name)
metrics_names = [metric.__class__.__name__.lower() for metric in eval_metrics]
models_res = {}
default_params = CALMID() 
i = 0
for model in models:
    i += 1
    if i <= 2:
        models_res[i] = {
            "model": model,
            "model_metrics": deepcopy(eval_metrics),
            "res_metrics": [],
            "res_preds": [],
            "param": [param for param, value in inspect.getmembers(model)
                                if not param.startswith('_') and not inspect.ismethod(value)
                                and getattr(model, param) != getattr(default_params, param)][-1],
            "value": str([value for param, value in inspect.getmembers(model)
                                if not param.startswith('_') and not inspect.ismethod(value)
                                and getattr(model, param) != getattr(default_params, param)][-1])
        }
    else:
        models_res[i] = {
            "model": model,
            "model_metrics": deepcopy(eval_metrics),
            "res_metrics": [],
            "res_preds": [],
            "param": [param for param, value in inspect.getmembers(model)
                                if not param.startswith('_') and not inspect.ismethod(value)
                                and getattr(model, param) != getattr(default_params, param)][0],
            "value": str([value for param, value in inspect.getmembers(model)
                                if not param.startswith('_') and not inspect.ismethod(value)
                                and getattr(model, param) != getattr(default_params, param)][0])
        }


for model_name, model_dict in models_res.items():
    path_to_dataset_res = os.path.join("results", "dependance", model_dict['param'])
    if not os.path.exists(path_to_dataset_res):
        os.mkdir(path_to_dataset_res)
    if not os.path.exists(os.path.join(path_to_dataset_res, model_dict['value'])):
        os.mkdir(os.path.join(path_to_dataset_res, model_dict['value']))



def main():
    for step, (x, y) in tqdm(enumerate(stream)):

        for model_name, model_dict in models_res.items():

            model, model_metrics, res_metrics, res_preds, param, value = model_dict.values()
            path_to_dataset_res = os.path.join("results", "dependance", param)

            if step == 0:
                model.learn_one(x, y)
                continue

            # pred and learn
            y_pred_probas = model.predict_proba_one(x)
            y_pred = model.predict_one(x)
            res_preds.append([y, y_pred, y_pred_probas])
            model.learn_one(x, y)

            # update metric
            for metric in model_metrics:
                if metric.requires_labels:
                    metric.update(y, y_pred)
                else:
                    metric.update(y, y_pred_probas)
            res_metrics.append([metric.get() for metric in model_metrics])

            # save sometimes
            if step % save_every_n_steps == 0:
                pd.DataFrame(columns=metrics_names, data=res_metrics).to_csv(
                    f"{os.path.join(path_to_dataset_res, value)}/metrics.csv",
                    index=False,
                )
                pd.DataFrame(
                    columns=["y", "y_pred", "y_pred_probas"], data=res_preds
                ).to_csv(
                    f"{os.path.join(path_to_dataset_res, value)}/preds.csv",
                    index=False,
                )

    # save at the end
    pd.DataFrame(columns=metrics_names, data=res_metrics).to_csv(
        f"{os.path.join(path_to_dataset_res, value)}/metrics.csv",
        index=False,
    )
    pd.DataFrame(
        columns=["y", "y_pred", "y_pred_probas"], data=res_preds
    ).to_csv(
        f"{os.path.join(path_to_dataset_res, value)}/preds.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
