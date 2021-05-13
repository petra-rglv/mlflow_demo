
## One tool to track, serve and containerize your model

**MLflow** is one of the tools available, which help you track your experiments, serve your models and also build containers. 

Install it with `pip install mlflow`

Run your experiment *sklearn_mlflow_train.ipynb* and visualize, search and compare results with the tracking UI

```mlflow ui```  

In the same directory, in Python, run `!mlflow ui`.   

In anaconda prompt, activate the environment, navigate to the folder and run `mlflow ui`

Open it at *localhost:5000*

## Serve a model with *MLflow*

```mlflow models serve -m runs:/some-run-uuid/sk_models --port 1234```

or

```mlflow models serve -m runs:/099cbc6f09db407fb1d14211ccf02cb7/knn_model --port 1234```

And you are ready to call your application at http://127.0.0.1:1234/invocations.

#### [Deploy MLflow models](https://mlflow.org/docs/latest/models.html#deploy-mlflow-models)

Example requests:
```
# split-oriented

curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
    "columns": ["a", "b", "c"],
    "data": [[1, 2, 3], [4, 5, 6]]
}'
```
```
# record-oriented (fine for vector rows, loses ordering for JSON records)

curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json; format=pandas-records' -d '[
    {"a": 1,"b": 2,"c": 3},
    {"a": 4,"b": 5,"c": 6}
]'
```

## Build a docker container for your model with *MLflow*

```mlflow models build-docker -m runs:/some-run-uuid/my-model -n my-mlflow-image```

or

``` mlflow models build-docker -m runs:/efa1107f48d04319a326b7eef97019e7/knn_model -n my-mlflow-image```

And run your container as usual

```docker run -p 5001:8080 my-mlflow-image```

Now call your containerized app at http://127.0.0.1:5001/invocations

[MLflow Command-Line Interface](https://www.mlflow.org/docs/latest/cli.html)

## Start mlflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns# mlflow_demo
