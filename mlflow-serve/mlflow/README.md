To build your mlfwlow model from the root directory of the project run

```sh
uv run mlflow-serve/mlflow/mlflow_model_wrapper.py
```

Then there are two options for the inference:

1. Run mlflow server with the model via

   ```sh
   mlflow models serve -m /mlflow/crispr_off_t_model -p 8888 --host 0.0.0.0 --no-conda
   ```

2. Run the mlflow model in a docker via

   ```sh
    sudo docker build -t mlflow-app -f mlflow-serve/mlflow/Dockerfile .
    sudo docker run -d -p 8888:8888 --name mlflow-server mlflow-app
   ```

Then you can send requests using following format

```sh

     curl -X POST http://localhost:8888/invocations         -H "Content-Type: application/json"   -d '{
             "inputs": [
                 {"sequence":["GTCACCAATCCTGTCCCTAGTGG"],
                 "Target sequence": ["TAAAGCAATCCTGTCCCCAGAGT"]
                 }
             ]
         }'

```

To run all services use

```sh
   sudo docker-compose -f mlflow-serve/docker-compose.yml up --build
```
