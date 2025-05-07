FROM python:3.10-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./

RUN pip install uv
RUN uv venv --python 3.10
RUN uv sync --frozen --no-install-project --compile-bytecode

COPY ./src ./src
COPY ./mlflow-serve ./mlflow-serve
COPY ./model_weights ./model_weights

RUN pip install hydra-core==1.3.2 mlflow==2.22.0 pandas==2.2.3 pre-commit==4.2.0 pytorch-lightning==2.5.0.post0 transformers==4.47.0


CMD ["mlflow", "models", "serve", "-m", "mlflow-serve/mlflow/crispr_off_t_model", "-p", "8888", "--host", "0.0.0.0", "--no-conda"]
#mlflow models serve -m /mlflow/crispr_off_t_model -p 8888 --host 0.0.0.0 --no-conda
#sudo docker build -t mlflow-app -f mlflow-serve/mlflow/Dockerfile .
#sudo docker run -d -p 8888:8888 --name mlflow-server mlflow-app

#curl -X POST http://localhost:8888/invocations -H "Content-Type: application/json"      -d '{"inputs": [{"sequence":"GTCACCAATCCTGTCCCTAGTGG", "Target sequence": "TAAAGCAATCCTGTCCCCAGAGG"}]}'
#for docker
#curl -X POST http://localhost:8888/invocations -H "Content-Type: application/json"      -d '{"inputs": [{"sequence":["GTCACCAATCCTGTCCCTAGTGG"], "Target sequence": ["TAAAGCAATCCTGTCCCCAGAGT"]}]}'
