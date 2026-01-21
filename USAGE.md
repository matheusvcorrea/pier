# Train with local MLflow tracking (default)
```she python3 train.py```

# Train with custom experiment name
```sh python3 train.py --experiment-name lottery_v2```

# Train with remote MLflow server
```sh python3 train.py --mlflow-tracking-uri http://localhost:5000 --experiment-name lottery_production```

# The API usage

On the dev enviroment run the command `fastapi dev api/api.py`. Then you can refer  to the Docs and perform some input examples at 
[Lottery Accumulation Prediction API](http://localhost:8000/docs#/).

The following is a command line usage example using the `curl` cli.

```sh
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "valorAcumuladoProximoConcurso": 1000000,
  "valorEstimadoProximoConcurso": 1500000,
  "valorArrecadado": 500000,
  "valorAcumuladoConcurso_0_5": 100000,
  "valorAcumuladoConcursoEspecial": 200000
}'
```
