# USAGE

Primeiro rode os arquivos de traino e depois a API com `FastAPI`.

# Train
Premeiro passo

```sh
python3 train.py

# Train with custom experiment name
python3 train.py --experiment-name lottery_v2

# Train with remote MLflow server
python3 train.py --mlflow-tracking-uri http://localhost:5000 --experiment-name lottery_production
```

Utilize `python3 train.py --help` para mais detalhes de usabilidade desse arquivo.

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

Em produção utilize o container definido em [](./Dockerfile).

# Model design

Optnei por um modelo que faça predição de acumalção. Criei um arquivo de analise inicial em [ead-modeling](./ead-modeling.ipynb) e arquivos auxiliares na raiz do projeto: [evaluate_report](./evaluate_report.py), [train](./train.py) e [classify_model](./classify_model.py). Esse ultimo é onde o treino realmente acontece.

**Justicativa**: Um problema de classifação exige alguns cuidados com relação a metricas de escolha do melhor modelo e também validação. Dando assim uma possibilidade interessante na solução proposta.

**Modelo escolhido**: Escolhi o `Lightgbm` por ser um abordagem de modelegem que suporta uma variedade de dados nativamente. Também, dado o tipo de dados tabular, decidi por um modelo que é reportado como um dos melhores resultados na literatura.

**Feature Selection**: Optei por features puramente numericas num primeiro passo, visto que são faceis de analisar, tratar e proporcionam rapida prototipagem de modelo. Também apliquei uma seleção automática de features utilizando utilizando um modelo RandomForest. Essa escolha foi por conta do tamanho da base de dados e numero de features. Outras abordagens poderiam ser aplicadas, mas optei por essa num primeiro momento.

Cada uma dessas escolhas iniciais se mostram boas na criação do modelo, resultando em metricas de qualidade do modelo boas.

# API Design

Aqui apenas segui as instruções presentes no README. Tentei manter o codigo enxuto e separado em modulos principais

```sh
api/
├── __init__.py
├── api.py
├── estimator.py
├── route.py
├── schemas.py
└── test_api.py
```

O aquivo é principal e *main* da aplicação é o api.py. O arquivo estimator carrega o modelo de acordo com a disponibilidade.

