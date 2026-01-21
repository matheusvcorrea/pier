# USAGE

Primeiro rode os arquivos de traino e depois a API com `FastAPI`.

# Train

Primeiro passo

```sh
python3 train.py

# Train with custom experiment name
python3 train.py --experiment-name lottery_v2

# Train with remote MLflow server
python3 train.py --mlflow-tracking-uri http://localhost:5000 --experiment-name lottery_production
```

Utilize `python3 train.py --help` para mais detalhes de usabilidade desse arquivo.

# The API usage

No ambiente de desenvolvimento execute o comando `fastapi dev api/api.py`. Em seguida você pode se dirigir até o endpoint `/docs` do FastAPI em [Lottery Accumulation Prediction API](http://localhost:8000/docs#/).

O comando a seguir é um exemplo de utilização do `curl` cli.

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

Em produção utilize o container definido em [Dockerfile](./Dockerfile).

# Model design

Optei por um modelo que faça predição de acumulação. Criei um arquivo de analise inicial em [ead-modeling](./ead-modeling.ipynb) e arquivos auxiliares na raiz do projeto: [evaluate_report](./evaluate_report.py), [train](./train.py) e [classify_model](./classify_model.py). Esse ultimo é onde o treino realmente acontece.

**Justificativa**: Um problema de classificação exige alguns cuidados com relação a métricas de escolha do melhor modelo e também validação. Dando assim uma possibilidade interessante na solução proposta.

**Modelo escolhido**: Escolhi o `Lightgbm` por ser um abordagem de modelegem que suporta uma variedade de dados nativamente. Também, dado o tipo de dados tabular, decidi por um modelo que é reportado como um dos melhores resultados na literatura.

**Feature Selection**: Optei por features puramente numéricas num primeiro passo, visto que são fáceis de analisar, tratar e proporcionam rápida prototipagem de modelo. Também apliquei uma seleção automática de features utilizando utilizando um modelo `RandomForest`. Essa escolha foi por conta do tamanho da base de dados e numero de features. Outras abordagens poderiam ser aplicadas, mas optei por essa num primeiro momento.

Cada uma dessas escolhas iniciais se mostram boas na criação do modelo, resultando em métricas de qualidade do modelo boas.

# API Design

Aqui apenas segui as instruções presentes no README. Tentei manter o código enxuto e separado em módulos principais

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

