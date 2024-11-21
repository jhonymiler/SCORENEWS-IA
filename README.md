
# Sistema de Análise de Confiabilidade de Notícias

Este projeto é um sistema para análise de notícias, focado em identificar se elas são confiáveis ou não. Ele utiliza técnicas de aprendizado de máquina e processamento de linguagem natural para detectar viés ou parcialidade em textos de notícias.

## Funcionalidades
- Processa e limpa textos de notícias.
- Identifica características de viés, como termos vagos e apelos emocionais.
- Classifica as notícias como confiáveis ou não confiáveis.
- Gera relatórios detalhados sobre sinais de viés detectados.

## Estrutura
- `app.py`: Contém o código principal para análise e classificação.
- `news_data.csv`: Conjunto de dados de exemplo contendo notícias e suas classificações.

## Como usar
1. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

2. Certifique-se de ter o arquivo `news_data.csv` no mesmo diretório que o `app.py`.

3. Execute o código:
    ```bash
    python app.py
    ```

4. Insira o texto de uma notícia no exemplo fornecido para ver a classificação e os sinais de viés detectados.

## Exemplo de Entrada
```text
"Grande escândalo é revelado e gera medo na população"
```

## Exemplo de Saída
```text
=== Análise da Notícia ===
Texto da notícia: Grande escândalo é revelado e gera medo na população

Resultado da Classificação: Não Confiável

=== Relatório de Viés ===
Sinais de viés detectados:
  Sensacionalismo: 1 ocorrência(s)
  Apelo emocional: 1 ocorrência(s)
  Termos vagos: 0 ocorrência(s)
```

## Dependências
- pandas
- scikit-learn
- nltk
- numpy
- scipy
