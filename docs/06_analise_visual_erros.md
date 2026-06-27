# Análise visual de erros do baseline CNN 2D

## 1. Objetivo

Esta etapa realiza uma análise visual dos erros e acertos do baseline CNN 2D no conjunto de teste.

O objetivo é selecionar casos representativos para inspeção qualitativa antes da aplicação de Grad-CAM.

Esta etapa não treina modelo, não altera splits e não recalcula predições. Ela apenas utiliza os resultados já gerados pelo baseline CNN 2D.

## 2. Entradas

O script principal desta etapa é:

    scripts/run_error_analysis.py

Ele utiliza as seguintes entradas:

### 2.1. Predições por slice

Arquivo:

    reports/tables/baseline_cnn_test_slice_predictions.csv

Colunas principais:

- `split`;
- `label`;
- `prob_positive`;
- `pred_label`;
- `inferred_group_id`;
- `file_path`.

Este arquivo contém a probabilidade predita pela CNN para cada slice do conjunto de teste.

### 2.2. Predições por grupo

Arquivo:

    reports/tables/baseline_cnn_test_group_predictions.csv

Colunas principais:

- `split`;
- `inferred_group_id`;
- `label`;
- `prob_positive`;
- `n_slices`;
- `prob_positive_std`;
- `pred_label`.

Este arquivo contém as predições agregadas por `inferred_group_id`, usando a média das probabilidades dos slices.

### 2.3. Índice completo dos slices

Arquivo:

    data/interim/split_slices.csv

Colunas principais:

- `class_name`;
- `label`;
- `filename`;
- `inferred_group_id`;
- `slice_id`;
- `file_path`;
- `split`.

Este arquivo é usado para recuperar informações visuais e metadados mínimos dos slices.

## 3. Processamento

O script realiza os seguintes passos:

1. carrega as predições por slice;
2. carrega as predições por grupo;
3. carrega o índice completo de slices;
4. filtra apenas o conjunto de teste;
5. classifica cada grupo como:
   - TP: verdadeiro positivo;
   - TN: verdadeiro negativo;
   - FP: falso positivo;
   - FN: falso negativo;
6. seleciona grupos representativos por tipo de resultado;
7. seleciona slices representativos dentro de cada grupo;
8. gera tabelas de apoio;
9. gera painéis visuais por grupo.

## 4. Critérios de seleção dos grupos

Foram selecionados até 3 grupos por tipo de resultado:

- FP: grupos negativos reais com maior probabilidade positiva;
- FN: grupos positivos reais com menor probabilidade positiva;
- TP: grupos positivos reais com maior probabilidade positiva;
- TN: grupos negativos reais com menor probabilidade positiva.

Como havia apenas 2 falsos positivos no teste, foram selecionados 2 grupos FP.

## 5. Saídas

O script gera as seguintes saídas locais:

### 5.1. Tabela de grupos selecionados

Arquivo:

    reports/tables/error_analysis_group_cases.csv

Conteúdo:

- tipo de resultado;
- split;
- `inferred_group_id`;
- classe real;
- classe predita;
- probabilidade positiva média do grupo;
- desvio padrão das probabilidades dos slices;
- número de slices.

### 5.2. Tabela de slices selecionados

Arquivo:

    reports/tables/error_analysis_slice_cases.csv

Conteúdo:

- tipo de resultado do grupo;
- grupo inferido;
- classe real;
- classe predita;
- probabilidade por slice;
- probabilidade média do grupo;
- caminho da imagem;
- informações do slice.

### 5.3. Figuras

Diretório:

    reports/figures/error_analysis/

Foram geradas figuras `.png` com painéis de slices para cada grupo selecionado.

As figuras não são versionadas no Git, pois são artefatos derivados.

## 6. Resumo dos resultados no teste

Distribuição dos grupos no conjunto de teste:

| Resultado | Número de grupos |
|---|---:|
| FN | 5 |
| FP | 2 |
| TN | 10 |
| TP | 18 |

## 7. Grupos selecionados

| Resultado | inferred_group_id | Classe real | Classe predita | Probabilidade positiva | Nº slices |
|---|---|---|---|---:|---:|
| FN | 207-img-00043 | Hepatic_Steatosis | Healthy | 0,1164 | 12 |
| FN | 231-img-00019 | Hepatic_Steatosis | Healthy | 0,1776 | 12 |
| FN | 72-img-00032 | Hepatic_Steatosis | Healthy | 0,3049 | 16 |
| FP | 114-img-00064 | Healthy | Hepatic_Steatosis | 0,8059 | 24 |
| FP | 96-img-00049 | Healthy | Hepatic_Steatosis | 0,6065 | 18 |
| TN | 106-img-00060 | Healthy | Healthy | 0,0592 | 21 |
| TN | 149-img-00097 | Healthy | Healthy | 0,0550 | 24 |
| TN | 82-img-00039 | Healthy | Healthy | 0,0602 | 21 |
| TP | 190-img-00028 | Hepatic_Steatosis | Hepatic_Steatosis | 0,9956 | 12 |
| TP | 194-img-00032 | Hepatic_Steatosis | Hepatic_Steatosis | 0,9947 | 12 |
| TP | 98-img-00052 | Hepatic_Steatosis | Hepatic_Steatosis | 0,9945 | 12 |

## 8. Interpretação inicial

A análise visual de erros prepara o terreno para a interpretabilidade.

Os falsos negativos são casos de `Hepatic_Steatosis` que receberam baixa probabilidade positiva. Esses casos são importantes para entender se o modelo não capturou sinais visuais relevantes ou se há características técnicas/anatômicas diferentes.

Os falsos positivos são casos `Healthy` classificados como `Hepatic_Steatosis`. Esses casos são importantes para investigar se o modelo está confundindo textura, contraste, ruído, bordas ou artefatos de JPEG com sinal de esteatose.

Os verdadeiros positivos e verdadeiros negativos foram selecionados como referências visuais de casos em que o modelo apresentou alta confiança.

## 9. Limitações

Esta etapa é qualitativa e exploratória.

As principais limitações são:

- os grupos foram definidos por `inferred_group_id`, não por identificação clínica validada;
- as imagens são JPEG;
- não há valores HU confiáveis;
- não há metadados clínicos;
- não há segmentação hepática;
- os painéis visuais não provam causalidade nem validade clínica;
- as figuras servem apenas para inspeção e preparação da etapa de Grad-CAM.

## 10. Próxima etapa

A próxima etapa será aplicar Grad-CAM sobre os grupos selecionados.

O objetivo será verificar se a CNN está concentrando sua atenção em regiões anatomicamente plausíveis ou se está usando artefatos, bordas, ruído, contraste global ou padrões técnicos do dataset.
