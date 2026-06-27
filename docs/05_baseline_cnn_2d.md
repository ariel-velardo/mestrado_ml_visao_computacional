# Baseline CNN 2D

## 1. Objetivo

Esta etapa documenta o primeiro baseline de rede neural convolucional 2D do projeto.

O objetivo foi treinar uma CNN simples para classificação binária das imagens JPEG de CT hepática:

- `Healthy`;
- `Hepatic_Steatosis`.

Este baseline não tem objetivo de representar uma solução diagnóstica definitiva. Ele serve como experimento inicial de modelagem visual, respeitando as decisões metodológicas já definidas nas etapas anteriores.

## 2. Contexto metodológico

O dataset contém imagens JPEG de tomografia computadorizada hepática, sem arquivos DICOM, NIfTI ou metadados clínicos.

Limitações importantes:

- não há valores HU confiáveis;
- não há informação de scanner;
- não há informação de protocolo de aquisição;
- não há fase de contraste;
- não há validação externa;
- o identificador `inferred_group_id` é inferido tecnicamente pelo nome dos arquivos.

Por esse motivo, os resultados devem ser interpretados como experimentais e exploratórios.

## 3. Regra principal de split

A regra metodológica central foi mantida:

    nunca dividir aleatoriamente por slice.

O treinamento, validação e teste usam os splits já definidos em:

    data/interim/split_slices.csv

Todos os slices de um mesmo `inferred_group_id` permanecem no mesmo split.

## 4. Arquivos implementados

A implementação do baseline CNN 2D foi organizada nos seguintes arquivos:

    src/liverct/models/cnn_dataset.py
    src/liverct/models/simple_cnn.py
    src/liverct/models/train_cnn.py
    src/liverct/evaluation/classification_metrics.py
    src/liverct/evaluation/group_aggregation.py
    scripts/train_baseline_cnn.py
    scripts/evaluate_baseline_cnn.py
    notebooks/05_baseline_cnn_2d.ipynb

## 5. Dataset PyTorch

Foi criado um dataset PyTorch para leitura das imagens a partir de:

    data/interim/split_slices.csv

O dataset:

- lê a coluna `file_path`;
- converte as imagens para escala de cinza;
- garante dimensão 256x256;
- normaliza os pixels para o intervalo `[0, 1]`;
- retorna imagem, label e metadados mínimos necessários para avaliação.

As variáveis `filename`, `file_path`, `slice_id`, `file_size_bytes`, `md5` e `inferred_group_id` não são usadas como variáveis preditoras.

## 6. Arquitetura da CNN

Foi utilizada uma CNN pequena, com poucas camadas convolucionais.

A decisão foi manter uma arquitetura simples para:

- reduzir risco de overfitting;
- facilitar interpretação;
- criar uma referência inicial;
- evitar buscar alta performance a qualquer custo.

O modelo foi treinado com `BCEWithLogitsLoss`.

## 7. Estratégia de treino

O script de treino usa:

- `train` para treinamento;
- `val` para validação;
- early stopping baseado na loss de validação;
- checkpoint do melhor modelo;
- seed fixa;
- logs por época.

O conjunto de teste não é usado durante o treinamento nem para escolha de hiperparâmetros.

## 8. Estratégia de avaliação

A avaliação foi feita em dois níveis:

1. nível de slice;
2. nível de grupo inferido (`inferred_group_id`).

No nível de grupo, as probabilidades dos slices foram agregadas por média para gerar uma probabilidade final por grupo.

Essa avaliação por grupo é metodologicamente mais importante, pois reduz o risco de superestimar a performance devido à correlação entre slices de um mesmo exame ou agrupamento.

## 9. Métricas utilizadas

Foram calculadas:

- accuracy;
- balanced accuracy;
- precision;
- recall/sensibilidade;
- especificidade;
- F1-score;
- ROC-AUC;
- average precision;
- matriz de confusão.

A classe positiva é:

    Hepatic_Steatosis

## 10. Resultado de validação

Após o treinamento com early stopping, o melhor checkpoint foi avaliado no conjunto de validação.

| Nível | N | Balanced Accuracy | Sensibilidade | Especificidade | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Slice | 538 | 0,8343 | 0,8014 | 0,8672 | 0,8339 | 0,9431 | 0,9507 |
| Grupo | 33 | 0,8182 | 0,8182 | 0,8182 | 0,8571 | 0,9339 | 0,9666 |

## 11. Resultado final no teste

O conjunto de teste foi usado apenas após o treinamento e seleção do melhor checkpoint.

| Nível | N | Balanced Accuracy | Sensibilidade | Especificidade | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Slice | 560 | 0,8155 | 0,8233 | 0,8077 | 0,8275 | 0,9227 | 0,9451 |
| Grupo | 35 | 0,8080 | 0,7826 | 0,8333 | 0,8372 | 0,9203 | 0,9624 |

## 12. Matriz de confusão no teste

### 12.1. Nível de slice

| | Predito Healthy | Predito Hepatic_Steatosis |
|---|---:|---:|
| Real Healthy | 210 | 50 |
| Real Hepatic_Steatosis | 53 | 247 |

### 12.2. Nível de grupo

| | Predito Healthy | Predito Hepatic_Steatosis |
|---|---:|---:|
| Real Healthy | 10 | 2 |
| Real Hepatic_Steatosis | 5 | 18 |

## 13. Comparação com baseline estatístico

O baseline estatístico anterior usou apenas:

- `mean_intensity`;
- `std_intensity`;
- `min_intensity`;
- `max_intensity`.

A comparação no teste mostra:

### 13.1. Nível de slice

| Métrica | Baseline estatístico | CNN 2D | Diferença |
|---|---:|---:|---:|
| Balanced Accuracy | 0,7340 | 0,8155 | +0,0815 |
| Sensibilidade | 0,7833 | 0,8233 | +0,0400 |
| Especificidade | 0,6846 | 0,8077 | +0,1231 |
| F1 | 0,7618 | 0,8275 | +0,0657 |
| ROC-AUC | 0,8549 | 0,9227 | +0,0678 |
| Average Precision | 0,8768 | 0,9451 | +0,0683 |

### 13.2. Nível de grupo

| Métrica | Baseline estatístico | CNN 2D | Diferença |
|---|---:|---:|---:|
| Balanced Accuracy | 0,7899 | 0,8080 | +0,0181 |
| Sensibilidade | 0,9130 | 0,7826 | -0,1304 |
| Especificidade | 0,6667 | 0,8333 | +0,1666 |
| F1 | 0,8750 | 0,8372 | -0,0378 |
| ROC-AUC | 0,8333 | 0,9203 | +0,0870 |
| Average Precision | 0,9057 | 0,9624 | +0,0567 |

## 14. Interpretação

A CNN 2D apresentou desempenho acima do acaso e superou o baseline estatístico em várias métricas, especialmente no nível de slice.

No nível de grupo, a melhora em balanced accuracy foi pequena:

    +0,0181

A CNN melhorou especificidade e ROC-AUC, mas reduziu sensibilidade e F1 em relação ao baseline estatístico.

Isso indica que a CNN não deve ser apresentada como solução claramente superior ou definitiva. Ela deve ser interpretada como um baseline visual inicial.

## 15. Implicações metodológicas

O fato de o baseline estatístico já apresentar desempenho relevante indica que existe sinal discriminativo nas estatísticas globais das imagens.

Portanto, a CNN pode estar aprendendo uma combinação de:

- padrão anatômico;
- diferenças globais de intensidade;
- textura;
- contraste;
- compressão JPEG;
- artefatos de exportação;
- características técnicas do dataset.

Por isso, a próxima etapa deve investigar onde o modelo está olhando e quais grupos estão sendo classificados incorretamente.

## 16. Limitações

As principais limitações desta etapa são:

- ausência de validação externa;
- ausência de metadados clínicos;
- ausência de DICOM/NIfTI;
- ausência de valores HU confiáveis;
- amostra pequena no nível de grupo;
- avaliação dependente de `inferred_group_id`;
- possível aprendizado de artefatos técnicos;
- ausência de interpretabilidade visual nesta etapa.

## 17. Conclusão

O baseline CNN 2D estabelece uma primeira referência visual para o projeto.

O resultado é promissor como experimento exploratório, mas exige cautela. A CNN melhora algumas métricas em relação ao baseline estatístico, mas não de forma suficiente para sustentar conclusão clínica forte.

A próxima etapa recomendada é análise visual de erros e interpretabilidade, especialmente com Grad-CAM.
