# Baseline estatístico de controle

## 1. Objetivo

Esta etapa documenta um baseline estatístico simples usado como teste de sanidade antes da modelagem com imagens.

O objetivo não é criar o melhor modelo possível, mas verificar se estatísticas globais simples das imagens já carregam sinal preditivo relevante para distinguir:

- `Healthy`
- `Hepatic_Steatosis`

Esse baseline ajuda a avaliar se modelos mais complexos, como CNNs, podem estar aprendendo padrões anatômicos relevantes ou apenas atalhos globais de intensidade/contraste.

## 2. Script utilizado

O baseline foi implementado nos arquivos:

    src/liverct/models/statistical_baseline.py
    scripts/run_statistical_baseline.py

O script usa como entrada:

    data/interim/image_quality_audit.csv

Esse arquivo é derivado localmente e não é versionado no Git.

## 3. Variáveis utilizadas

O baseline utilizou apenas estatísticas simples de intensidade dos pixels:

- `mean_intensity`
- `std_intensity`
- `min_intensity`
- `max_intensity`

Essas variáveis foram calculadas a partir das imagens JPEG convertidas para escala de cinza.

## 4. Variáveis propositalmente excluídas

Para evitar vazamento ou uso de artefatos técnicos explícitos, o baseline não utilizou:

- `file_size_bytes`
- `filename`
- `file_path`
- `inferred_group_id`
- `slice_id`
- `md5`
- qualquer informação derivada diretamente do nome do arquivo

O objetivo foi testar somente o poder preditivo de estatísticas globais de intensidade.

## 5. Modelos avaliados

Foram avaliados dois modelos:

1. `DummyClassifier`, com estratégia `most_frequent`;
2. Regressão logística com padronização das variáveis e `class_weight='balanced'`.

O modelo dummy serve como referência mínima. A regressão logística serve como baseline estatístico simples.

## 6. Estratégia de avaliação

A avaliação foi feita em dois níveis:

1. nível de slice;
2. nível de grupo inferido (`inferred_group_id`).

No nível de grupo, as probabilidades dos slices foram agregadas por média para produzir uma probabilidade final por `inferred_group_id`.

Essa avaliação respeita a decisão metodológica principal do projeto:

    todos os slices de um mesmo inferred_group_id permanecem no mesmo split.

## 7. Métricas reportadas

Foram calculadas:

- acurácia;
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

## 8. Resultados principais

### 8.1. Dummy classifier

O modelo dummy apresentou balanced accuracy igual a 0,50 em todos os splits e níveis de avaliação, como esperado para um classificador que prediz sempre a classe mais frequente.

Esse resultado confirma que a regressão logística deve ser comparada contra uma referência trivial.

### 8.2. Regressão logística — nível de slice

| Split | N | Balanced Accuracy | Sensibilidade | Especificidade | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 2.459 | 0,7253 | 0,7236 | 0,7269 | 0,7449 | 0,7815 | 0,7799 |
| Validation | 538 | 0,8160 | 0,8156 | 0,8164 | 0,8229 | 0,9077 | 0,9247 |
| Test | 560 | 0,7340 | 0,7833 | 0,6846 | 0,7618 | 0,8549 | 0,8768 |

### 8.3. Regressão logística — nível de grupo

| Split | N grupos | Balanced Accuracy | Sensibilidade | Especificidade | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 157 | 0,6812 | 0,7019 | 0,6604 | 0,7487 | 0,7564 | 0,8420 |
| Validation | 33 | 0,8864 | 0,8636 | 0,9091 | 0,9048 | 0,9339 | 0,9697 |
| Test | 35 | 0,7899 | 0,9130 | 0,6667 | 0,8750 | 0,8333 | 0,9057 |

## 9. Interpretação

A regressão logística usando apenas estatísticas globais de intensidade apresentou desempenho relevante, especialmente no nível de grupo.

Esse resultado indica que existe sinal preditivo nas estatísticas simples da imagem.

Isso pode significar uma ou mais possibilidades:

- diferença real de padrão visual associada à esteatose;
- diferença global de intensidade/contraste;
- diferença de textura;
- diferença técnica de compressão JPEG;
- diferença de janela de exportação;
- artefato associado à origem ou processamento das imagens.

Portanto, o resultado não deve ser interpretado automaticamente como evidência clínica forte.

## 10. Implicações para CNN

A CNN deverá ser comparada contra este baseline estatístico.

Se uma CNN apresentar desempenho apenas levemente superior ao baseline estatístico, isso pode indicar que ela está explorando sinais globais semelhantes, e não necessariamente aprendendo padrões anatômicos mais ricos.

Por outro lado, se a CNN superar o baseline de forma consistente, especialmente no teste por grupo, ainda será necessário verificar:

- matriz de confusão por grupo;
- erros por classe;
- probabilidades agregadas;
- Grad-CAM ou outra técnica de interpretabilidade;
- amostras de falsos positivos e falsos negativos;
- estabilidade entre treino, validação e teste.

## 11. Decisão metodológica

O baseline estatístico será mantido como referência obrigatória para os próximos experimentos.

A modelagem com CNN só deve avançar respeitando:

- split por `inferred_group_id`;
- avaliação por grupo;
- comparação contra o baseline estatístico;
- documentação explícita de limitações;
- análise visual dos erros;
- cautela contra atalhos técnicos.

## 12. Conclusão

O baseline estatístico mostra que estatísticas simples de pixel já possuem capacidade discriminativa relevante no dataset.

Isso reforça a importância de tratar o projeto como estudo experimental e metodologicamente controlado, e não como validação clínica diagnóstica.

A próxima etapa será implementar um baseline CNN 2D simples, mantendo a comparação com este baseline estatístico.
