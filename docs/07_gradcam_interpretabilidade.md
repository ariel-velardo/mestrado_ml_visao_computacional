# Grad-CAM para interpretabilidade do baseline CNN 2D

## 1. Objetivo

Esta etapa aplica Grad-CAM ao baseline CNN 2D treinado para classificação binária de imagens JPEG de CT hepática.

O objetivo é investigar qualitativamente quais regiões da imagem mais contribuíram para a predição da CNN nos casos selecionados pela análise visual de erros.

Esta etapa não treina modelo, não altera splits e não recalcula a seleção dos casos. Ela apenas carrega o checkpoint treinado e aplica Grad-CAM sobre os slices selecionados.

## 2. Contexto metodológico

O projeto trabalha com duas classes:

- `Healthy = 0`;
- `Hepatic_Steatosis = 1`.

As imagens são JPEG e não há arquivos DICOM, NIfTI, valores HU confiáveis, metadados clínicos, informações de scanner, protocolo, fase de aquisição ou segmentação hepática.

O agrupamento utilizado é `inferred_group_id`, inferido tecnicamente a partir do nome dos arquivos. Esse campo não deve ser interpretado como identificação clínica validada de paciente.

## 3. Entradas

O script principal desta etapa é:

    scripts/run_gradcam.py

Ele utiliza as seguintes entradas:

### 3.1. Checkpoint da CNN

Arquivo:

    models/checkpoints/baseline_cnn_best.pt

Checkpoint treinado do baseline CNN 2D.

Informações confirmadas no checkpoint:

- `model_name`: `SimpleCNN2D`;
- `epoch`: 11;
- `val_loss`: 0.28499032041840394.

### 3.2. Casos selecionados para análise

Arquivo:

    reports/tables/error_analysis_slice_cases.csv

Este arquivo foi gerado por:

    scripts/run_error_analysis.py

Ele contém os slices representativos dos grupos selecionados na etapa de análise visual de erros.

### 3.3. Imagens originais

As imagens são carregadas a partir da coluna:

    file_path

presente em `error_analysis_slice_cases.csv`.

## 4. Implementação

A implementação do Grad-CAM está em:

    src/liverct/explainability/gradcam.py

O script de execução está em:

    scripts/run_gradcam.py

A classe correta do modelo é:

    SimpleCNN2D

Import correto:

    from liverct.models.simple_cnn import SimpleCNN2D

A camada convolucional alvo é detectada automaticamente como a última camada `Conv2d` do modelo.

Nesta execução, a camada utilizada foi:

    features.8

## 5. Processamento

O script realiza os seguintes passos:

1. valida a existência do checkpoint e da tabela de casos;
2. carrega o modelo `SimpleCNN2D`;
3. carrega os pesos salvos em `baseline_cnn_best.pt`;
4. identifica a última camada convolucional;
5. carrega cada imagem em escala de cinza;
6. normaliza a imagem para o intervalo `[0, 1]`;
7. calcula o logit e a probabilidade positiva;
8. calcula o Grad-CAM em relação ao logit positivo;
9. gera a figura com:
   - imagem original;
   - heatmap Grad-CAM;
   - sobreposição do heatmap sobre a imagem;
10. salva a tabela consolidada dos casos processados.

## 6. Saídas

O script gera as seguintes saídas locais:

### 6.1. Tabela de casos Grad-CAM

Arquivo:

    reports/tables/gradcam_cases.csv

Conteúdo principal:

- tipo de resultado do grupo;
- `inferred_group_id`;
- classe real;
- classe predita;
- probabilidade original do slice;
- probabilidade recalculada pelo Grad-CAM;
- logit;
- camada alvo;
- caminho da figura gerada.

### 6.2. Figuras Grad-CAM

Diretório:

    reports/figures/gradcam/

Foram geradas 66 figuras `.png`.

Cada figura contém:

- imagem original;
- mapa Grad-CAM;
- sobreposição entre imagem e mapa Grad-CAM.

As figuras são artefatos derivados e não são versionadas no Git.

## 7. Resultados da execução

A execução processou 66 slices selecionados a partir da análise visual de erros.

Os casos contemplam:

- falsos negativos;
- falsos positivos;
- verdadeiros negativos;
- verdadeiros positivos.

A camada utilizada para geração dos mapas foi:

    features.8

## 8. Interpretação esperada

A análise por Grad-CAM deve ser usada para investigar se a CNN está concentrando evidência visual em regiões anatomicamente plausíveis ou se está utilizando atalhos visuais.

Exemplos de sinais de alerta:

- ativação forte em bordas da imagem;
- ativação em áreas pretas fora do corpo;
- ativação em marcas, ruído ou artefatos de compressão;
- ativação difusa sem relação anatômica clara;
- foco em regiões fora do fígado.

Caso os mapas se concentrem fora de regiões clinicamente plausíveis, isso enfraquece a interpretação do modelo como classificador visual robusto de esteatose.

## 9. Limitações

Grad-CAM é uma técnica aproximada de interpretabilidade visual.

Neste projeto, a interpretação deve ser ainda mais cautelosa porque:

- as imagens são JPEG;
- não há valores HU confiáveis;
- não há segmentação hepática;
- não há metadados clínicos;
- não há informações sobre scanner, protocolo ou fase;
- o modelo pode aprender atalhos técnicos do dataset;
- o mapa Grad-CAM não prova causalidade;
- o mapa Grad-CAM não valida achado clínico.

## 10. Conclusão

A etapa de Grad-CAM adiciona uma camada qualitativa de interpretabilidade ao baseline CNN 2D.

Ela não transforma o modelo em ferramenta clínica, mas ajuda a avaliar se a CNN parece usar regiões visualmente plausíveis ou se pode estar se apoiando em artefatos técnicos.

A próxima etapa recomendada é construir o notebook de apresentação:

    notebooks/07_gradcam_interpretabilidade.ipynb

Esse notebook deve organizar os mapas por tipo de resultado: FN, FP, TN e TP.
