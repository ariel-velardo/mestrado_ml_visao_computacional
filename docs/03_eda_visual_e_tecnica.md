# EDA visual e técnica das imagens

## 1. Objetivo

Esta etapa documenta a análise exploratória visual e técnica das imagens após a indexação, criação dos splits e auditoria de qualidade.

O objetivo foi verificar:

- amostras visuais por classe;
- amostras visuais por split e classe;
- distribuição da intensidade média dos pixels;
- distribuição do tamanho dos arquivos;
- possíveis diferenças técnicas entre `Healthy` e `Hepatic_Steatosis`;
- indícios iniciais de artefatos visuais ou viés técnico.

## 2. Script utilizado

A EDA visual foi implementada nos arquivos:

    src/liverct/visualization/eda_images.py
    scripts/run_visual_eda.py

O script utiliza como entrada:

    data/interim/image_quality_audit.csv

Esse arquivo é derivado localmente e não é versionado no Git.

## 3. Arquivos gerados localmente

As figuras e tabelas geradas são saídas derivadas dos dados e ficam ignoradas pelo Git.

Principais saídas:

    reports/figures/eda_samples_by_class.png
    reports/figures/eda_samples_by_split_class.png
    reports/figures/eda_mean_intensity_hist_by_class.png
    reports/figures/eda_file_size_hist_by_class.png
    reports/figures/eda_mean_intensity_boxplot_by_class.png
    reports/figures/eda_file_size_boxplot_by_class.png
    reports/figures/eda_mean_intensity_boxplot_by_split_class.png
    reports/figures/eda_file_size_boxplot_by_split_class.png

    reports/tables/eda_image_summary_by_class.csv
    reports/tables/eda_image_summary_by_split_class.csv

## 4. Amostras visuais por classe

A inspeção visual mostra que as imagens são cortes axiais de tomografia computadorizada da região hepática.

Foram observadas variações entre imagens, incluindo:

- diferenças de posição anatômica do corte;
- diferentes proporções visíveis de fígado, baço, estômago e estruturas adjacentes;
- variações de campo de visão;
- variações de ruído/granulação;
- diferenças sutis de contraste e textura;
- presença de bordas e regiões pretas associadas à exportação da imagem.

Essas variações reforçam que o modelo pode aprender padrões anatômicos, mas também pode capturar artefatos técnicos se o pipeline não for cuidadosamente avaliado.

## 5. Amostras visuais por split e classe

A EDA por split indica que os conjuntos de treino, validação e teste contêm exemplos visualmente plausíveis das duas classes.

Não foi observada, nesta inspeção inicial, uma falha grosseira de composição visual entre os splits. Ainda assim, a avaliação quantitativa e a análise de erro serão necessárias para confirmar se o modelo generaliza entre grupos.

## 6. Distribuição da intensidade média

A distribuição da intensidade média dos pixels apresentou forte sobreposição entre `Healthy` e `Hepatic_Steatosis`.

Interpretação:

- a separação entre classes não parece ser trivialmente explicada apenas pela média de brilho da imagem;
- isso reduz, mas não elimina, o risco de o modelo aprender um atalho simples baseado em intensidade global;
- como as imagens estão em JPEG, esses valores representam intensidade de pixel da imagem exportada, não unidades Hounsfield confiáveis.

## 7. Distribuição do tamanho de arquivo

A distribuição do tamanho de arquivo mostrou uma diferença mais visível entre as classes.

Na auditoria anterior, o tamanho médio dos arquivos foi:

| Classe | Tamanho médio do arquivo |
|---|---:|
| Healthy | 38.595,2092 bytes |
| Hepatic_Steatosis | 41.793,7174 bytes |

Interpretação:

- `Hepatic_Steatosis` apresenta, em média, arquivos maiores;
- essa diferença pode estar associada a textura, ruído, compressão JPEG, conteúdo anatômico ou origem/processamento das imagens;
- esse ponto deve ser tratado como possível fonte de viés técnico;
- a modelagem não deve usar tamanho de arquivo como variável preditora;
- a performance do modelo deve ser acompanhada por análise de erro e interpretabilidade visual.

## 8. Limitações da EDA

Esta EDA é uma análise exploratória sobre imagens JPEG.

Limitações:

- não há DICOM;
- não há NIfTI;
- não há metadata clínico;
- não há valores HU confiáveis;
- não há informação de scanner, protocolo ou fase de contraste;
- o identificador de grupo é inferido tecnicamente pelo nome do arquivo;
- as conclusões visuais não substituem validação clínica.

## 9. Implicações para a modelagem

A EDA permite avançar para baseline exploratório, mas com cautela.

Regras que continuam obrigatórias:

- split por `inferred_group_id`;
- proibição de split aleatório por slice;
- avaliação por grupo além da avaliação por slice;
- agregação das probabilidades dos slices por grupo;
- análise de erro visual;
- cautela com diferenças técnicas entre classes.

## 10. Decisão metodológica

Após a EDA visual e técnica, o projeto pode avançar para um primeiro baseline de modelagem, desde que a formulação continue sendo experimental e metodologicamente limitada:

    classificação exploratória de imagens JPEG de CT hepática,
    com split por agrupamento inferido e avaliação por grupo.

A modelagem inicial deve priorizar reprodutibilidade, controle de leakage e interpretação crítica dos resultados, em vez de buscar apenas alta acurácia.
