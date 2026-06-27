# Auditoria de qualidade das imagens

## 1. Objetivo

Esta etapa documenta a auditoria técnica inicial das imagens do dataset, antes da modelagem.

O objetivo foi verificar:

- existência dos arquivos;
- legibilidade das imagens;
- dimensões;
- duplicatas exatas por hash;
- estatísticas simples de intensidade;
- diferenças técnicas entre classes e splits.

## 2. Script utilizado

A auditoria foi implementada nos seguintes arquivos:

    src/liverct/data/audit_images.py
    scripts/audit_images.py

O script utiliza como entrada preferencial:

    data/interim/split_slices.csv

Caso esse arquivo não exista, utiliza:

    data/interim/dataset_index.csv

Os arquivos gerados são locais e ignorados pelo Git:

    data/interim/image_quality_audit.csv
    data/interim/duplicate_hashes.csv
    data/interim/image_quality_summary.json

## 3. Resultados gerais

| Métrica | Resultado |
|---|---:|
| Total de imagens | 3.557 |
| Arquivos ausentes | 0 |
| Imagens ilegíveis | 0 |
| Grupos de duplicatas exatas | 0 |
| Arquivos duplicados exatos | 0 |

Todas as imagens foram abertas corretamente.

## 4. Dimensões

| Dimensão | Número de imagens |
|---|---:|
| 256x256 | 3.557 |

Todas as imagens possuem a mesma dimensão espacial.

## 5. Resumo por classe

| Classe | Imagens | Grupos | Média de intensidade | Média do desvio de intensidade | Tamanho médio do arquivo |
|---|---:|---:|---:|---:|---:|
| Healthy | 1.611 | 76 | 39,4444 | 61,2759 | 38.595,2092 |
| Hepatic_Steatosis | 1.946 | 149 | 40,1071 | 58,5789 | 41.793,7174 |

Observação: as médias de intensidade são próximas entre as classes. No entanto, a classe `Hepatic_Steatosis` apresentou tamanho médio de arquivo maior que a classe `Healthy`. Essa diferença deve ser acompanhada nas próximas etapas para reduzir o risco de o modelo aprender artefatos técnicos de compressão, contraste ou origem dos arquivos em vez de padrões anatômicos relevantes.

## 6. Resumo por split e classe

| Split | Classe | Imagens | Grupos | Média de intensidade | Média do desvio de intensidade | Tamanho médio do arquivo |
|---|---|---:|---:|---:|---:|---:|
| Test | Healthy | 260 | 12 | 40,5170 | 61,9478 | 40.007,5885 |
| Test | Hepatic_Steatosis | 300 | 23 | 43,1512 | 59,7174 | 42.006,1167 |
| Train | Healthy | 1.095 | 53 | 39,3134 | 61,1031 | 38.471,4831 |
| Train | Hepatic_Steatosis | 1.364 | 104 | 39,8037 | 58,5942 | 41.583,5711 |
| Validation | Healthy | 256 | 11 | 38,9155 | 61,3331 | 37.689,9805 |
| Validation | Hepatic_Steatosis | 282 | 22 | 38,3357 | 57,2938 | 42.584,2128 |

## 7. Interpretação metodológica

A auditoria não encontrou problemas críticos de qualidade dos arquivos:

- não há imagens ausentes;
- não há imagens ilegíveis;
- não há duplicatas exatas;
- todas as imagens têm dimensão padronizada de 256x256.

No entanto, por se tratar de imagens JPEG sem DICOM, NIfTI ou metadata clínico, as estatísticas de intensidade devem ser interpretadas apenas como valores de pixel da imagem disponível. Elas não representam unidades Hounsfield confiáveis.

## 8. Implicações para a modelagem

A modelagem pode avançar para uma etapa exploratória, desde que respeite as decisões metodológicas anteriores:

- split por `inferred_group_id`;
- ausência de vazamento entre treino, validação e teste;
- avaliação principal por grupo, não apenas por slice;
- documentação explícita das limitações do formato JPEG;
- análise de erro e interpretabilidade visual antes de qualquer conclusão clínica.

## 9. Próximos passos

Antes de treinar modelos finais, recomenda-se:

1. criar EDA visual com amostras por classe e split;
2. comparar distribuições de intensidade por classe;
3. comparar tamanho de arquivo por classe;
4. verificar se há artefatos visuais sistemáticos;
5. criar um baseline simples de classificação;
6. avaliar performance por slice e por `inferred_group_id`;
7. agregar probabilidades por grupo na avaliação final.
