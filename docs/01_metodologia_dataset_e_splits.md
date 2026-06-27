# Metodologia inicial: organização do dataset e split por grupo

## 1. Contexto do dataset

Este projeto utiliza o dataset público **Liver CT Image Dataset**, disponível no Kaggle, para uma tarefa inicial de classificação binária entre:

- `Healthy`
- `Hepatic_Steatosis`

O dataset está organizado em duas pastas, uma por classe. As imagens disponíveis estão em formato `.jpg`.

## 2. Premissas confirmadas

Até esta etapa, foram confirmados os seguintes pontos:

- As imagens estão em formato JPEG.
- Não há arquivos DICOM disponíveis.
- Não há arquivos NIfTI disponíveis.
- Não há metadata clínico disponível no diretório local do dataset.
- As imagens parecem representar slices/cortes de tomografia computadorizada.
- Existem múltiplos slices por agrupamento inferido.
- O identificador do agrupamento foi reconstruído a partir do nome do arquivo.
- Esse identificador é técnico e inferido; portanto, não deve ser tratado como `patient_id` clinicamente validado.

## 3. Limitações do dataset

Como o dataset está disponível apenas em JPEG e sem metadata clínico, o estudo deve ser interpretado como uma análise experimental/reprodutível de visão computacional, e não como validação clínica diagnóstica.

Principais limitações:

- ausência de valores HU confiáveis;
- ausência de voxel spacing;
- ausência de informações de scanner;
- ausência de protocolo de aquisição;
- ausência de fase de contraste;
- ausência de metadados clínicos;
- rótulo herdado da estrutura de pastas;
- impossibilidade de validação clínica forte apenas com as imagens JPEG.

## 4. Estrutura local dos dados

As imagens reais não são versionadas no GitHub. O repositório contém apenas código, documentação e arquivos de configuração.

A pasta local do dataset deve ser configurada via `configs/config.local.yaml`, que é ignorado pelo Git.

Estrutura esperada:

    Liver_CT_Image_Dataset/
    ├── Healthy/
    └── Hepatic_Steatosis/

## 5. Contagem inicial de imagens

A auditoria inicial encontrou:

| Classe | Número de imagens/slices |
|---|---:|
| Healthy | 1.611 |
| Hepatic_Steatosis | 1.946 |
| Total | 3.557 |

Todos os arquivos encontrados possuem extensão `.jpg`.

## 6. Regra de parsing do nome do arquivo

Os arquivos seguem padrão semelhante a:

    1-img-00004-00080.jpg

A regra usada nesta etapa foi:

    inferred_group_id = primeiros três blocos do nome
    slice_id = quarto bloco do nome

Exemplo:

| Filename | inferred_group_id | slice_id |
|---|---|---|
| `1-img-00004-00080.jpg` | `1-img-00004` | `00080` |

O campo `inferred_group_id` é um identificador técnico reconstruído a partir do nome do arquivo. Ele não deve ser interpretado como identificador clínico validado de paciente.

## 7. Índices locais criados

Foram criados dois arquivos locais em `data/interim/`, ambos ignorados pelo Git:

    data/interim/dataset_index.csv
    data/interim/group_index.csv

### 7.1. `dataset_index.csv`

Uma linha por imagem/slice.

Colunas:

- `class_name`
- `label`
- `filename`
- `inferred_group_id`
- `slice_id`
- `extension`
- `file_size_bytes`
- `file_path`

### 7.2. `group_index.csv`

Uma linha por agrupamento inferido.

Colunas:

- `inferred_group_id`
- `class_name`
- `label`
- `n_slices`
- `min_slice_id`
- `max_slice_id`
- `total_size_bytes`

## 8. Estatísticas por grupo inferido

A auditoria encontrou 225 grupos inferidos:

| Classe | Grupos inferidos | Mín. slices/grupo | Média slices/grupo | Máx. slices/grupo |
|---|---:|---:|---:|---:|
| Healthy | 76 | 11 | 21,20 | 36 |
| Hepatic_Steatosis | 149 | 10 | 13,06 | 24 |

Observação importante: a classe `Healthy` possui menos grupos, mas mais slices por grupo em média. A classe `Hepatic_Steatosis` possui mais grupos, mas menos slices por grupo em média.

Por isso, a avaliação do modelo não deve ser centrada apenas em slices. A unidade metodológica principal deve ser o `inferred_group_id`.

## 9. Regra metodológica central contra data leakage

Como existem múltiplos slices por agrupamento, o split não deve ser feito por imagem individual.

Regra adotada:

    Todos os slices do mesmo inferred_group_id devem permanecer no mesmo split.

Portanto, o fluxo correto é:

    grupos inferidos -> split -> slices

E não:

    slices -> split

Essa decisão reduz o risco de data leakage por correlação entre slices do mesmo exame/paciente inferido.

## 10. Split inicial

Foi criado um split estratificado por classe, usando `inferred_group_id` como unidade de separação.

Configuração:

    seed = 42
    train = 70%
    validation = 15%
    test = 15%

Arquivos locais criados:

    data/interim/train_groups.csv
    data/interim/val_groups.csv
    data/interim/test_groups.csv

    data/interim/train_slices.csv
    data/interim/val_slices.csv
    data/interim/test_slices.csv

Esses arquivos são locais e não devem ser versionados.

## 11. Distribuição dos grupos por split

| Split | Healthy | Hepatic_Steatosis | Total |
|---|---:|---:|---:|
| Train | 53 | 104 | 157 |
| Validation | 11 | 22 | 33 |
| Test | 12 | 23 | 35 |

## 12. Distribuição dos slices por split

| Split | Healthy | Hepatic_Steatosis | Total |
|---|---:|---:|---:|
| Train | 1.143 | 1.360 | 2.503 |
| Validation | 206 | 270 | 476 |
| Test | 262 | 316 | 578 |

## 13. Validação de leakage

Após o split por `inferred_group_id`, foi verificada a interseção de grupos entre os conjuntos.

Resultado:

    Train-Val leakage: 0
    Train-Test leakage: 0
    Val-Test leakage: 0

Portanto, não há compartilhamento de `inferred_group_id` entre treino, validação e teste.

## 14. Implicações para a modelagem

A modelagem poderá usar slices como entrada, mas a avaliação principal deve considerar o agrupamento inferido.

Estratégia recomendada:

1. Treinar modelos usando slices dos grupos de treino.
2. Validar usando slices dos grupos de validação.
3. Testar usando apenas grupos mantidos fora do treino e da validação.
4. Agregar probabilidades por `inferred_group_id` para produzir uma predição final por grupo.
5. Reportar métricas por grupo além de métricas por slice.

Formulação sugerida:

    O modelo gera probabilidades por slice; as probabilidades são agregadas por inferred_group_id para produzir a classificação final do agrupamento.

## 15. Próximos passos técnicos

Antes da modelagem, recomenda-se:

1. Transformar os comandos exploratórios em scripts reprodutíveis.
2. Adicionar leitura de dimensões das imagens.
3. Verificar imagens corrompidas.
4. Verificar duplicatas exatas por hash.
5. Verificar quase duplicatas, se necessário.
6. Avaliar diferenças técnicas entre classes, como tamanho de arquivo, dimensões e distribuição de intensidade.
7. Criar notebook de auditoria visual.
8. Só então iniciar baseline de modelagem.

## 16. Decisão metodológica atual

O projeto seguirá inicialmente como estudo experimental de classificação de imagens JPEG de CT hepática, com:

- dados públicos;
- ausência de metadata clínico;
- split por agrupamento inferido;
- avaliação por grupo;
- documentação explícita das limitações;
- proibição de split aleatório por imagem/slice.
