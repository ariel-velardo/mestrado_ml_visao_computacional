@'
# Mestrado ML Visão Computacional

Projeto de visão computacional aplicada a imagens médicas, com foco inicial em tomografias de fígado e classificação entre exames saudáveis e esteatose hepática.

## Objetivo

Construir um pipeline reprodutível para:

1. indexação do dataset;
2. análise exploratória das classes;
3. pré-processamento das imagens;
4. modelagem baseline;
5. avaliação de desempenho;
6. interpretabilidade visual;
7. documentação metodológica.

## Estrutura

- `configs/`: arquivos de configuração exemplo.
- `data/`: dados locais não versionados.
- `notebooks/`: notebooks numerados do projeto.
- `src/liverct/`: código-fonte modular.
- `scripts/`: scripts executáveis.
- `docs/`: documentação técnica e metodológica.
- `reports/`: figuras e tabelas geradas.
- `models/`: modelos locais não versionados.
- `tests/`: testes automatizados.

## Observação sobre dados

Dados médicos, imagens, metadados sensíveis e modelos treinados não devem ser versionados neste repositório.
'@ | Set-Content README.md -Encoding UTF8