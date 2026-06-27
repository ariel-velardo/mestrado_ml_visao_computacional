@'
# AGENTS.md

## Objetivo do projeto

Projeto de visão computacional aplicada a imagens médicas, com foco inicial em tomografias de fígado e classificação entre exames saudáveis e esteatose hepática.

## Regras obrigatórias

- Não commitar dados médicos, imagens, CSVs, modelos treinados ou checkpoints.
- Não criar arquivos com dados sensíveis, nomes de pacientes, metadados pessoais ou identificadores clínicos.
- Não acessar, copiar ou mover arquivos fora do projeto sem instrução explícita.
- Não deletar arquivos existentes sem autorização.
- Antes de alterar estrutura relevante, explicar o plano.
- Preferir scripts pequenos, testáveis e modulares.
- Código deve ser em Python, com funções claras e docstrings.
- Notebooks devem ser usados para exploração e apresentação, não para concentrar toda a lógica.
- Toda lógica reutilizável deve ir para `src/liverct/`.
- Usar `configs/config.example.yaml` como referência pública.
- Usar `configs/config.local.yaml` apenas localmente; ele é ignorado pelo Git.
- Priorizar métricas clínicas: sensibilidade, especificidade, matriz de confusão, ROC-AUC, PR-AUC e análise de erro.
- Nunca apresentar o modelo como diagnóstico autônomo; tratar como apoio experimental à decisão.

## Entregáveis esperados

- README claro.
- Notebooks numerados.
- Código modular em `src/liverct/`.
- Documentação metodológica em `docs/`.
- Resultados em `reports/`.
- Nenhum dado bruto versionado.
'@ | Set-Content AGENTS.md -Encoding UTF8