#!/bin/bash

NOTEBOOK_ENTRADA="web_cam.ipynb"
NOTEBOOK_SAIDA="resultado_web_cam.ipynb"

# Define um tempo limite de 60 segundos
# Se a célula demorar mais que isso, ela será interrompida
TIMEOUT=60 

echo "Executando o notebook: $NOTEBOOK_ENTRADA (com timeout de ${TIMEOUT}s)"
echo "A saída será salva em: $NOTEBOOK_SAIDA"

jupyter nbconvert --to notebook --execute "$NOTEBOOK_ENTRADA" \
  --output "$NOTEBOOK_SAIDA" \
  --ExecutePreprocessor.timeout=$TIMEOUT \
  --allow-errors

# Verifica se o comando foi bem-sucedido
if [ $? -eq 0 ]; then
  echo "Execução concluída."
  echo "AVISO: Células longas podem ter sido interrompidas por timeout."
else
  echo "Erro durante a execução do notebook."
fi