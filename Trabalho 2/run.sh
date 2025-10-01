#!/usr/bin/env bash
# run.sh - Executa os notebooks etapa2, etapa3, etapa4nova e visualizacoes_resultados
# Uso: ./run.sh
# Requisitos: jupyter (nbconvert) ou papermill
set -euo pipefail

# Tempo limite por célula (segundos). Pode customizar via env: NOTEBOOK_TIMEOUT
: "${NOTEBOOK_TIMEOUT:=600}"

# Define os notebooks a serem executados
NOTEBOOKS=("etapa2.ipynb" "etapa3.ipynb" "etapa4nova.ipynb" "visualizacoes_resultados.ipynb")

# Escolhe executor disponível
if command -v jupyter >/dev/null 2>&1; then
    EXECUTOR="nbconvert"
elif command -v papermill >/dev/null 2>&1; then
    EXECUTOR="papermill"
else
    echo "Erro: instale 'jupyter' (nbconvert) ou 'papermill'." >&2
    exit 1
fi

failures=0

for NOTEBOOK in "${NOTEBOOKS[@]}"; do
    if [ ! -f "$NOTEBOOK" ]; then
        echo "Notebook $NOTEBOOK não encontrado."
        failures=$((failures+1))
        continue
    fi

    echo "== Executando: $NOTEBOOK =="
    if [ "$EXECUTOR" = "nbconvert" ]; then
        if ! jupyter nbconvert --to notebook --execute --inplace \
                --ExecutePreprocessor.timeout="${NOTEBOOK_TIMEOUT}" "$NOTEBOOK"; then
            echo "FALHOU: $NOTEBOOK" >&2
            failures=$((failures+1))
        fi
    else
        # papermill: executa em arquivo temporário e substitui o original se OK
        tmp="${NOTEBOOK}.run.tmp.ipynb"
        if ! papermill "$NOTEBOOK" "$tmp" --log-output; then
            echo "FALHOU: $NOTEBOOK" >&2
            rm -f "$tmp"
            failures=$((failures+1))
        else
            mv "$tmp" "$NOTEBOOK"
        fi
    fi

done

if [ "$failures" -gt 0 ]; then
    echo "Execução completa com $failures falha(s)." >&2
    exit 2
fi

echo "Todos os notebooks executados com sucesso."