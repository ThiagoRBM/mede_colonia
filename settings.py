from datetime import datetime
import os
import json
import sys


json_ = open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "diretorio_colonias.json"
    )
)

DIRETORIO = json.load(json_)["dir"]
if not os.path.exists(DIRETORIO):
    print(f"Diretorio: {DIRETORIO}\nnão encontrado, verificar se existe")
    sys.exit()

# HOJE = datetime.now().strftime("%Y%m%d_%H%M%S")

DIRETORIO_TRATADAS = os.path.join(DIRETORIO, "colonias_tratadas")
if not os.path.exists(DIRETORIO_TRATADAS):
    os.makedirs(DIRETORIO_TRATADAS)

# DIRETORIO_PB_FOLHA = os.path.join(DIRETORIO_PB, "folhas_recortadas")
# if not os.path.exists(DIRETORIO_PB_FOLHA):
#     os.makedirs(DIRETORIO_PB_FOLHA)

# DIRETORIO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
