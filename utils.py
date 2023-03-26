import glob
import cv2
import numpy as np
from skimage import measure

import os
import settings
import matplotlib.pyplot as plt
import math
import re
from PIL import Image
import pandas as pd


def get_colonias_caminho(diretorio):
    """Funcao que pega os caminhos dos arquivos com as as fotos das placas.
    Recebe uma string com o diretorio onde estao as fotos
    """
    arqs = glob.glob(f"{diretorio}/*.jpg")

    return arqs


def abre_img(caminhos):
    """Funcao que abre uma imagem"""
    if isinstance(caminhos, str):
        caminhos = [caminhos]

    for caminho in caminhos:
        imagem = cv2.imread(caminho)

    return imagem


def plota(img):
    """Funcao simples para plotar uma imagem."""
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    return 1


def salva_img(img, diretorio, nome_img):
    """Funcao que salva o arquivo gerado.
    Recebe uma imagem, o nome com que a imagem será salva e o diretorio em que
    ela será salva
    """

    cv2.imwrite(
        os.path.join(diretorio, f"{nome_img}"),
        img,
        [cv2.IMWRITE_JPEG_QUALITY, 100],
    )

    return 1


def get_escala_pixel(foto_bruta):
    """Funcao que recebe uma imagem de uma foto de fungo em placa de petri.
    Ela binariza a imagem de modo a pegar a placa de petri.
    Após a binarização, ela considera o objeto com maior eixo como sendo a
    placa de petri.
    Uma média do maior e menor eixo é calculada, para tentar diminuir erros.
    Além disso, uma imagem nova, apenas com a placa e o que tem nela é criada
    Retorna:
    1. A média do eixo maior e menor da placa, para servir
    de escala (em pixels)
    2. Uma imagem nova, com tudo o que não for a placa e o que tem nela
    retirado (essa nova imagem pode ser útil para detectar problemas
    no script, não será usada a princípio)
    """
    banda_vermelha = foto_bruta[:, :, 2]
    banda_vermelha = cv2.GaussianBlur(banda_vermelha, (5, 5), 0)

    ## primeira binarização (definir placa de petri e escala)
    _, binarizada_otsu = cv2.threshold(banda_vermelha, 0, 255, cv2.THRESH_OTSU)
    img_limpa_2 = measure.label(binarizada_otsu)
    props_limpa = measure.regionprops(img_limpa_2)

    objs = list(range(1, img_limpa_2.max() + 1))
    eixos_maior = [int(obj.axis_major_length) for obj in props_limpa]
    # pega os maiores eixos dos objetos
    eixos_menor = [int(obj.axis_minor_length) for obj in props_limpa]
    # pega os menores eixos dos objetos
    zip_img = list(zip(objs, eixos_maior, eixos_menor))

    placa_petri = max(zip_img, key=lambda tup: tup[1])
    # pega o objeto que é a placa de petri (é o maior objeto da foto)
    escala = (placa_petri[1] + placa_petri[2]) / 2
    # tira a média entre os valores, para tentar chegar o mais perto possível
    # do valor real

    img_petri = np.zeros_like(foto_bruta)
    mask = np.isin(img_limpa_2, placa_petri[0])
    # mantendo só o que é o maior objeto
    img_petri[mask] = foto_bruta[mask]

    return [escala, img_petri]


def get_colonia_fungo(foto_bruta, arq):
    """Funcao que recebe uma imagem de uma foto de fungo em placa de petri
    e o nome do arquivo.
    Binariza a imagem e retira apenas a colônia do fungo.
    Retorna um dicionário com as seguintes informações sobre a colônia:
        1. número do respectivo objeto na imagem bruta, já que podem ser
        identificado mais objetos além dela, como sujeitas e etc
        2. area (em pixels) da colônia
        3. eixo maior (em pixels) da colônia
        4. eixo menor (em pixels) da colônia
        5. ponto central ('centroide') da colônia
        6. inclinação da colônia
    Além do dicionário:
        1. Uma imagem da colônia colorida é salva em um diretório dentro do
        diretório onde estão as imagens brutas, mantendo o nome com o sufixo
        "_colonia" aicionado.
        2. Um dicionário contendo a própria imagem salva e a imagem
        binarizada é retornado
    """
    img_limpa_azul = foto_bruta[:, :, 0]
    # img_limpa_azul = cv2.cvtColor(foto_bruta, cv2.COLOR_BGR2GRAY)
    img_limpa_azul = cv2.GaussianBlur(img_limpa_azul, (5, 5), 0)
    _, binarizada_otsu_2 = cv2.threshold(
        img_limpa_azul, 0, 255, cv2.THRESH_OTSU
    )
    # binarizando, para pegar a colônia de fungos

    kernel_fill = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    binarizada_otsu_2 = cv2.morphologyEx(
        binarizada_otsu_2, cv2.MORPH_CLOSE, kernel_fill
    )

    # acima, diminuindo "buracos" na colônia
    # img_limpa_azul = cv2.GaussianBlur(binarizada_otsu_2, (5, 5), 0)

    labels_2 = measure.label(binarizada_otsu_2)
    props_limpa = measure.regionprops(labels_2)
    objs = list(range(1, labels_2.max() + 1))

    areas = [int(obj.area) for obj in props_limpa]
    eixos_maior = [int(obj.axis_major_length) for obj in props_limpa]
    # pega os maiores eixos dos objetos (agora para a colonia)
    eixos_menor = [int(obj.axis_minor_length) for obj in props_limpa]
    # pega os menores eixos dos objetos (agora para a colonia)
    centroides = [obj.centroid for obj in props_limpa]
    # # pega o centroide dos objetos
    bboxes = [obj.bbox for obj in props_limpa]
    # orientations = [obj.orientation for obj in props_limpa]
    # pega a inclinação dos obejtos (para plotar os eixos na figura, depois)
    zip_img = list(
        zip(objs, areas, eixos_maior, eixos_menor, centroides, bboxes)
    )

    eixos_ord = sorted(zip_img, key=lambda x: x[2], reverse=True)
    # ordenando os objetos pelo maior eixo, do maior para o menor. A placa de petri
    # vai ter o maior, a colônia de fungos é para ter a segunda maior

    colonia_info = {
        "obj": eixos_ord[1][0],
        "area": eixos_ord[1][1],
        # "eixo_maior": eixos_ord[1][2],
        # "eixo_menor": eixos_ord[1][3],
        "centroide_colonia": eixos_ord[1][4],
        # "orientation": eixos_ord[1][5],
        "bbox_colonia": eixos_ord[1][5],
    }
    # breakpoint()
    # pega o objeto que é a colônia do fungo. ATENÇÃO, o fungo vai ter o segundo
    # maior eixo da imagem, porque o primeiro é o da placa de petri
    colonia_img = np.zeros_like(foto_bruta)
    mask = np.isin(labels_2, colonia_info["obj"])
    # mantendo só o que é o fungo
    colonia_img[mask] = foto_bruta[mask]
    ## retornando a imagem colorida, com os 3 canais
    salva_img(colonia_img, settings.DIRETORIO_TRATADAS, f"{arq}_colonia.jpg")

    return {
        "dict_colonia": colonia_info,
        "colonia_rgb": colonia_img,
        "colonia_bin": mask,
    }


def get_eixos_x_y(imagem_binarizada, colonia_info):
    """Funcao que recebe uma imagem da binarizada da colonia e o bbox obtidos
    com a funcao 'get_colonia_fungo'), binariza a imagem e calcula:
    1. Ponto central do bbox no eixo x
    2. Ponto central do bbox no eixo y
    3. Onde a colônia se inicia e termina no eixo x
    4. Onde a colônia se inicia e terminal no eixo y
    5. O diâmetro da colônia na região central no eixo x
    6. O diâmetro da colônia na regiâo central no eixo y
    """

    ymin, xmin, altura, largura = colonia_info["bbox_colonia"]

    # pegando o centro do bbox no eixo x e no y
    centro_bbox_y = ymin + round((altura - ymin) / 2)
    centro_bbox_x = xmin + round((largura - xmin) / 2)

    # Calculando o diâmetro da colônia na região central do bbox no eixo x
    # e no eixo y
    linha_central_eixo_x = [
        i for i, x in enumerate(imagem_binarizada[centro_bbox_y, :]) if x
    ]
    # pegando os indices no eixo y que sejam > 1, equivalem a coordenada de
    # início e fim da colônia naquela linha da bbox
    inicio_fim_x = (linha_central_eixo_x[0], linha_central_eixo_x[-1])
    # colocando o índice inicial e final em uma tupla (será o início e
    # final da colônia)

    linha_central_eixo_y = [
        i for i, x in enumerate(imagem_binarizada[:, centro_bbox_x]) if x
    ]
    # pegando os indices no eixo x que sejam > 1, equivalem a coordenada de
    # início e fim da colônia naquela coluna da bbox
    inicio_fim_y = (linha_central_eixo_y[0], linha_central_eixo_y[-1])
    # colocando o índice inicial e final em uma tupla (será o início e
    # final da colônia)

    return {
        "centro_bbox_x": centro_bbox_x,
        "centro_bbox_y": centro_bbox_y,
        "inicio_fim_x": inicio_fim_x,
        "inicio_fim_y": inicio_fim_y,
        "diametro_eixo_x": inicio_fim_x[1] - inicio_fim_x[0],
        "diametro_eixo_y": inicio_fim_y[1] - inicio_fim_y[0],
    }


def salva_figura_eixos(
    arq,
    informacoes_colonia,
    imagem_colonia_isolada,
    escala_px,
    placa_petri=9,
    mostra_eixos=False,
):
    """Funcao que salva a imagem da colonia de fungos isolada do resto da
    imagem, com uma barra, indicando o que é 1 cm na imagem.
    Recebe:
    1. nome do arquivo com a imagem bruta
    2. imagem da colônia isolada (funcao 'get_colonia_fungo')
    3. o dicionário com as informações sobre os eixos da colônia
    (funcao 'get_eixos_x_y')
    4. a escala, ou seja, diâmetro em pixels da placa de petri (funcao
    get_escala_pixel)
    5. o diâmetro, em cm, da placa de petri, o padrão é 9.
    A imagem é salva em um diretório dentro do diretório onde estão as
    imagens brutas, mantendo o nome com o sufixo "_colonia_eixos" aicionado.
    Por padrão, os iexos usados nos cálculos não são mostrados na imagem salva,
    mas caso 'mostra_eixos' for substituído para 'True', eles serão mostrados
    na imagem salva também.
    """

    caminho = os.path.join(
        settings.DIRETORIO_TRATADAS, f"{arq}_colonia_eixos.png"
    )
    barra_cm = escala_px / placa_petri
    shape_img = imagem_colonia_isolada.shape
    pos_y = shape_img[0] * 0.96
    pos_x = shape_img[1] * 0.1

    imagem_colonia_isolada = cv2.cvtColor(
        imagem_colonia_isolada, cv2.COLOR_BGR2RGB
    )
    plt.imshow(imagem_colonia_isolada)
    plt.axis("off")

    if mostra_eixos == True:
        plt.plot(
            (
                informacoes_colonia["inicio_fim_x"][0],
                informacoes_colonia["inicio_fim_x"][1],
            ),
            (
                informacoes_colonia["centro_bbox_y"],
                informacoes_colonia["centro_bbox_y"],
            ),
            "r",
            linewidth=1.0,
        )
        plt.plot(
            (
                informacoes_colonia["centro_bbox_x"],
                informacoes_colonia["centro_bbox_x"],
            ),
            (
                informacoes_colonia["inicio_fim_y"][0],
                informacoes_colonia["inicio_fim_y"][1],
            ),
            "g",
            linewidth=1.0,
        )

    plt.annotate(
        "1 cm",
        (pos_x, pos_y - 5),
        ha="left",
        va="bottom",
        size=6,
        color="white",
    )
    plt.plot(
        (pos_x, pos_x + barra_cm),
        (pos_y, pos_y),
        "w",
        linewidth=1.5,
    )

    print(f"imagem da colônia, com escala, salva em:\n{caminho}")

    return plt.savefig(
        caminho,
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )


def diametro_colonia_cm(arq, eixos_centrais_colonia, escala_px, placa_petri=9):
    """Funcao que recebe:
    1. o nome do arquivo
    2. uma lista com os valores dos eixos, maior e menor (em pixels),
    da colônia do fungo (funcao 'get_colonia_fungo'). A funcao pode receber
    o apenas um valor de eixo também, ao invés de 2.
    3. escala em pixels (funcao 'get_escala_pixel'),
    4. o diâmetro da placa de petri **em cm** (valor padrão para é de 9 cm).
    A função calcula um valor médio para o diâmetro, em cm, da colônia.
    Retorna um dicionário com o nome do arquivo, o diâmetro no eixo x,
    diâmetro no eixo y e a média do diâmetro dos eixos da colônia.
    """

    if type(eixos_centrais_colonia) is not list:
        eixos_centrais_colonia = [eixos_centrais_colonia]

    eixos_cm = []
    for eixo in eixos_centrais_colonia:
        diametro_colonia_cm = round(placa_petri * eixo / escala_px, 2)
        eixos_cm.append(diametro_colonia_cm)

    med_eixos = [round(sum(eixos_cm) / len(eixos_centrais_colonia), 2)]

    return {
        "placa": arq,
        "eixo_x": eixos_cm[0],
        "eixo_y": eixos_cm[1],
        "media": med_eixos,
    }


def info_list_to_df(lista_diametro):
    """Funcao que recebe um dicionário com as informações sobre a colônia
    (por enquanto, diâmetro de eixos e diâmetro médio), transforma em um
    pd DataFrame e salva em um arquivo '.csv'.
    """
    df = pd.DataFrame.from_dict(lista_diametro)

    return df
