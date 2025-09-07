import cv2
import numpy as np
import sys

"""
Código traduzido de C++ para python. Desenvolvido por: https://stackoverflow.com/a/21479072

"""

def _check_interior_exterior(mask, interior_bb):
    """
    Função auxiliar interna para verificar as bordas da caixa delimitadora.
    Prefixo '_' indica que não é para ser usada fora deste módulo.
    """
    x, y, w, h = [int(c) for c in interior_bb]

    if w <= 0 or h <= 0:
        return True, 0, 0, 0, 0

    sub_mask = mask[y:y + h, x:x + w]

    c_top = np.sum(sub_mask[0, :] == 0)
    c_bottom = np.sum(sub_mask[-1, :] == 0)
    c_left = np.sum(sub_mask[:, 0] == 0)
    c_right = np.sum(sub_mask[:, -1] == 0)

    finished = (c_top + c_bottom + c_left + c_right) == 0
    if finished:
        return True, 0, 0, 0, 0

    top, bottom, left, right = 0, 0, 0, 0

    if c_top > c_bottom and c_top > c_left and c_top > c_right:
        top = 1
    elif c_bottom > c_left and c_bottom > c_right:
        bottom = 1

    if c_left >= c_right and c_left >= c_bottom and c_left >= c_top:
        left = 1
    elif c_right >= c_top and c_right >= c_bottom:
        right = 1

    if not any([top, bottom, left, right]):
        counts = {'top': c_top, 'bottom': c_bottom, 'left': c_left, 'right': c_right}
        max_val = max(counts.values())
        if counts['top'] == max_val: top = 1
        if counts['bottom'] == max_val: bottom = 1
        if counts['left'] == max_val: left = 1
        if counts['right'] == max_val: right = 1

    return finished, top, bottom, left, right

def obter_retangulo(imagem_bgr):
    """
    Recebe uma imagem (BGR, como lida pelo OpenCV), encontra o maior objeto
    com base no contorno em um fundo preto e retorna a imagem recortada.

    Args:
        imagem_bgr (np.ndarray): A imagem de entrada no formato BGR.

    Returns:
        np.ndarray: A imagem recortada. Retorna None se nenhum objeto for encontrado
                    ou se a caixa delimitadora tiver tamanho zero.
    """

    if imagem_bgr is None:
        print("Erro: A imagem de entrada é inválida (None).")
        return None

    gray = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        print("Aviso: Nenhum contorno foi encontrado na imagem.")
        return None

    max_contour = max(contours, key=len)
    
    contour_mask = np.zeros(imagem_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_mask, [max_contour], -1, 255, -1)

    c_sorted_x = sorted(max_contour, key=lambda p: p[0][0])
    c_sorted_y = sorted(max_contour, key=lambda p: p[0][1])

    min_x_id, max_x_id = 0, len(c_sorted_x) - 1
    min_y_id, max_y_id = 0, len(c_sorted_y) - 1

    interior_bb = None

    while min_x_id < max_x_id and min_y_id < max_y_id:
        min_p = (c_sorted_x[min_x_id][0][0], c_sorted_y[min_y_id][0][1])
        max_p = (c_sorted_x[max_x_id][0][0], c_sorted_y[max_y_id][0][1])

        interior_bb = (min_p[0], min_p[1], max_p[0] - min_p[0], max_p[1] - min_p[1])

        finished, oc_top, oc_bottom, oc_left, oc_right = _check_interior_exterior(contour_mask, interior_bb)
        if finished:
            break

        if oc_left: min_x_id += 1
        if oc_right: max_x_id -= 1
        if oc_top: min_y_id += 1
        if oc_bottom: max_y_id -= 1

    if interior_bb:
        x, y, w, h = [int(c) for c in interior_bb]

        if w > 0 and h > 0:
            return imagem_bgr[y:y + h, x:x + w]

    print("Aviso: Não foi possível determinar uma caixa delimitadora válida.")
    return None

