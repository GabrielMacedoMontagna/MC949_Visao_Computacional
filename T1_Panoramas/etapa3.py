import cv2
import matplotlib.pyplot as plt
import numpy as np

# Carregar as imagens
img1 = cv2.imread('1.jpg')
img2 = cv2.imread('2.jpg')
img3 = cv2.imread('3.jpg')
img4 = cv2.imread('4.jpg')
img5 = cv2.imread('5.jpg')
img6 = cv2.imread('6.jpg')
img7 = cv2.imread('7.jpg')
img8 = cv2.imread('8.jpg')

# Converter para escala de cinza para melhor performance
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
gray7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
gray8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)

# Inicializar detectores
sift = cv2.SIFT_create()
orb = cv2.ORB_create()

# Detectar keypoints e descritores para SIFT
kp_sift = []
desc_sift = []
images_gray = [gray1, gray2, gray3, gray4, gray5, gray6, gray7, gray8]

for i, gray in enumerate(images_gray):
    kp, desc = sift.detectAndCompute(gray, None)
    kp_sift.append(kp)
    desc_sift.append(desc)
    print(f"SIFT - Imagem {i+1}: {len(kp)} keypoints detectados")

# Detectar keypoints e descritores para ORB
kp_orb = []
desc_orb = []

for i, gray in enumerate(images_gray):
    kp, desc = orb.detectAndCompute(gray, None)
    kp_orb.append(kp)
    desc_orb.append(desc)
    print(f"ORB - Imagem {i+1}: {len(kp)} keypoints detectados")

# Função para aplicar ratio test de Lowe
def apply_ratio_test(matches, ratio_threshold=0.75):
    """
    Aplica o ratio test de David Lowe para filtrar bons matches
    """
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    return good_matches

# Função para realizar matching entre duas imagens
def match_images_sift(desc1, desc2, matcher_type='FLANN'):
    """
    Realiza matching entre duas imagens usando SIFT
    """
    if desc1 is None or desc2 is None:
        return []
    
    if matcher_type == 'FLANN':
        # FLANN parameters para SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)
    else:  # Brute Force
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1, desc2, k=2)
    
    # Aplicar ratio test
    good_matches = apply_ratio_test(matches)
    return good_matches

def match_images_orb(desc1, desc2, matcher_type='BF'):
    """
    Realiza matching entre duas imagens usando ORB
    """
    if desc1 is None or desc2 is None:
        return []
    
    if matcher_type == 'FLANN':
        # FLANN parameters para ORB (binary descriptors)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                           table_number=6,
                           key_size=12,
                           multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)
    else:  # Brute Force (recomendado para ORB)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(desc1, desc2, k=2)
    
    # Aplicar ratio test
    good_matches = apply_ratio_test(matches)
    return good_matches

# Realizar matching entre imagens consecutivas
print("\n=== MATCHING COM SIFT ===")
sift_matches = []
for i in range(len(desc_sift) - 1):
    # Testar tanto FLANN quanto Brute Force
    matches_flann = match_images_sift(desc_sift[i], desc_sift[i+1], 'FLANN')
    matches_bf = match_images_sift(desc_sift[i], desc_sift[i+1], 'BF')
    
    sift_matches.append({
        'FLANN': matches_flann,
        'BF': matches_bf
    })
    
    print(f"Imagem {i+1} -> {i+2}: FLANN={len(matches_flann)} matches, BF={len(matches_bf)} matches")

print("\n=== MATCHING COM ORB ===")
orb_matches = []
for i in range(len(desc_orb) - 1):
    # Testar tanto FLANN quanto Brute Force
    matches_flann = match_images_orb(desc_orb[i], desc_orb[i+1], 'FLANN')
    matches_bf = match_images_orb(desc_orb[i], desc_orb[i+1], 'BF')
    
    orb_matches.append({
        'FLANN': matches_flann,
        'BF': matches_bf
    })
    
    print(f"Imagem {i+1} -> {i+2}: FLANN={len(matches_flann)} matches, BF={len(matches_bf)} matches")

# Visualizar os melhores matches
def draw_matches_custom(img1, kp1, img2, kp2, matches, title):
    """
    Desenha matches entre duas imagens
    """
    if len(matches) == 0:
        print(f"Nenhum match encontrado para {title}")
        return None
    
    # Limitar o número de matches exibidos para melhor visualização
    matches_to_show = matches[:50] if len(matches) > 50 else matches
    
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches_to_show, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches

# Criar visualizações para os melhores casos
plt.figure(figsize=(20, 24))

# SIFT Matches - FLANN
subplot_idx = 1
for i in range(min(4, len(sift_matches))):  # Mostrar apenas os primeiros 4 pares
    matches = sift_matches[i]['FLANN']
    if len(matches) > 10:  # Só mostrar se tiver matches suficientes
        img_matches = draw_matches_custom(images_gray[i], kp_sift[i], 
                                        images_gray[i+1], kp_sift[i+1], 
                                        matches, f"SIFT FLANN {i+1}-{i+2}")
        if img_matches is not None:
            plt.subplot(6, 2, subplot_idx)
            plt.title(f'SIFT FLANN - Imagem {i+1} -> {i+2} ({len(matches)} matches)')
            plt.imshow(img_matches, cmap='gray')
            plt.axis('off')
            subplot_idx += 1

# SIFT Matches - Brute Force
for i in range(min(4, len(sift_matches))):
    matches = sift_matches[i]['BF']
    if len(matches) > 10:
        img_matches = draw_matches_custom(images_gray[i], kp_sift[i], 
                                        images_gray[i+1], kp_sift[i+1], 
                                        matches, f"SIFT BF {i+1}-{i+2}")
        if img_matches is not None:
            plt.subplot(6, 2, subplot_idx)
            plt.title(f'SIFT Brute Force - Imagem {i+1} -> {i+2} ({len(matches)} matches)')
            plt.imshow(img_matches, cmap='gray')
            plt.axis('off')
            subplot_idx += 1

# ORB Matches - FLANN
for i in range(min(2, len(orb_matches))):
    matches = orb_matches[i]['FLANN']
    if len(matches) > 5:
        img_matches = draw_matches_custom(images_gray[i], kp_orb[i], 
                                        images_gray[i+1], kp_orb[i+1], 
                                        matches, f"ORB FLANN {i+1}-{i+2}")
        if img_matches is not None:
            plt.subplot(6, 2, subplot_idx)
            plt.title(f'ORB FLANN - Imagem {i+1} -> {i+2} ({len(matches)} matches)')
            plt.imshow(img_matches, cmap='gray')
            plt.axis('off')
            subplot_idx += 1

# ORB Matches - Brute Force
for i in range(min(2, len(orb_matches))):
    matches = orb_matches[i]['BF']
    if len(matches) > 5:
        img_matches = draw_matches_custom(images_gray[i], kp_orb[i], 
                                        images_gray[i+1], kp_orb[i+1], 
                                        matches, f"ORB BF {i+1}-{i+2}")
        if img_matches is not None:
            plt.subplot(6, 2, subplot_idx)
            plt.title(f'ORB Brute Force - Imagem {i+1} -> {i+2} ({len(matches)} matches)')
            plt.imshow(img_matches, cmap='gray')
            plt.axis('off')
            subplot_idx += 1

plt.tight_layout()
plt.show()

# Análise dos resultados e desafios encontrados
print("\n" + "="*60)
print("ANÁLISE DOS RESULTADOS E DESAFIOS ENCONTRADOS")
print("="*60)

print("\n1. COMPARAÇÃO ENTRE ALGORITMOS DE MATCHING:")
print("   • FLANN (Fast Library for Approximate Nearest Neighbors):")
print("     - Mais rápido para grandes conjuntos de dados")
print("     - Usa aproximações, pode perder alguns matches precisos")
print("     - Recomendado para SIFT (descritores de ponto flutuante)")
print("   • Brute Force Matcher:")
print("     - Mais preciso, testa todas as combinações")
print("     - Mais lento, especialmente com muitos keypoints")
print("     - Recomendado para ORB (descritores binários)")

print("\n2. RATIO TEST DE DAVID LOWE:")
print("   • Threshold usado: 0.75")
print("   • Filtra matches ambíguos comparando as duas melhores correspondências")
print("   • Reduz falsos positivos significativamente")

total_sift_flann = sum(len(m['FLANN']) for m in sift_matches)
total_sift_bf = sum(len(m['BF']) for m in sift_matches)
total_orb_flann = sum(len(m['FLANN']) for m in orb_matches)
total_orb_bf = sum(len(m['BF']) for m in orb_matches)

print(f"\n3. ESTATÍSTICAS DE MATCHES:")
print(f"   • SIFT + FLANN: {total_sift_flann} matches totais")
print(f"   • SIFT + Brute Force: {total_sift_bf} matches totais")
print(f"   • ORB + FLANN: {total_orb_flann} matches totais")
print(f"   • ORB + Brute Force: {total_orb_bf} matches totais")

print("\n4. DESAFIOS ENCONTRADOS:")
print("   • KEYPOINTS INCORRETOS:")
print("     - Estruturas repetitivas podem gerar matches falsos")
print("     - Texturas similares em diferentes objetos")
print("     - Reflexos e sombras podem confundir os detectores")

print("   • OBJETOS MÓVEIS:")
print("     - Pessoas ou veículos em movimento entre frames")
print("     - Keypoints detectados em objetos móveis não têm correspondência")
print("     - Necessário filtrar matches em regiões dinâmicas")

print("   • MUDANÇAS DE ILUMINAÇÃO:")
print("     - Variações de luz afetam a detecção de keypoints")
print("     - SIFT é mais robusto a mudanças de iluminação que ORB")

print("   • MUDANÇAS DE PERSPECTIVA:")
print("     - Grandes mudanças de ângulo reduzem matches válidos")
print("     - Distorções geométricas dificultam correspondências")

print("   • QUALIDADE DOS DESCRITORES:")
print("     - ORB produz menos matches devido à natureza binária")
print("     - SIFT gera descritores mais discriminativos mas com maior custo computacional")

print("\n5. RECOMENDAÇÕES:")
print("   • Para aplicações em tempo real: ORB + Brute Force")
print("   • Para máxima precisão: SIFT + FLANN com ratio test")
print("   • Considerar RANSAC para filtrar outliers geométricos")
print("   • Usar máscaras para excluir regiões com objetos móveis")
