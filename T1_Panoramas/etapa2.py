import cv2
import matplotlib.pyplot as plt

# Carregar as imagens (substitua 'image1.jpg' e 'image2.jpg' pelos caminhos das suas imagens)
img1 = cv2.imread('1.jpg')
img2 = cv2.imread('2.jpg')
img3 = cv2.imread('3.jpg')
img4 = cv2.imread('4.jpg')
img5 = cv2.imread('5.jpg')
img6 = cv2.imread('6.jpg')
img7 = cv2.imread('7.jpg')
img8 = cv2.imread('8.jpg')

# Inicializar detectores
sift = cv2.SIFT_create()
orb = cv2.ORB_create()

# Detectar keypoints e descritores
keypoints_sift1, descriptors_sift1 = sift.detectAndCompute(img1, None)
keypoints_sift2, descriptors_sift2 = sift.detectAndCompute(img2, None)
keypoints_sift3, descriptors_sift3 = sift.detectAndCompute(img3, None)
keypoints_sift4, descriptors_sift4 = sift.detectAndCompute(img4, None)
keypoints_sift5, descriptors_sift5 = sift.detectAndCompute(img5, None)
keypoints_sift6, descriptors_sift6 = sift.detectAndCompute(img6, None)
keypoints_sift7, descriptors_sift7 = sift.detectAndCompute(img7, None)
keypoints_sift8, descriptors_sift8 = sift.detectAndCompute(img8, None)

keypoints_orb1, descriptors_orb1 = orb.detectAndCompute(img1, None)
keypoints_orb2, descriptors_orb2 = orb.detectAndCompute(img2, None)
keypoints_orb3, descriptors_orb3 = orb.detectAndCompute(img3, None)
keypoints_orb4, descriptors_orb4 = orb.detectAndCompute(img4, None)
keypoints_orb5, descriptors_orb5 = orb.detectAndCompute(img5, None)
keypoints_orb6, descriptors_orb6 = orb.detectAndCompute(img6, None)
keypoints_orb7, descriptors_orb7 = orb.detectAndCompute(img7, None)
keypoints_orb8, descriptors_orb8 = orb.detectAndCompute(img8, None)

# Visualizar keypoints
img_sift1 = cv2.drawKeypoints(img1, keypoints_sift1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_sift2 = cv2.drawKeypoints(img2, keypoints_sift2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_sift3 = cv2.drawKeypoints(img3, keypoints_sift3, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_sift4 = cv2.drawKeypoints(img4, keypoints_sift4, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_sift5 = cv2.drawKeypoints(img5, keypoints_sift5, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_sift6 = cv2.drawKeypoints(img6, keypoints_sift6, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_sift7 = cv2.drawKeypoints(img7, keypoints_sift7, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_sift8 = cv2.drawKeypoints(img8, keypoints_sift8, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

img_orb1 = cv2.drawKeypoints(img1, keypoints_orb1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_orb2 = cv2.drawKeypoints(img2, keypoints_orb2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_orb3 = cv2.drawKeypoints(img3, keypoints_orb3, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_orb4 = cv2.drawKeypoints(img4, keypoints_orb4, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_orb5 = cv2.drawKeypoints(img5, keypoints_orb5, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_orb6 = cv2.drawKeypoints(img6, keypoints_orb6, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_orb7 = cv2.drawKeypoints(img7, keypoints_orb7, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_orb8 = cv2.drawKeypoints(img8, keypoints_orb8, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Mostrar resultados
plt.figure(figsize=(20, 12))

# SIFT Keypoints - primeira linha (4 imagens)
plt.subplot(4, 4, 1)
plt.title('SIFT - Imagem 1')
plt.imshow(cv2.cvtColor(img_sift1, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(4, 4, 2)
plt.title('SIFT - Imagem 2')
plt.imshow(cv2.cvtColor(img_sift2, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(4, 4, 3)
plt.title('SIFT - Imagem 3')
plt.imshow(cv2.cvtColor(img_sift3, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(4, 4, 4)
plt.title('SIFT - Imagem 4')
plt.imshow(cv2.cvtColor(img_sift4, cv2.COLOR_BGR2RGB))
plt.axis('off')

# SIFT Keypoints - segunda linha (4 imagens)
plt.subplot(4, 4, 5)
plt.title('SIFT - Imagem 5')
plt.imshow(cv2.cvtColor(img_sift5, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(4, 4, 6)
plt.title('SIFT - Imagem 6')
plt.imshow(cv2.cvtColor(img_sift6, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(4, 4, 7)
plt.title('SIFT - Imagem 7')
plt.imshow(cv2.cvtColor(img_sift7, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(4, 4, 8)
plt.title('SIFT - Imagem 8')
plt.imshow(cv2.cvtColor(img_sift8, cv2.COLOR_BGR2RGB))
plt.axis('off')

# ORB Keypoints - terceira linha (4 imagens)
plt.subplot(4, 4, 9)
plt.title('ORB - Imagem 1')
plt.imshow(cv2.cvtColor(img_orb1, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(4, 4, 10)
plt.title('ORB - Imagem 2')
plt.imshow(cv2.cvtColor(img_orb2, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(4, 4, 11)
plt.title('ORB - Imagem 3')
plt.imshow(cv2.cvtColor(img_orb3, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(4, 4, 12)
plt.title('ORB - Imagem 4')
plt.imshow(cv2.cvtColor(img_orb4, cv2.COLOR_BGR2RGB))
plt.axis('off')

# ORB Keypoints - quarta linha (4 imagens)
plt.subplot(4, 4, 13)
plt.title('ORB - Imagem 5')
plt.imshow(cv2.cvtColor(img_orb5, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(4, 4, 14)
plt.title('ORB - Imagem 6')
plt.imshow(cv2.cvtColor(img_orb6, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(4, 4, 15)
plt.title('ORB - Imagem 7')
plt.imshow(cv2.cvtColor(img_orb7, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(4, 4, 16)
plt.title('ORB - Imagem 8')
plt.imshow(cv2.cvtColor(img_orb8, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()

# Contar keypoints detectados
print(f"SIFT - Imagem 1: {len(keypoints_sift1)} keypoints")
print(f"SIFT - Imagem 2: {len(keypoints_sift2)} keypoints")
print(f"SIFT - Imagem 3: {len(keypoints_sift3)} keypoints")
print(f"SIFT - Imagem 4: {len(keypoints_sift4)} keypoints")
print(f"SIFT - Imagem 5: {len(keypoints_sift5)} keypoints")
print(f"SIFT - Imagem 6: {len(keypoints_sift6)} keypoints")
print(f"SIFT - Imagem 7: {len(keypoints_sift7)} keypoints")
print(f"SIFT - Imagem 8: {len(keypoints_sift8)} keypoints")
print(f"ORB - Imagem 1: {len(keypoints_orb1)} keypoints")
print(f"ORB - Imagem 2: {len(keypoints_orb2)} keypoints")
print(f"ORB - Imagem 3: {len(keypoints_orb3)} keypoints")
print(f"ORB - Imagem 4: {len(keypoints_orb4)} keypoints")
print(f"ORB - Imagem 5: {len(keypoints_orb5)} keypoints")
print(f"ORB - Imagem 6: {len(keypoints_orb6)} keypoints")
print(f"ORB - Imagem 7: {len(keypoints_orb7)} keypoints")
print(f"ORB - Imagem 8: {len(keypoints_orb8)} keypoints")