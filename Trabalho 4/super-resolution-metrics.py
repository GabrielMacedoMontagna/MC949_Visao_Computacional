# pip install --no-cache-dir scikit-image


from pathlib import Path
import argparse

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_image(path: Path):
    img = Image.open(path).convert("RGB")
    return np.array(img)


def eval_scale(root_dir: Path, scale: int):
    """
    Compara HR vs SR para um dado scale (2, 3 ou 4).

    HR:  image_SRF_{scale}/img_XXX_SRF_{scale}_HR.png
    SR:  results_SR_x{scale}/img_XXX_SRF_{scale}_SRx{scale}.png
    """
    hr_dir = root_dir / f"image_SRF_{scale}"
    sr_dir = root_dir / f"results_SR_x{scale}"

    if not hr_dir.exists():
        raise RuntimeError(f"Pasta HR não encontrada: {hr_dir}")
    if not sr_dir.exists():
        raise RuntimeError(f"Pasta SR não encontrada: {sr_dir} (rode o script de super-res primeiro)")

    # pega todas HR
    hr_paths = sorted(hr_dir.glob(f"*SRF_{scale}_HR.png"))
    if not hr_paths:
        raise RuntimeError(f"Nenhuma HR encontrada em {hr_dir} com padrão *SRF_{scale}_HR.png")

    psnrs = []
    ssims = []

    print(f"\n=== Avaliando scale {scale}x ===")
    for hr_path in hr_paths:
        base = hr_path.name.replace("_HR.png", "")   # img_001_SRF_2
        sr_name = f"{base}_SRx{scale}.png"          # img_001_SRF_2_SRx2.png
        sr_path = sr_dir / sr_name

        if not sr_path.exists():
            print(f"[AVISO] Não encontrei SR para {hr_path.name} (esperado: {sr_name})")
            continue

        hr = load_image(hr_path)
        sr = load_image(sr_path)

        if hr.shape != sr.shape:
            print(f"[AVISO] Tamanho diferente para {hr_path.name}: HR{hr.shape} vs SR{sr.shape} — pulando.")
            continue

        psnr = peak_signal_noise_ratio(hr, sr, data_range=255)
        ssim = structural_similarity(hr, sr, channel_axis=2)

        psnrs.append(psnr)
        ssims.append(ssim)

        print(f"{hr_path.name}  ->  PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")

    if not psnrs:
        print("Nenhuma imagem válida para avaliação.")
        return

    print(f"\nMÉDIAS para scale {scale}x:")
    print(f"PSNR médio : {np.mean(psnrs):.2f} dB")
    print(f"SSIM médio : {np.mean(ssims):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale",
        type=int,
        choices=[2, 3, 4],
        required=True,
        help="Fator de escala: 2, 3 ou 4",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="Set14dataset/Set14",
        help="Caminho para a pasta Set14",
    )

    args = parser.parse_args()
    root_dir = Path(args.root).resolve()
    eval_scale(root_dir, args.scale)


if __name__ == "__main__":
    main()
