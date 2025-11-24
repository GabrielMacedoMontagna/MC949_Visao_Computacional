# para rodar: python super-resolution.py --scale 2 (ou 3 ou 4)

from pathlib import Path
import argparse
from contextlib import nullcontext

import torch
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline


def load_pipeline(device: str = "cuda"):
    """
    Carrega o modelo de super-resolução x4 do Stable Diffusion.
    """
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()  # economiza memória

    return pipe


def find_lr_hr_pairs(lr_dir: Path, scale: int):
    """
    Encontra pares (LR, HR) com nomes do tipo:
    img_001_SRF_2_LR.png
    img_001_SRF_2_HR.png
    """
    lr_paths = sorted(lr_dir.glob(f"*SRF_{scale}_LR.png"))
    if not lr_paths:
        raise RuntimeError(
            f"Nenhuma LR encontrada em {lr_dir} com padrão *SRF_{scale}_LR.png"
        )

    pairs = []
    for lr_path in lr_paths:
        # base: tira o sufixo "_LR.png"
        base = lr_path.name.replace("_LR.png", "")
        hr_name = f"{base}_HR.png".replace("_LR_HR", "_HR")  # segurança extra
        hr_path = lr_dir / hr_name
        if not hr_path.exists():
            print(f"Aviso: não encontrei HR para {lr_path.name} (esperado: {hr_name})")
            continue
        pairs.append((lr_path, hr_path))

    if not pairs:
        raise RuntimeError("Nenhum par LR/HR completo encontrado.")

    return pairs


def upscale_set14_for_scale(
    root_dir: Path,
    scale: int,
    device: str = "cuda",
    prompt: str = "",
    match_hr_size: bool = True,
):
    """
    Lê pares LR/HR de image_SRF_scale e gera imagens SR.

    - root_dir: caminho para Set14 (ex: Set14dataset/Set14)
    - scale: 2, 3 ou 4
    - match_hr_size: se True, redimensiona SR para o tamanho da HR
    """
    lr_dir = root_dir / f"image_SRF_{scale}"
    out_dir = root_dir / f"results_SR_x{scale}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_lr_hr_pairs(lr_dir, scale)
    print(f"Scale = {scale}x | Encontrados {len(pairs)} pares LR/HR em {lr_dir}")

    pipe = load_pipeline(device=device)

    if device == "cuda":
        ctx = torch.autocast("cuda")
    else:
        ctx = nullcontext()

    for lr_path, hr_path in pairs:
        print(f"\nProcessando: {lr_path.name}")
        lr_image = Image.open(lr_path).convert("RGB")
        hr_image = Image.open(hr_path).convert("RGB")

        print(f"  Tamanho LR: {lr_image.size}, HR: {hr_image.size}")

        with ctx:
            result = pipe(
                prompt=prompt,          # pode ser "" mesmo
                image=lr_image,
                num_inference_steps=40, # ajusta se quiser mais rápido/lento
                guidance_scale=0.0,     # 0.0 pra não forçar o texto
            )

        sr_image = result.images[0]

        # O modelo é x4; pra scale=2 ou 3, ajustamos para o tamanho da HR
        if match_hr_size:
            sr_image = sr_image.resize(hr_image.size, Image.BICUBIC)

        # Nome de saída: usa o nome da LR trocando o sufixo
        # ex: img_001_SRF_2_LR.png -> img_001_SRF_2_SRx2.png
        stem_base = lr_path.name.replace("_LR.png", "")
        out_name = f"{stem_base}_SRx{scale}.png"
        out_path = out_dir / out_name

        sr_image.save(out_path)
        print(f"  -> SR salva em: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale",
        type=int,
        choices=[2, 3, 4],
        required=True,
        help="Fator de escala do Set14: 2, 3 ou 4",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="Set14dataset/Set14",
        help="Caminho para a pasta Set14 (onde estão image_SRF_2/3/4)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Prompt de texto opcional para o upscaler",
    )
    parser.add_argument(
        "--no_match_hr",
        action="store_true",
        help="Se passado, NÃO ajusta a SR para o tamanho da HR",
    )

    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando device: {device}")
    print(f"Root Set14: {root_dir}")

    upscale_set14_for_scale(
        root_dir=root_dir,
        scale=args.scale,
        device=device,
        prompt=args.prompt,
        match_hr_size=not args.no_match_hr,
    )


if __name__ == "__main__":
    main()