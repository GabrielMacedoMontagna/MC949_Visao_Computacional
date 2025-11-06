#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grava um vídeo com overlay de detecções + narrações TTS, amostrando por timestamp
(Option A). O mesmo vídeo de entrada é amostrado, por exemplo, a 10 FPS para ambos
os modelos (YOLO ou RT-DETR), garantindo a mesma cobertura temporal e duração.

Pré-requisitos (instale uma vez):
  pip install ultralytics transformers torch pillow opencv-python gTTS numpy

Também é necessário ter o ffmpeg instalado no sistema (no PATH).
  - Ubuntu/Debian: sudo apt-get install ffmpeg
  - macOS (Homebrew): brew install ffmpeg
  - Windows (Chocolatey): choco install ffmpeg
"""

import os
import cv2
import time
import shutil
import argparse
import subprocess
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# TTS
from gtts import gTTS

# Modelos
import torch
from ultralytics import YOLO
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor


# =========================
# Utilidades e desenho
# =========================
def thirds_position(x_center: float, width: int) -> str:
    left = width / 3.0
    right = 2.0 * width / 3.0
    if x_center < left:
        return "à esquerda"
    elif x_center <= right:
        return "ao centro"
    else:
        return "à direita"

def draw_boxes(img_bgr: np.ndarray, boxes: List[Dict[str, Any]],
               class_names: Dict[int, str], font_scale=0.9, text_th=2, box_th=2) -> np.ndarray:
    """boxes: [{xyxy:[x1,y1,x2,y2], cls:int, conf:float}, ...]"""
    out = img_bgr.copy()
    H, W = out.shape[:2]
    scale = max(0.6, (W / 640.0))
    fs = font_scale * scale
    tt = max(1, int(round(text_th * scale)))
    bt = max(2, int(round(box_th * scale)))
    pad = max(3, int(round(3 * scale)))

    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        cls_id = b["cls"]
        conf = b["conf"]
        label = class_names.get(cls_id, str(cls_id))
        txt = f"{label} {conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), bt)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, tt)
        y_text = max(y1, th + 2 * pad)
        cv2.rectangle(out, (x1, y_text - th - 2 * pad), (x1 + tw + 2 * pad, y_text), (0, 255, 0), cv2.FILLED)
        cv2.putText(out, txt, (x1 + pad, y_text - pad), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), tt, cv2.LINE_AA)
    return out

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        raise RuntimeError("ffmpeg não encontrado no PATH. Instale e/ou adicione ao PATH do sistema.") from e


# =========================
# Amostragem por timestamp (arquivo de vídeo)
# =========================
def iterate_by_fps(video_path: str, target_fps: float):
    """
    Gera (frame_bgr, timestamp_seg) amostrando por tempo.
    Usa seeks por timestamp, garantindo frames nos tempos: 0, 1/fps, 2/fps, ...
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = nframes / max(1e-6, src_fps)

    t = 0.0
    step = 1.0 / target_fps
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)  # busca por timestamp
        ok, frame = cap.read()
        if not ok:
            break
        yield frame, t
        t += step
    cap.release()


# =========================
# Detectores
# =========================
class RTDETR:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu",
                 ckpt="PekingU/rtdetr_r18vd"):
        self.device = device
        self.processor = RTDetrImageProcessor.from_pretrained(ckpt)
        self.model = RTDetrForObjectDetection.from_pretrained(ckpt).to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    @torch.no_grad()
    def infer(self, img_bgr: np.ndarray, score_threshold=0.25, max_det=300,
              allowed: set | None = None):
        rgb = img_bgr[:, :, ::-1]
        pil = Image.fromarray(rgb)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        H, W = pil.size[1], pil.size[0]
        post = self.processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([[H, W]], device=self.device),
            threshold=score_threshold
        )[0]

        keep = []
        for i, (lbl, sc) in enumerate(zip(post["labels"], post["scores"])):
            name = self.id2label[int(lbl)]
            if (allowed is None) or (name in allowed):
                keep.append(i)
        keep = sorted(keep, key=lambda i: float(post["scores"][i].item()), reverse=True)[:max_det]

        boxes = []
        for i in keep:
            x1, y1, x2, y2 = post["boxes"][i].tolist()
            x1 = int(max(0, min(x1, W - 1))); x2 = int(max(0, min(x2, W - 1)))
            y1 = int(max(0, min(y1, H - 1))); y2 = int(max(0, min(y2, H - 1)))
            if x2 > x1 and y2 > y1:
                boxes.append({
                    "xyxy": [x1, y1, x2, y2],
                    "cls": int(post["labels"][i].item()),
                    "conf": float(post["scores"][i].item())
                })
        return boxes, self.id2label


class YOLOX:
    def __init__(self, weights="yolov8n.pt", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = YOLO(weights)
        self.names = self.model.names

    @torch.no_grad()
    def infer(self, img_bgr: np.ndarray, score_threshold=0.25, allowed: set | None = None):
        rgb = img_bgr[:, :, ::-1]
        r = self.model.predict(rgb, conf=score_threshold, verbose=False)[0]
        boxes = []
        for b in r.boxes:
            cls_id = int(b.cls[0].item())
            name = r.names[cls_id]
            if (allowed is not None) and (name not in allowed):
                continue
            x1, y1, x2, y2 = b.xyxy[0].int().cpu().numpy().tolist()
            conf = float(b.conf[0].item()) if hasattr(b, "conf") else 1.0
            boxes.append({"xyxy": [x1, y1, x2, y2], "cls": cls_id, "conf": conf})
        return boxes, r.names


# =========================
# TTS e mux (ffmpeg)
# =========================
class TTSTrack:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.events: List[Dict[str, Any]] = []
        self.counter = 0

    def add(self, text: str, start_s: float, lang="pt"):
        mp3_path = self.base_dir / f"tts_{self.counter:04d}.mp3"
        tts = gTTS(text, lang=lang)
        tts.save(str(mp3_path))
        self.events.append({"path": str(mp3_path), "start": float(start_s)})
        self.counter += 1

def mux_with_ffmpeg(video_in: Path, video_out: Path, events: List[Dict[str, Any]], fps: float):
    """
    Junta o MP4 sem áudio (video_in) com os MP3 do TTS (events: [{'path', 'start'}])
    usando ffmpeg. Compatível com FFmpeg 5.1 (amix=inputs=...).
    """
    check_ffmpeg()
    cmd = ['ffmpeg', '-y', '-i', str(video_in)]
    for ev in events:
        cmd += ['-i', ev['path']]

    if events:
        parts = []
        for i, ev in enumerate(events, start=1):
            delay_ms = int(ev['start'] * 1000)
            parts.append(f'[{i}:a]adelay={delay_ms}[a{i}]')  # gTTS = mono
        mixed_inputs = ''.join(f'[a{i}]' for i in range(1, len(events) + 1))
        parts.append(f'{mixed_inputs}amix=inputs={len(events)}:normalize=0[aout]')
        filter_complex = ';'.join(parts)

        cmd += [
            '-filter_complex', filter_complex,
            '-map', '0:v', '-map', '[aout]',
            '-c:v', 'libx264', '-c:a', 'aac',
            '-r', str(fps), str(video_out)
        ]
    else:
        cmd += ['-c:v', 'libx264', '-an', '-r', str(fps), str(video_out)]

    subprocess.run(cmd, check=True)


# =========================
# Narração (heurística simples)
# =========================
ALLOWED_CLASSES_DEFAULT = {
    "person", "car", "bicycle", "motorbike", "bus", "truck", "chair", "bench",
    "dog", "cat", "cup", "tv", "tvmonitor", "traffic light", "backpack",
    "handbag", "suitcase", "laptop"
}

def pick_sentence_from_boxes(boxes: List[Dict[str, Any]], names: Dict[int, str],
                             width: int, min_conf=0.5) -> str | None:
    """
    Exemplo: escolhe o maior objeto acima do limiar e narra posição.
    """
    cand = []
    for b in boxes:
        if b["conf"] < min_conf:
            continue
        x1, y1, x2, y2 = b["xyxy"]
        area = (x2 - x1) * (y2 - y1)
        cx = (x1 + x2) / 2.0
        pos = thirds_position(cx, width)
        label = names.get(b["cls"], str(b["cls"]))
        cand.append((area, f"Atenção: {label} {pos}."))
    if not cand:
        return None
    cand.sort(key=lambda t: t[0], reverse=True)
    return cand[0][1]


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description="Gravar vídeo com overlay e TTS amostrando por timestamp (mesmos frames para ambos os modelos).")
    ap.add_argument("--model", choices=["yolo", "rtdetr"], required=True, help="Detector a usar (yolo ou rtdetr).")
    ap.add_argument("--source", required=True, help="Caminho do arquivo de vídeo (obrigatório nesta opção).")
    ap.add_argument("--out", default=None, help="Arquivo final de saída (.mp4). Se omitido, usa exports/demo_<tag>.mp4")
    ap.add_argument("--sample-fps", type=float, default=10.0, help="FPS de amostragem temporal (mesmo usado na exportação).")
    ap.add_argument("--conf", type=float, default=0.5, help="Limiar de confiança para FALAR.")
    ap.add_argument("--draw-conf", type=float, default=0.25, help="Limiar de confiança para DESENHAR.")
    ap.add_argument("--weights", default="yolov8n.pt", help="Weights do YOLO (se --model yolo).")
    ap.add_argument("--duration", type=float, default=None, help="(Opcional) limitar duração em segundos do vídeo final.")
    ap.add_argument("--show", action="store_true", help="Exibir janela com a gravação (overlay).")
    ap.add_argument("--speak-classes", nargs="*", default=None, help="Classes para narração; default pré-definido.")
    args = ap.parse_args()

    # Verifica se é arquivo (esta opção requer arquivo, não webcam)
    if args.source.isdigit():
        raise ValueError("Opção A requer arquivo de vídeo. Use --source com caminho para o arquivo (não índice de webcam).")

    # Saídas
    out_dir = Path("exports"); out_dir.mkdir(parents=True, exist_ok=True)
    tag = "yolo" if args.model == "yolo" else "rtdetr"
    video_raw = out_dir / f"demo_{tag}_noaudio.mp4"
    video_final = Path(args.out) if args.out else out_dir / f"demo_{tag}.mp4"

    # TTS
    tts_dir = out_dir / f"tts_{tag}"
    shutil.rmtree(tts_dir, ignore_errors=True)
    tts_dir.mkdir(parents=True, exist_ok=True)
    tts_track = TTSTrack(tts_dir)

    # Detector
    device = "cuda" if torch.cuda.is_available() else "cpu"
    allowed = set(args.speak_classes) if args.speak_classes else ALLOWED_CLASSES_DEFAULT
    if args.model == "yolo":
        det = YOLOX(weights=args.weights, device=device)
        infer = lambda img: det.infer(img, score_threshold=args.draw_conf, allowed=None)
    else:
        det = RTDETR(device=device)
        infer = lambda img: det.infer(img, score_threshold=args.draw_conf, allowed=None)

    # Writer será criado no primeiro frame (quando soubermos W,H)
    writer = None
    frames_written = 0

    # Narração: cooldown no tempo do VÍDEO (timestamp dos frames)
    last_tts_t = -1e9
    TTS_COOLDOWN = 3.0  # segundos

    # Loop por timestamp fixo
    for frame, t in iterate_by_fps(args.source, args.sample_fps):
        H, W = frame.shape[:2]

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_raw), fourcc, args.sample_fps, (W, H))

        # Inferência e overlay
        boxes, names = infer(frame)
        frame_ov = draw_boxes(frame, boxes, names, font_scale=0.9)

        # HUD simples (timestamp e fps alvo)
        cv2.putText(frame_ov, f"t={t:5.2f}s | target={args.sample_fps:.1f} FPS",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2, cv2.LINE_AA)

        # Narração (baseada no timestamp do frame)
        if (t - last_tts_t) >= TTS_COOLDOWN:
            boxes_speak = [b for b in boxes if names.get(b["cls"], "") in allowed]
            sent = pick_sentence_from_boxes(boxes_speak, names, width=W, min_conf=args.conf)
            if sent:
                tts_track.add(sent, start_s=t)
                last_tts_t = t

        writer.write(frame_ov)
        frames_written += 1

        if args.show:
            cv2.imshow("Recording (overlay - amostragem temporal)", frame_ov)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Limita duração final, se solicitado
        if args.duration is not None and t >= float(args.duration):
            break

    if writer is not None:
        writer.release()
    if args.show:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    print("Vídeo sem áudio salvo em:", video_raw)
    print("Mux de áudio TTS com ffmpeg...")
    mux_with_ffmpeg(video_raw, video_final, tts_track.events, fps=args.sample_fps)
    print("Vídeo final salvo em:", video_final)

    # (Opcional) limpar MP3 temporários
    try:
        shutil.rmtree(tts_dir, ignore_errors=True)
    except Exception:
        pass

    print("Pronto ✅  (mesmos frames no tempo para ambos os modelos)")

if __name__ == "__main__":
    main()
