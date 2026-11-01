"""Gradio demo for the project's trained image dehazing checkpoints."""

from __future__ import annotations

import argparse
import base64
import io
import os
import socket
import sys
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import gradio as gr
    import qrcode
except ImportError as exc:
    raise SystemExit(
        "Demo dependencies are missing. Install them with:\n"
        '  python -m pip install -e ".[demo]"\n'
        f"Missing package: {exc.name}"
    ) from exc

from dehazing.evaluate import tiled_predict
from dehazing.models import build_model


MODEL_RUNS: dict[str, tuple[str, str]] = {
    "DehazeFormer-T · RESIDE baseline": (
        "dehazeformer_t",
        "Baseline trained on RESIDE-6K-Outdoor.",
    ),
    "DehazeFormer-T · three-source baseline": (
        "dehazeformer_t_reside_nhhaze2_hdnhhaze_bs8_tiles512_w0_clampfix",
        "Baseline trained on RESIDE-6K-Outdoor, NH-HAZE2, and HD-NH-HAZE.",
    ),
    "SCAF (M1) · three-source training": (
        "dehazeformer_t_m1_full50",
        "DehazeFormer-T with spatial-channel adaptive fusion.",
    ),
    "SCAF + haze map (M2) · three-source training": (
        "dehazeformer_t_m2_full50",
        "M1 with a learned haze-map guidance branch.",
    ),
    "SCAF + haze map + Haar (M3) · three-source training": (
        "dehazeformer_t_m3_full50",
        "M2 with a Haar high-frequency detail branch.",
    ),
    "Physical loss + MCBM (M6) · three-source training": (
        "dehazeformer_t_m6_mcbm_full50",
        "DehazeFormer-T with atmospheric consistency loss and MCBM haze augmentation.",
    ),
    "SLRNet · three-source training": (
        "slrnet_recent_full50_bs16",
        "Lightweight SLRNet trained on the three-source training split.",
    ),
    "WLD-Net · three-source training": (
        "wldnet_recent_full50",
        "Wavelet-based WLD-Net trained on the three-source training split.",
    ),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"
TILE_SIZE = 512
TILE_OVERLAP = 64
MODEL_LOCK = threading.Lock()
LOADED_LABEL: str | None = None
LOADED_MODEL: torch.nn.Module | None = None
QR_TARGET_URL = os.environ.get("DEHAZE_DEMO_URL", "").strip()
QR_URL_READY = threading.Event()
if QR_TARGET_URL:
    QR_URL_READY.set()


def available_models() -> dict[str, tuple[Path, str]]:
    """Return the configured models whose best checkpoints exist."""
    found: dict[str, tuple[Path, str]] = {}
    for label, (run_name, description) in MODEL_RUNS.items():
        checkpoint = PROJECT_ROOT / "runs" / run_name / "checkpoints" / "best.pth"
        if checkpoint.is_file():
            found[label] = (checkpoint, description)
    return found


def qr_image(url: str) -> Image.Image | None:
    """Render a scannable QR code for the exact demo URL."""
    if not url:
        return None
    code = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=12,
        border=4,
    )
    code.add_data(url)
    code.make(fit=True)
    return code.make_image(fill_color="#102a43", back_color="white").convert("RGB")


def qr_html(url: str) -> str:
    """Make a clickable QR preview that expands into a large overlay."""
    image = qr_image(url)
    if image is None:
        return "<div class='qr-empty'>QR sẽ xuất hiện khi URL demo sẵn sàng.</div>"
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    source = f"data:image/png;base64,{encoded}"
    return (
        "<details class='qr-zoom'>"
        "<summary aria-label='Nhấn để phóng to mã QR'>"
        f"<img class='qr-preview' src='{source}' alt='Mã QR mở demo'>"
        "<span class='qr-caption'>Nhấn vào QR để phóng to</span>"
        "<span class='qr-close'>Đóng ✕</span>"
        "</summary>"
        "<div class='qr-overlay'><img class='qr-large' "
        f"src='{source}' alt='Mã QR mở demo kích thước lớn'></div>"
        "</details>"
    )


def network_url(port: int) -> str:
    """Find the machine's LAN address so another device on the same Wi-Fi can scan."""
    address = "127.0.0.1"
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        address = probe.getsockname()[0]
    except OSError:
        pass
    finally:
        probe.close()
    return "" if address.startswith("127.") else f"http://{address}:{port}"


def load_model(label: str) -> torch.nn.Module:
    """Load one checkpoint at a time to keep device memory bounded."""
    global LOADED_LABEL, LOADED_MODEL
    models = available_models()
    if label not in models:
        raise ValueError("Checkpoint is not available for the selected model.")
    with MODEL_LOCK:
        if LOADED_LABEL == label and LOADED_MODEL is not None:
            return LOADED_MODEL

        checkpoint_path, _ = models[label]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        config = checkpoint.get("config", {})
        model_config = config.get("model", {})
        model_name = str(model_config.get("name", "dehazeformer_t"))
        if bool(model_config.get("use_depth", False)):
            raise ValueError(
                "This demo accepts RGB uploads only; depth-conditioned checkpoints "
                "need an additional depth-map input."
            )
        model = build_model(
            model_name,
            use_depth=False,
            base_dim=int(model_config.get("base_dim", 32)),
        )
        model.load_state_dict(state_dict, strict=True)
        if hasattr(model, "reparameterize"):
            model.reparameterize()
        model.to(DEVICE).eval()

        if LOADED_MODEL is not None:
            del LOADED_MODEL
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        LOADED_MODEL = model
        LOADED_LABEL = label
        return model


@torch.inference_mode()
def dehaze(image: Image.Image | None, label: str) -> tuple[Image.Image | None, str]:
    if image is None:
        return None, "Upload ảnh haze trước khi chạy."
    if label not in available_models():
        return None, "Checkpoint của model này chưa có trong thư mục runs/."

    try:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)
        model = load_model(label)
        result = tiled_predict(
            model,
            tensor,
            DEVICE,
            AMP,
            tile_size=TILE_SIZE,
            overlap=TILE_OVERLAP,
        )
        output = result[0].permute(1, 2, 0).numpy()
        output_image = Image.fromarray(np.round(np.clip(output, 0, 1) * 255).astype(np.uint8))
        description = available_models()[label][1]
        status = (
            f"Model: {label}\n"
            f"{description}\n"
            f"Input: {rgb.width} × {rgb.height} px · "
            f"Device: {'NVIDIA GPU' if DEVICE.type == 'cuda' else 'CPU'}"
        )
        return output_image, status
    except Exception as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            return None, (
                "Ảnh vượt quá bộ nhớ GPU. Hãy thử ảnh nhỏ hơn hoặc giảm TILE_SIZE "
                f"trong demo.py. Chi tiết: {exc}"
            )
        return None, f"Không thể xử lý ảnh: {type(exc).__name__}: {exc}"


def refresh_qr() -> tuple[str, str]:
    # Only encode an explicitly configured URL or Gradio's public share URL.
    # A Colab/LAN address is not reachable by viewers outside that network.
    # The browser's load event can arrive before launch() returns its share URL.
    QR_URL_READY.wait(timeout=90)
    url = QR_TARGET_URL
    note = (
        f"Quét mã để mở demo: {url}"
        if url
        else "Bật --share để tạo link public, hoặc đặt DEHAZE_DEMO_URL thành URL public trước khi chạy."
    )
    return qr_html(url), note


def build_app() -> gr.Blocks:
    models = available_models()
    choices = list(models)
    preferred_model = "Physical loss + MCBM (M6) · three-source training"
    initial_model = preferred_model if preferred_model in models else (choices[0] if choices else None)
    with gr.Blocks(title="Demo khử sương ảnh") as app:
        gr.Markdown(
            """
            # Demo khử sương ảnh
            Tải ảnh haze lên, chọn checkpoint đã huấn luyện và so sánh kết quả.
            """
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=7):
                model_choice = gr.Dropdown(
                    choices=choices,
                    value=initial_model,
                    label="Chọn model",
                    interactive=bool(choices),
                )
                model_description = gr.Markdown(
                    models[initial_model][1] if initial_model else
                    "Chưa tìm thấy checkpoint hỗ trợ. Hãy kiểm tra thư mục runs."
                )
                with gr.Row():
                    input_image = gr.Image(
                        label="Ảnh haze đầu vào",
                        type="pil",
                        image_mode="RGB",
                        height=390,
                    )
                    output_image = gr.Image(
                        label="Ảnh sau khử sương",
                        type="pil",
                        image_mode="RGB",
                        height=390,
                    )
                run_button = gr.Button(
                    "Khử sương",
                    variant="primary",
                    interactive=bool(choices),
                )
                status = gr.Markdown("Tải ảnh lên rồi nhấn **Khử sương**.")
            with gr.Column(scale=3, min_width=300):
                gr.Markdown("## Mở demo bằng QR")
                qr = gr.HTML(
                    value=qr_html(QR_TARGET_URL),
                    elem_classes=["qr-container"],
                )
                qr_note = gr.Markdown()
                gr.Markdown(
                    "QR dẫn đến URL của phiên demo đang chạy. Điện thoại cần truy cập "
                    "được URL đó (cùng mạng Wi-Fi hoặc link public)."
                )

        model_choice.change(
            fn=lambda name: models.get(name, (None, "Chưa có checkpoint."))[1],
            inputs=model_choice,
            outputs=model_description,
        )
        run_button.click(
            fn=dehaze,
            inputs=[input_image, model_choice],
            outputs=[output_image, status],
        )
        app.load(fn=refresh_qr, inputs=None, outputs=[qr, qr_note])
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Gradio dehazing demo.")
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio URL.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: all interfaces).")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--url",
        default="",
        help="Public demo URL to encode in the QR (overrides --share URL).",
    )
    args = parser.parse_args()

    global QR_TARGET_URL
    QR_TARGET_URL = args.url.strip() or QR_TARGET_URL
    if QR_TARGET_URL:
        QR_URL_READY.set()
    else:
        QR_URL_READY.clear()
    app = build_app().queue(default_concurrency_limit=1)
    _, local_url, share_url = app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        prevent_thread_lock=True,
        show_error=True,
        theme=gr.themes.Soft(),
        max_file_size="20mb",
        css="""
        .qr-container { display: flex; justify-content: center; }
        .qr-zoom { text-align: center; }
        .qr-zoom > summary { list-style: none; cursor: zoom-in; }
        .qr-zoom > summary::-webkit-details-marker { display: none; }
        .qr-preview { display: block; width: min(100%, 330px); margin: 0 auto; }
        .qr-caption { display: block; color: #52606d; font-size: 0.9rem; margin-top: 0.35rem; }
        .qr-close { display: none; }
        .qr-zoom[open] {
          position: fixed; inset: 0; z-index: 99999; background: rgba(10, 20, 35, 0.88);
        }
        .qr-zoom[open] > summary {
          position: fixed; top: 18px; right: 24px; z-index: 100000;
          cursor: pointer; color: white; background: #243b53; border-radius: 8px;
          padding: 0.7rem 1rem; font-size: 1rem;
        }
        .qr-zoom[open] .qr-preview, .qr-zoom[open] .qr-caption { display: none; }
        .qr-zoom[open] .qr-close { display: inline; }
        .qr-overlay {
          position: fixed; inset: 0; display: grid; place-items: center;
          pointer-events: none; padding: 5vh 5vw;
        }
        .qr-large { width: min(78vw, 680px); max-height: 86vh; object-fit: contain; }
        .qr-empty { padding: 1.5rem; color: #52606d; text-align: center; }
        """,
    )
    if args.url.strip():
        QR_TARGET_URL = args.url.strip()
    elif share_url:
        QR_TARGET_URL = share_url
    else:
        QR_TARGET_URL = ""
    QR_URL_READY.set()
    print(f"QR demo URL: {QR_TARGET_URL or '(chưa có URL public; chạy với --share)'}")
    app.block_thread()


if __name__ == "__main__":
    main()
