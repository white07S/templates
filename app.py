#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pdf_to_markdown_smol_docling.py

Usage:
  python pdf_to_markdown_smol_docling.py \
      --pdf /path/to/file.pdf \
      --smol /models/ds4sd/SmolDocling-256M-preview \
      --out ./out_dir \
      --device cuda \
      [--caption-model /models/ibm-granite/granite-vision-3.2-2b]

- Forces local model usage (offline).
- Runs SmolDocling VLM page-by-page and exports Markdown.
- Optionally adds picture descriptions (captions) using a local VLM for images.
"""

import os
from pathlib import Path
import argparse
from typing import Optional

# --- force offline/local-only ---
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption, ConversionResult

# VLM (SmolDocling) pipeline
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import (
    InlineVlmOptions,
    InferenceFramework,
    TransformersModelType,
    ResponseFormat,
)

# Accelerator options (CUDA/MPS/CPU)
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

# Export helpers (save images & markdown)
from docling_core.types.doc import ImageRefMode

# (Optional) captioning enrichment in a second pass (local VLM)
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionVlmOptions,
)


def run_smol_vlm_local(
    pdf_path: Path,
    smol_model_path: Path,
    out_dir: Path,
    device: str = "cuda",
    images_scale: float = 2.0,
) -> ConversionResult:
    """
    Convert PDF with local SmolDocling (Transformers), page-by-page.
    Generates picture images so they can be referenced in Markdown.
    """
    if device.lower() == "cuda":
        accel = AcceleratorOptions(device=AcceleratorDevice.CUDA)
    elif device.lower() == "mps":
        accel = AcceleratorOptions(device=AcceleratorDevice.MPS)
    else:
        accel = AcceleratorOptions(device=AcceleratorDevice.CPU)

    # Tell Docling to use a local HF repository folder for the VLM
    vlm_opts = InlineVlmOptions(
        repo_id=str(smol_model_path),                    # local folder
        response_format=ResponseFormat.DOCTAGS,         # SmolDocling's native format
        inference_framework=InferenceFramework.TRANSFORMERS,
        transformers_model_type=TransformersModelType.AUTOMODEL_VISION2SEQ,
        supported_devices=[AcceleratorDevice.CUDA, AcceleratorDevice.CPU, AcceleratorDevice.MPS],
        scale=images_scale,
        temperature=0.0,
    )

    pipe_opts = VlmPipelineOptions(
        vlm_options=vlm_opts,
        accelerator_options=accel,
        images_scale=images_scale,
        generate_picture_images=True,   # needed so figures are exported/linked
        generate_page_images=False,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipe_opts,
            )
        }
    )

    conv_res = converter.convert(pdf_path)

    # Save per-page Markdown (iterating pages) + a full-document Markdown
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = conv_res.input.file.stem

    # Full-document markdown with externally referenced images
    md_all = out_dir / f"{stem}.md"
    conv_res.document.save_as_markdown(md_all, image_mode=ImageRefMode.REFERENCED)

    # Also dump page-wise markdown (simple split by page)
    # We stream items by page and write a minimal page header.
    for page_no, page in conv_res.document.pages.items():
        md_page = out_dir / f"{stem}-page-{page_no}.md"
        with md_page.open("w", encoding="utf-8") as fp:
            fp.write(f"# Page {page_no}\n\n")
            # Walk items that belong to this page and print as markdown fragments
            for item, _lvl in conv_res.document.iterate_items():
                # every item has provenance; check page number match
                try:
                    prov = item.prov[0]
                    if getattr(prov, "page_no", None) != page_no:
                        continue
                except Exception:
                    continue
                # Append each item's own markdown (Docling will render headings, text, tables, pictures)
                fp.write(item.to_markdown(doc=conv_res.document))
                fp.write("\n\n")

    return conv_res


def add_picture_descriptions_with_local_vlm(
    pdf_path: Path,
    out_dir: Path,
    device: str,
    caption_model_path: Path,
    images_scale: float = 2.0,
) -> ConversionResult:
    """
    Optional: run a light second pass to add picture descriptions (captions) using a LOCAL VLM
    like Granite Vision or SmolVLM, then export Markdown with captions included.

    This uses Docling's picture-description enrichment config. (Runs offline; repo_id points to a local folder.)
    """
    if device.lower() == "cuda":
        accel = AcceleratorOptions(device=AcceleratorDevice.CUDA)
    elif device.lower() == "mps":
        accel = AcceleratorOptions(device=AcceleratorDevice.MPS)
    else:
        accel = AcceleratorOptions(device=AcceleratorDevice.CPU)

    pic_vlm = PictureDescriptionVlmOptions(
        repo_id=str(caption_model_path),  # local folder for Granite Vision / SmolVLM
        scale=images_scale,
        prompt="Describe the image in 1–3 concise, factual sentences.",
    )

    pdf_opts = PdfPipelineOptions()
    pdf_opts.accelerator_options = accel
    pdf_opts.images_scale = images_scale
    pdf_opts.generate_picture_images = True
    pdf_opts.do_picture_description = True
    pdf_opts.picture_description_options = pic_vlm

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
    )
    conv_res = converter.convert(pdf_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = conv_res.input.file.stem

    # Markdown with externally referenced pictures (captions will be embedded as text)
    md_with_caps = out_dir / f"{stem}.captions.md"
    conv_res.document.save_as_markdown(md_with_caps, image_mode=ImageRefMode.REFERENCED)

    return conv_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=Path, help="Path to input PDF")
    ap.add_argument("--smol", required=True, type=Path, help="Local folder of ds4sd/SmolDocling-256M-preview")
    ap.add_argument("--out", required=True, type=Path, help="Output directory")
    ap.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"], help="Accelerator")
    ap.add_argument("--caption-model", type=Path, default=None,
                    help="(Optional) local folder of a caption VLM (e.g., ibm-granite/granite-vision-3.2-2b or HuggingFaceTB/SmolVLM-256M-Instruct)")
    ap.add_argument("--scale", type=float, default=2.0, help="Raster scale used for page/picture images")
    args = ap.parse_args()

    # Pass 1: SmolDocling VLM (full extraction)
    smol_res = run_smol_vlm_local(
        pdf_path=args.pdf,
        smol_model_path=args.smol,
        out_dir=args.out,
        device=args.device,
        images_scale=args.scale,
    )

    # Pass 2 (optional): caption pictures with a local small VLM
    if args.caption_model is not None:
        add_picture_descriptions_with_local_vlm(
            pdf_path=args.pdf,
            out_dir=args.out,
            device=args.device,
            caption_model_path=args.caption_model,
            images_scale=args.scale,
        )

    print(f"[OK] Wrote Markdown(s) in: {args.out.resolve()}")


if __name__ == "__main__":
    main()
