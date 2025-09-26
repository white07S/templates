#!/usr/bin/env python3
"""
PDF to Markdown converter using SmolDocling-256M-preview model
with GPU acceleration and vLLM for fast inference.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Optional
import argparse

# PDF processing with images
import fitz  # PyMuPDF
from PIL import Image
import io

# vLLM for fast GPU inference
from vllm import LLM, SamplingParams

# Docling for document processing
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SmolDoclingPDFConverter:
    """Convert PDFs to Markdown using SmolDocling with GPU acceleration."""
    
    def __init__(
        self,
        model_path: str = None,
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.9,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        image_scale: float = 1.0,
        dpi: int = 300
    ):
        """
        Initialize the converter with model and processing parameters.
        
        Args:
            model_path: Path to local SmolDocling model (if None, uses HF hub)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory fraction to use (0-1)
            max_tokens: Maximum tokens for generation
            temperature: Sampling temperature (0 for deterministic)
            image_scale: Scale factor for image processing
            dpi: DPI for PDF rendering
        """
        self.model_path = model_path or "ds4sd/SmolDocling-256M-preview"
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.image_scale = image_scale
        self.dpi = dpi
        
        # Initialize vLLM model
        self._init_model()
        
        # Sampling parameters for generation
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        
        # Chat template for SmolDocling
        self.chat_template = "<|im_start|>User:<image>Convert page to Docling.<end_of_utterance>\nAssistant:"
        
    def _init_model(self):
        """Initialize vLLM model with tensor parallelism for multi-GPU."""
        logger.info(f"Initializing SmolDocling model from: {self.model_path}")
        logger.info(f"Using {self.tensor_parallel_size} GPUs with tensor parallelism")
        
        try:
            self.llm = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=True,
                limit_mm_per_prompt={"image": 1},
                dtype="bfloat16",
                max_model_len=16384,
                enforce_eager=False  # Use CUDA graphs for optimization
            )
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """
        Convert PDF pages to PIL images with optional image extraction.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of PIL images (one per page)
        """
        images = []
        
        try:
            pdf_doc = fitz.open(pdf_path)
            logger.info(f"Processing {len(pdf_doc)} pages from {pdf_path}")
            
            for page_num, page in enumerate(pdf_doc, 1):
                # Render page to image
                mat = fitz.Matrix(self.dpi / 72.0, self.dpi / 72.0)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Optionally scale image
                if self.image_scale != 1.0:
                    new_size = (
                        int(img.width * self.image_scale),
                        int(img.height * self.image_scale)
                    )
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                images.append(img)
                logger.info(f"Processed page {page_num}/{len(pdf_doc)}")
                
            pdf_doc.close()
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
            
        return images
    
    def process_images_batch(self, images: List[Image.Image]) -> List[str]:
        """
        Process multiple images in batch using vLLM for efficiency.
        
        Args:
            images: List of PIL images
            
        Returns:
            List of DocTags strings
        """
        prompts = []
        for img in images:
            prompts.append({
                "prompt": self.chat_template,
                "multi_modal_data": {"image": img}
            })
        
        logger.info(f"Processing batch of {len(prompts)} images")
        
        # Generate outputs for all images
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)
            
        return results
    
    def doctags_to_markdown(self, doctags_list: List[str]) -> str:
        """
        Convert DocTags output to clean Markdown format.
        
        Args:
            doctags_list: List of DocTags strings (one per page)
            
        Returns:
            Markdown string
        """
        markdown_pages = []
        
        for page_num, doctags in enumerate(doctags_list, 1):
            try:
                # Parse DocTags and convert to DoclingDocument
                doc = DocTagsDocument.from_tags(doctags)
                docling_doc = DoclingDocument.from_doctags(doc)
                
                # Export to markdown
                page_md = docling_doc.export_to_markdown()
                
                # Add page separator
                if len(doctags_list) > 1:
                    page_md = f"\n---\n<!-- Page {page_num} -->\n{page_md}"
                
                markdown_pages.append(page_md)
                
            except Exception as e:
                logger.warning(f"Failed to parse DocTags for page {page_num}: {e}")
                # Fallback: clean the raw output
                cleaned = self._clean_raw_output(doctags)
                markdown_pages.append(cleaned)
        
        return "\n\n".join(markdown_pages)
    
    def _clean_raw_output(self, text: str) -> str:
        """
        Clean raw output if DocTags parsing fails.
        
        Args:
            text: Raw text output
            
        Returns:
            Cleaned markdown text
        """
        # Remove common artifacts
        text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
        text = text.replace("<end_of_utterance>", "").replace("<|endoftext|>", "")
        
        # Remove DocTags if present but unparseable
        import re
        text = re.sub(r'<[^>]+>', '', text)
        
        return text.strip()
    
    def convert_pdf(
        self,
        pdf_path: str,
        output_path: Optional[str] = None,
        batch_size: int = 4
    ) -> str:
        """
        Convert PDF to Markdown with GPU acceleration.
        
        Args:
            pdf_path: Path to input PDF
            output_path: Path for output markdown (if None, saves in same dir)
            batch_size: Number of pages to process in parallel
            
        Returns:
            Path to saved markdown file
        """
        start_time = time.time()
        
        # Validate input
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Set output path
        if output_path is None:
            output_path = pdf_path.with_suffix('.md')
        else:
            output_path = Path(output_path)
        
        logger.info(f"Converting: {pdf_path} -> {output_path}")
        
        # Convert PDF to images
        images = self.pdf_to_images(str(pdf_path))
        logger.info(f"Extracted {len(images)} pages")
        
        # Process images in batches for efficiency
        all_doctags = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = self.process_images_batch(batch)
            all_doctags.extend(batch_results)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(images) + batch_size - 1)//batch_size}")
        
        # Convert to markdown
        markdown = self.doctags_to_markdown(all_doctags)
        
        # Add metadata header
        metadata = f"""---
source: {pdf_path.name}
pages: {len(images)}
converted: {time.strftime('%Y-%m-%d %H:%M:%S')}
model: SmolDocling-256M-preview
processing_time: {time.time() - start_time:.2f}s
---

"""
        markdown = metadata + markdown
        
        # Save to file
        output_path.write_text(markdown, encoding='utf-8')
        
        elapsed = time.time() - start_time
        pages_per_second = len(images) / elapsed
        logger.info(f"Conversion complete in {elapsed:.2f}s ({pages_per_second:.2f} pages/sec)")
        logger.info(f"Saved to: {output_path}")
        
        return str(output_path)


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown using SmolDocling with GPU acceleration"
    )
    parser.add_argument(
        "pdf_path",
        help="Path to input PDF file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output markdown file path (default: same dir as PDF)"
    )
    parser.add_argument(
        "-m", "--model-path",
        default="./SmolDocling-256M-preview",  # Local path
        help="Path to local SmolDocling model directory"
    )
    parser.add_argument(
        "-g", "--gpus",
        type=int,
        default=2,
        help="Number of GPUs to use for tensor parallelism (default: 2)"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=4,
        help="Batch size for processing pages (default: 4)"
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.9,
        help="GPU memory utilization (0-1, default: 0.9)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF rendering (default: 300)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens per page (default: 8192)"
    )
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = SmolDoclingPDFConverter(
        model_path=args.model_path,
        tensor_parallel_size=args.gpus,
        gpu_memory_utilization=args.gpu_memory,
        dpi=args.dpi,
        max_tokens=args.max_tokens
    )
    
    # Convert PDF
    try:
        output_file = converter.convert_pdf(
            args.pdf_path,
            args.output,
            args.batch_size
        )
        print(f"\n✅ Success! Markdown saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
