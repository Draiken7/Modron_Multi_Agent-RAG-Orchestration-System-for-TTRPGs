import os
import time
import gc
from io import BytesIO
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from typing import Union, List



from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions

from src.env import PersistentVars


class BatchDoclingParser:
    """Parses massive PDFs using purely in-memory slicing to save Disk I/O, 
    while buffering and flushing Markdown to disk to prevent RAM exhaustion."""
    def __init__ (self, chunk_size: int = 15, hold_size: int = 5):
        self.chunk_size = chunk_size
        self.hold_size = hold_size
        
        # Suppress PyTorch/OpenMP thread collisions on Windows
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        print("Initializing Docling models...")
        # self.converter = DocumentConverter()
        
        
        
        # 1. Define the pipeline options to force CUDA
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4, 
            device="cuda" # Forces PyTorch to use your NVIDIA GPU
        )
        
        # Optional: If the GPU STILL hangs on the stat blocks, uncomment the line below.
        # This disables deep table structure inference, trading table formatting for speed.
        # pipeline_options.do_table_structure = False 

        # 2. Inject the options into the DocumentConverter
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
    def _flush_buffer(self, buffer: list, temp_dir: Path, part_number: int):
        """Writes the held Markdown text to a temporary disk file and clears the buffer."""
        part_path = temp_dir / f"part_{part_number}.md"
        with open(part_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(buffer))
        print(f"--> Flushed {len(buffer)} Markdown chunks to {part_path.name}")
        buffer.clear()
        
    def _chunk_parser(self, pdf_path: Path, temp_dir: Path, part_counter: int=1):
        """Parses chunks of single file into temp files"""
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        print(f"Starting Batch parsing of {pdf_path.name} with {total_pages} pages!")
        

        hold_buffer = []
        part_counter = 1
        
        for start_idx in range(0, total_pages, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_pages)
            
            chunk_name = f"temp_chunk_{start_idx}_to_{end_idx}.pdf"
            # temp_chunk_path = pdf_path.parent / chunk_name

            
            # Write selected pages to the temp_chunk_path
            writer = PdfWriter()
            
            for i in range(start_idx, end_idx):
                writer.add_page(reader.pages[i])  
                
            # with open(temp_chunk_path, "wb") as f:
            #     writer.write(f)
                
            pdf_buffer = BytesIO()
            writer.write(pdf_buffer)
            
            # Reset the Buffer pointer to the start
            pdf_buffer.seek(0)
            
            # Wrap the memory buffer in the Doclings Document buffer stream class
            doc_stream = DocumentStream(name=chunk_name, stream=pdf_buffer)
            
            
            # Actual Processing
            print(f"Processing pages {start_idx +1} to {end_idx}..........")
            start_time = time.time()
            
            try:
                result = self.converter.convert(doc_stream)
                md = result.document.export_to_markdown()
                hold_buffer.append(md)
                # CRITICAL: Destroy result and run garbage collection to clear PyTorch tensors
                del result
                gc.collect()
                
            except Exception as e:
                print(f"Failed to process {chunk_name}: {e}")
                
            finally:
                pdf_buffer.close()
                # temp_chunk_path.unlink()
                
            print(f"Processing of chunk completed in {time.time() - start_time:.3f}s")
            
            if len(hold_buffer) >= self.hold_size:
                self._flush_buffer(hold_buffer, temp_dir, part_counter)
                part_counter += 1
                  
        # Flush any remaining items in the buffer
        if hold_buffer:
            self._flush_buffer(hold_buffer, temp_dir, part_counter)
            
        return part_counter
        
    def parse_to_markdown(self, pdf_path: Union[Path, List[Path]], output_dir: Path = None) -> str:
        """Slices the PDF, parses chunks, and returns stitched Markdown."""
        if output_dir is None:
            output_dir = PersistentVars.DATA_FOLDER / pdf_path[0].parent if isinstance(pdf_path, List) else pdf_path.parent  / "mds"
        output_dir.mkdir(exist_ok=True)
        
        temp_dir = output_dir / "temp_md_parts"
        temp_dir.mkdir(exist_ok=True)
        
        if isinstance(pdf_path, List):
            part_counter = 1
            for path in pdf_path:
                part_counter = self._chunk_parser(pdf_path=path, temp_dir=temp_dir, part_counter=part_counter)
        else:
            part_counter = self._chunk_parser(pdf_path=pdf_path, temp_dir=temp_dir)
            
        # --- PHASE 2: COLLATION ---
        print("\nCollating intermediate parts into the final Markdown file...")
        final_save_path = output_dir / f"{pdf_path[0].stem}_parsed.md"
        
        with open(final_save_path, "w", encoding="utf-8") as final_file:
            # Sort the part files carefully to ensure numerical order
            part_files = sorted(temp_dir.glob("part_*.md"), key=lambda x: int(x.stem.split('_')[1]))
            
            for part_file in part_files:
                with open(part_file, "r", encoding="utf-8") as p_file:
                    final_file.write(p_file.read())
                    final_file.write("\n\n") 
                
                # Clean up intermediate file
                part_file.unlink()
                
        # Remove empty temp directory
        temp_dir.rmdir()
        
        print(f"Success! Unified Markdown saved to {final_save_path}")
        return final_save_path
        
        # # Stitch all Markdown chunks together separated by newlines
        # return "\n\n".join(full_markdown)
