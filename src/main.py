#!/usr/bin/env python

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from dotenv import load_dotenv

# Set up console and logging
console = Console()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rich")

# Load environment variables
load_dotenv()

# Create Typer app
app = typer.Typer(help="Boon Entity Mapper - Extract and map entities from documents")


@app.command()
def process(
    input_path: str = typer.Argument(..., help="Path to input document or directory"),
    output_dir: str = typer.Option(
        os.getenv("OUTPUT_DIR", "./results"), help="Directory to save results"
    ),
    model: str = typer.Option(
        os.getenv("DEFAULT_MODEL", "gpt-4o"), help="LLM model to use for extraction"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Process documents and map entities to the database."""
    # Set log level based on verbose flag
    if verbose:
        logging.getLogger("rich").setLevel(logging.DEBUG)
    
    # Validate input path
    input_path = Path(input_path)
    if not input_path.exists():
        log.error(f"Input path does not exist: {input_path}")
        raise typer.Exit(code=1)
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TODO: Implement document processing and entity mapping
    log.info(f"Processing document: {input_path}")
    log.info(f"Using model: {model}")
    log.info(f"Output directory: {output_dir}")
    
    # Placeholder for actual implementation
    log.info("✓ Processing completed successfully")


@app.command()
def version():
    """Display the current version of the tool."""
    console.print("Boon Entity Mapper v0.1.0")


if __name__ == "__main__":
    app()