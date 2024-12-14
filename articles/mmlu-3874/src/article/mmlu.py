from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
from random import sample
import shutil
import tarfile
import tempfile
from typing import Callable, Iterator, Mapping, NamedTuple, Sequence, override

from IPython.display import display
from llama_models.llama3.api import Tokenizer
from llama_models.llama3.reference_impl.model import RMSNorm
import numpy as np
from pandas import DataFrame
import requests
from rich.progress import Progress
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import trange

from .llama import LlamaHead, ModelConfig, load_tokenizer, LlamaModel
from .tools import executor, default_arg

__all__ = [
    "Question",
    "Answer",
    "OPTIONS",
    "Questions",
    "Answers",
    "download_dataset",
    "load_dataset",
    "generate_prompt",
    "display_questions",
]


class Question(NamedTuple):
    """Represents an MMLU question."""

    qid: int
    category: str
    question: str
    A: str
    B: str
    C: str
    D: str
    answer: str


class Answer(NamedTuple):
    """Represents an answer to MMLU question."""

    qid: int

    expected: str

    actual: str

    scores: dict[str, float]

    correct: bool


OPTIONS = tuple(["A", "B", "C", "D"])
Questions = Sequence[Question]
Answers = Sequence[Answer]

def download_dataset(output_path: Path):

    # Check if it exists already
    if output_path.exists():
        print(f"Dataset {output_path.name} exists. Skipping.")
        return
    
    dataset_url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

    work_dir = tempfile.TemporaryDirectory()
    work_path = Path(work_dir.name)

    # Download tarball
    response = requests.get(dataset_url, stream=True)
    total = int(response.headers["Content-Length"])
    
    with Progress() as progress, tempfile.NamedTemporaryFile() as tarball:
        task = progress.add_task("Downloading...", total=total)

        for data in response.iter_content(chunk_size=5*1024*1024):
            tarball.write(data)            
            progress.update(task, advance=len(data), refresh=True)
        
        with tarfile.open(tarball.name) as tf:
            tf.extractall(work_path, filter="data")
        
        shutil.move(work_path / "data", output_path)

        
def load_dataset(dataset_path: Path) -> tuple[Questions, Questions]:
    """Load MMLU examples and questions."""
    
    def load_data_file(path: Path) -> Questions:        
        # Infer category from file name: x_y_z_test.csv -> x y z
        category = " ".join(path.stem.split("_")[0:-1])
    
        with open(path, mode="r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            questions = tuple(Question(i, category, *row) for i, row in enumerate(reader))
    
        return questions

    def load_segment(segment: str) -> Questions:
        # Sort paths to ensure consistent order
        paths = sorted(path for path in dataset_path.glob(f"{segment}/*.csv"))
    
        # Load data files in parallel
        futures = [executor.submit(load_data_file, path) for path in paths]
    
        # Collect results
        collected = ()
        for future in futures:
            collected += future.result()
    
        # Reassign ids
        questions = ()
        for i, question in enumerate(collected):
            questions += (Question(i, *question[1:]),)
    
        return questions

    examples = load_segment("dev")
    questions = load_segment("test")

    return examples, questions


def generate_prompt(
    question: Question,
    *,
    n_shots: int | None = None,
    examples: Questions | None = None,
    header: bool | None = None,
):
    """Generate prompt for specified question."""
    # Defaults
    n_shots = default_arg(n_shots, 0)
    header = default_arg(header, True)

    # Validate
    if n_shots < 0 or n_shots > 5:
        raise ValueError("n_shots must be between 0 and 5")
    
    if n_shots > 0 and examples is None:
        raise ValueError("n_shots specified without examples")

    selected_examples = None
    if n_shots > 0:
        # Select examples for category
        selected_examples = [e for e in examples if e.category == question.category]

        # Deterministically select n_shots if specified
        selected_examples = selected_examples[:n_shots]

    content = ""

    # Start with examples
    if header:
        content += f"The following are multiple choice questions (with answers) about {question.category}.\n\n"

    if selected_examples:
        for row in selected_examples:
            content += f"Question: {row.question}\n\nA) {row.A}\nB) {row.B}\nC) {row.C}\nD) {row.D}\n\nAnswer: {row.answer}\n\n"

    # Pose question
    content += (
        f"Question: {question.question}\n"
        f"\n"
        f"A) {question.A}\n"
        f"B) {question.B}\n"
        f"C) {question.C}\n"
        f"D) {question.D}\n"
        f"\n"
        f"Answer: "
    )

    return content


def display_questions(questions: Questions):
    """Render questions as a table."""
    display(DataFrame(questions))

    
class MMLULlamaHead(LlamaHead):
    """Custom Llama head for MMLU."""

    def __init__(self, config: ModelConfig, device: torch.device, tokenizer: Tokenizer):
        super().__init__(config, device)

        # Calculate token ids for each MMLU option        
        self.token_ids = {option: tokenizer.encode(option, bos=False, eos=False)[0] for option in OPTIONS}

    def forward(self, x: Tensor) -> Mapping[str, float]:
        # Project semantic embeddings to token space
        x = super().forward(x)

        # Select logits for MMLU options
        logits = torch.tensor([x[self.token_ids[option]] for option in OPTIONS], device=x.device)

        # Convert to scores
        scores = F.softmax(logits, dim=-1)

        # Map options to scores
        scores = {option: scores[i] for i, option in enumerate(OPTIONS)}

        # Convert scores back to floats
        scores = {k: v.item() for k, v in scores.items()}

        return scores


class MMLULlamaGenerator(nn.Module):

    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()

        self.device = device
        
        self.tokenizer = load_tokenizer(config)
        
        self.model = LlamaModel(config, device)

        self.head = MMLULlamaHead(config, device, self.tokenizer)

    def __call__(
        self,
        questions: Questions,
        n_shots: int | None = None,
        examples: Questions | None = None,
    ) -> Iterator[Answer]:
        """Generate answers."""
        
        # Prepare models
        self.model.eval()
        self.head.eval()

        with torch.no_grad():
            
            for question in questions:
                
                # Generate prompt
                prompt = generate_prompt(question, n_shots=n_shots, examples=examples)
                
                # Split raw text into tokens
                token_ids = self.tokenizer.encode(prompt, bos=True, eos=False)

                # Load token ids into a tensor
                x = torch.tensor(token_ids, device=self.device)

                # Transform tokens into semantic embeddings
                x = self.model(x)
        
                # Score MMLU options
                scores = self.head(x)
        
                # Calculate answer
                actual = max(scores, key=scores.get)

                # Yield answer
                yield Answer(
                    qid=question.qid,
                    expected=question.answer,
                    actual=actual,
                    scores=scores,
                    correct=(actual == question.answer),
                )