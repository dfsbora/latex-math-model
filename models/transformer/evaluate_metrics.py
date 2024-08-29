import re
import subprocess
import os
import tempfile
import torch

def calculate_perplexity(loss):
    return torch.exp(torch.tensor(loss))

def calculate_bleu_score(reference_text, generated_text):
    from nltk.translate.bleu_score import sentence_bleu
    return sentence_bleu([reference_text.split()], generated_text.split())

def calculate_rouge_score(reference_text, generated_text):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, generated_text)
    return scores['rougeL'].fmeasure

def calculate_token_accuracy(predicted_tokens, ground_truth_tokens):
    correct_tokens = (predicted_tokens == ground_truth_tokens).float().sum()
    total_tokens = ground_truth_tokens.numel()
    return correct_tokens / total_tokens

def calculate_f1_score(predicted_labels, ground_truth_labels):
    from sklearn.metrics import f1_score
    return f1_score(ground_truth_labels, predicted_labels, average='weighted')

def measure_inference_speed(model, tokenizer, prompt, device, max_length=512):
    import time
    start_time = time.time()
    _ = model.generate_text(tokenizer, prompt, max_length=max_length)
    inference_time = time.time() - start_time
    return inference_time

def compile_latex(latex_content):
    latex_template = r"""
    \documentclass{article}
    \usepackage{amsmath}
    \usepackage{amsthm}
    \usepackage{amsfonts}
    \usepackage{graphicx}
    \usepackage{hyperref}

    \begin{document}

    %s

    \end{document}
    """

    complete_latex_code = latex_template % latex_content

    with tempfile.NamedTemporaryFile(suffix=".tex", delete=False) as temp_file:
        tex_path = temp_file.name
        with open(tex_path, 'w') as f:
            f.write(complete_latex_code)

    result = subprocess.run(['pdflatex', '-interaction=nonstopmode', tex_path],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = result.stdout.decode('utf-8')
    stderr = result.stderr.decode('utf-8')

    os.remove(tex_path)
    for ext in ['.aux', '.log', '.pdf']:
        path = tex_path.replace('.tex', ext)
        if os.path.exists(path):
            os.remove(path)

    error_count = len(re.findall(r'! LaTeX Error:', stderr)) + len(re.findall(r'! LaTeX Error:', stdout))
    warning_count = len(re.findall(r'LaTeX Warning:', stderr)) + len(re.findall(r'LaTeX Warning:', stdout))

    return stderr if stderr else stdout, error_count, warning_count

def log_metrics(epoch, avg_loss, avg_val_loss, perplexity, bleu_score, rouge_score, token_accuracy, f1_score, inference_time, error_count=None, warning_count=None):
    metrics = {
        "epoch": epoch + 1,
        "training_loss": avg_loss,
        "validation_loss": avg_val_loss,
        "perplexity": perplexity,
        "BLEU_score": bleu_score,
        "ROUGE_L": rouge_score,
        "token_accuracy": token_accuracy,
        "F1_score": f1_score,
        "inference_time": inference_time,
    }
    if error_count is not None:
        metrics["latex_error_count"] = error_count
    if warning_count is not None:
        metrics["latex_warning_count"] = warning_count

    print(f"Epoch {epoch + 1}, Perplexity: {perplexity}, BLEU: {bleu_score}, ROUGE-L: {rouge_score}, Token Accuracy: {token_accuracy}, F1-Score: {f1_score}, Inference Time: {inference_time}, Errors: {error_count}, Warnings: {warning_count}")
    wandb.log(metrics)
