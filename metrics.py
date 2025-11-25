'''
@ 2025 Omni-Mol Project

Some metrics based on https://github.com/blender-nlp/MolT5
'''
import re
import numpy as np
import selfies

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, AllChem
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from Levenshtein import distance as lev
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from rouge_score.rouge_scorer import RougeScorer
from sklearn.metrics import mean_absolute_error, r2_score
from loggers import WrappedLogger
from utils import save_json
from typing import Optional

logger = WrappedLogger(__name__)
RDLogger.DisableLog('rdApp.*')

_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def decode_selfies(selfies_str: str) -> Optional[str]:
    """Decodes a SELFIES string into a SMILES string with error handling"""
    if not selfies_str:
        return None
    try:
        return selfies.decoder(selfies_str)
    except Exception as e:
        return None

def convert_to_canonical_smiles(smiles: str) -> Optional[str]:
    """Converts a SMILES string to its canonical form."""
    if not smiles:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule:
        return Chem.MolToSmiles(molecule, isomericSmiles=False, canonical=True)
    return None

def process_entry_to_smiles(
    gt_selfies: str, pred_selfies: str
) -> dict[str, Optional[str]]:
    """
    Decodes and canonicalizes SELFIES strings.
    
    Returns a dictionary with 'gt_smi' and 'pred_smi' keys.
    """
    gt_smi_decoded = decode_selfies(gt_selfies)
    pred_smi_decoded = decode_selfies(pred_selfies)

    return {
        "gt_smi": convert_to_canonical_smiles(gt_smi_decoded),
        "pred_smi": convert_to_canonical_smiles(pred_smi_decoded)
    }

def calculate_reaction_metrics(
    data: list[dict], 
    save_path: str = None, 
    morgan_r: int = 2, 
    eos_token = "<|end|>"
) -> dict:
    """Calculate metrics for reaction tasks

    Args:
        data (list[dict]): list of dicts, containing keywords like pred and gt
        save_path (str, optional): Optional path to save the metrics. Defaults to None.
        morgan_r (int, optional): Morgan R. Defaults to 2.
        eos_token (str, optional): EOS token, if provided, will remove it from the answer. Defaults to "<|end|>".

    Returns:
        dict: A metric dict.
    """
    total_entries = len(data)
    valid_predictions = 0
    num_exact_match = 0
    
    sim_scores = {'maccs': [], 'rdk': [], 'morgan': []}
    lev_scores = {'selfies': [], 'smiles': []}
    bleu_references = []
    bleu_hypotheses = []
    
    pbar = tqdm(total=len(data), desc="Reaction Metrics")
    for entry in data:
        pbar.update(1)
        gt_selfies = entry['gt']
        if eos_token is not None:
            pred_selfies = entry['pred'].split(eos_token)[0]
        else:
            pred_selfies = entry['pred']

        smiles = process_entry_to_smiles(gt_selfies, pred_selfies)
        gt_smi = smiles['gt_smi']
        pred_smi = smiles['pred_smi']

        # 1. Validate Ground Truth
        if not gt_smi:
            pbar.write(f"Could not parse ground truth: {gt_selfies}. Skipping entry.")
            continue
        gt_mol = Chem.MolFromSmiles(gt_smi)
        if not gt_mol:
            pbar.write(f"Could not build molecule from GT SMILES: {gt_smi}. Skipping entry.")
            continue

        # 2. Validate Prediction (for validity metric)
        if not pred_smi:
            pbar.write(f"Invalid predicted SMILES: {pred_smi} from SELFIES: {pred_selfies}")
            continue
        pred_mol = Chem.MolFromSmiles(pred_smi)
        if not pred_mol:
            pbar.write(f"Could not build molecule from Pred SMILES: {pred_smi}. Skipping entry.")
            continue
        
        # 3. If we're here, the prediction is valid
        valid_predictions += 1
        
        # 4. Calculate all metrics for this valid pair
        # Fingerprints
        sim_scores['maccs'].append(DataStructs.FingerprintSimilarity(
            MACCSkeys.GenMACCSKeys(gt_mol), MACCSkeys.GenMACCSKeys(pred_mol), 
            metric=DataStructs.TanimotoSimilarity
        ))
        sim_scores['rdk'].append(DataStructs.FingerprintSimilarity(
            Chem.RDKFingerprint(gt_mol), Chem.RDKFingerprint(pred_mol), 
            metric=DataStructs.TanimotoSimilarity
        ))
        sim_scores['morgan'].append(DataStructs.TanimotoSimilarity(
            AllChem.GetMorganFingerprint(gt_mol, morgan_r), 
            AllChem.GetMorganFingerprint(pred_mol, morgan_r)
        ))

        # Levenshtein
        lev_scores['smiles'].append(lev(pred_smi, gt_smi))

        # BLEU (on SELFIES tokens)
        bleu_references.append([[c for c in gt_selfies]]) # Note: list of lists of tokens, for multiple references
        bleu_hypotheses.append([c for c in pred_selfies]) # Note: list of tokens

        # Exact Match (InChI)
        try:
            gt_inchi = Chem.MolToInchi(gt_mol)
            pred_inchi = Chem.MolToInchi(pred_mol)
            if gt_inchi == pred_inchi:
                num_exact_match += 1
        except Exception as e:
            pbar.write(f"Could not convert to InChI for {pred_smi}: {e}")


    validity_score = valid_predictions / total_entries if total_entries > 0 else 0.0
    exact_match_score = num_exact_match / valid_predictions if valid_predictions > 0 else 0.0
    maccs_sim = np.mean(sim_scores['maccs']) if sim_scores['maccs'] else 0.0
    rdk_sim = np.mean(sim_scores['rdk']) if sim_scores['rdk'] else 0.0
    morgan_sim = np.mean(sim_scores['morgan']) if sim_scores['morgan'] else 0.0
    lev_smiles = np.mean(lev_scores['smiles']) if lev_scores['smiles'] else 0.0
    bleu_score = corpus_bleu(bleu_references, bleu_hypotheses) if bleu_references else 0.0

    metrics = {
        "Validity": float(validity_score),
        "Exact Match (InChI)": float(exact_match_score),
        "SELFIES BLEU": float(bleu_score),
        "Average MACCS Similarity": float(maccs_sim),
        "Average RDK Similarity": float(rdk_sim),
        "Average Morgan Similarity": float(morgan_sim),
        "Average SMILES Levenshtein": float(lev_smiles),
        "NumValidPredictions": int(valid_predictions),
        "TotalEntries": int(total_entries)
    }

    pbar.write("--- Evaluation Metrics ---")
    for key, value in metrics.items():
        pbar.write(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    pbar.write("--------------------------")

    if save_path:
        save_json(metrics, save_path)
        pbar.write(f"Metric file saved to {save_path}")
        
    pbar.close()
        
    return metrics

def _calculate_safe_mae(gts: list[float], preds: list[float]) -> Optional[float]:
    if not gts or not preds:
        return None
    return mean_absolute_error(gts, preds)

def calculate_homolumo_metrics(
    data: list[dict], 
    save_path: str = None, 
    eos_token = "<|end|>"
) -> dict:
    """Calculate MAE for HOMOLUMO task

    Args:
        data (list[dict]): list of dicts, containing keywords like pred and gt
        save_path (str, optional): Optional path to save the metrics. Defaults to None.
        eos_token (str, optional): EOS token, if provided, will remove it from the answer. Defaults to "<|end|>".

    Returns:
        dict: A metric dict.
    """
    maes = {
        "homo": {"gts": [], "preds": []},
        "lumo": {"gts": [], "preds": []},
        "gap": {"gts": [], "preds": []}
    }
    skipped_item_count = 0

    # 1. Process Results
    pbar = tqdm(total=len(data), desc="HOMOLUMO MAE")
    for result in data:
        pbar.update(1)
        if eos_token is not None:
            pred_str = result.get('pred', '').split(eos_token)[0].strip()
        else:
            pred_str = result.get('pred', '').strip()
        gt_str = result.get('gt', '').strip()
        prompt = result.get('prompt', '')

        # Attempt to convert to float
        try:
            gt_val = float(gt_str)
            pred_val = float(pred_str)
        except ValueError as e:
            pbar.write(f"Invalid answer {pred_str}, {e}")
            skipped_item_count += 1
            continue
        
        # Classification
        is_homo = "HOMO" in prompt
        is_lumo = "LUMO" in prompt
        if is_homo and not is_lumo:
            maes["homo"]["gts"].append(gt_val)
            maes["homo"]["preds"].append(pred_val)
        elif not is_homo and is_lumo:
            maes["lumo"]["gts"].append(gt_val)
            maes["lumo"]["preds"].append(pred_val)
        elif is_homo and is_lumo:
            maes["gap"]["gts"].append(gt_val)
            maes["gap"]["preds"].append(pred_val)
        else:
            pbar.write(f"Warning: Unable to parse prompt, skipping item: {prompt}")
            skipped_item_count += 1

    # 2. Calculate Metrics
    homo_err = _calculate_safe_mae(maes["homo"]["gts"], maes["homo"]["preds"])
    lumo_err = _calculate_safe_mae(maes["lumo"]["gts"], maes["lumo"]["preds"])
    gap_err = _calculate_safe_mae(maes["gap"]["gts"], maes["gap"]["preds"])

    # Calculate total average
    total_gts = maes["homo"]["gts"] + maes["lumo"]["gts"] + maes["gap"]["gts"]
    total_preds = maes["homo"]["preds"] + maes["lumo"]["preds"] + maes["gap"]["preds"]
    average_err = _calculate_safe_mae(total_gts, total_preds)

    metrics = {
        "HOMO MAE": homo_err,
        "LUMO MAE": lumo_err,
        "GAP MAE": gap_err,
        "Average": average_err,
        "Skipped items": skipped_item_count
    }
    
    pbar.write("--- Evaluation Metrics ---")
    for key, value in metrics.items():
        pbar.write(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    pbar.write("--------------------------")
    
    if save_path:
        save_json(metrics, save_path)
        pbar.write(f"Metric file saved to {save_path}")
        
    pbar.close()
        
    return metrics

def _extract_first_float(text: str) -> Optional[float]:
    """
    Searches a string for the first number and returns it as a float.
    Returns None if invalid
    """
    match = _NUMBER_PATTERN.search(text)
    if match:
        try:
            return float(match.group(0))
        except (ValueError, TypeError):
            # This should be rare given the regex, but it's safe to handle
            return None
    return None

def calculate_mae_with_text(
    data: list[dict], 
    save_path: str = None, 
    eos_token = "<|end|>"
):
    ground_truths = []
    predictions = []
    pbar = tqdm(total=len(data), desc="MAE with Text")
    skipped_item_count = 0
    for result in data:
        pbar.update(1)
        if eos_token is not None:
            pred = result['pred'].split(eos_token)[0]
        else:
            pred = result['pred']
            
        gt = result['gt']
        ex_gt = _extract_first_float(gt)
        if ex_gt is None:
            pbar.write(f"Invalid GT {gt}")
            skipped_item_count += 1
            continue

        ex_pred = _extract_first_float(pred)
        if ex_pred is None:
            pbar.write(f"Invalid Prediction {pred}")
            skipped_item_count += 1
            continue
        
        ground_truths.append(ex_gt)
        predictions.append(ex_pred)
        
    mae = _calculate_safe_mae(ground_truths, predictions)    
    metrics = {
        "Average MAE": mae,
        "Skipped items": skipped_item_count
    }

    pbar.write("--- Evaluation Metrics ---")
    for key, value in metrics.items():
        pbar.write(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    pbar.write("--------------------------")
    
    if save_path:
        save_json(metrics, save_path)
        pbar.write(f"Metric file saved to {save_path}")
        
    pbar.close()
        
    return metrics

def calculate_r2(
    data: list[dict], 
    save_path: str = None, 
    eos_token = "<|end|>"
):
    ground_truths = []
    predictions = []
    pbar = tqdm(total=len(data), desc="R2 Score")
    skipped_item_count = 0
    for result in data:
        pbar.update(1)
        if eos_token is not None:
            pred = result['pred'].split(eos_token)[0]
        else:
            pred = result['pred']
            
        gt = result['gt']
            
        # Attempt to convert to float
        try:
            gt_val = float(gt)
            pred_val = float(pred)
        except ValueError as e:
            pbar.write(f"Invalid answer {pred}, {e}")
            skipped_item_count += 1
            continue
        
        ground_truths.append(gt_val)
        predictions.append(pred_val)
        
    r2 = r2_score(ground_truths, predictions) 
    metrics = {
        "R2 Score": r2,
        "Skipped items": skipped_item_count
    }

    pbar.write("--- Evaluation Metrics ---")
    for key, value in metrics.items():
        pbar.write(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    pbar.write("--------------------------")
    
    if save_path:
        save_json(metrics, save_path)
        pbar.write(f"Metric file saved to {save_path}")
        
    pbar.close()
        
    return metrics

def _tokenize_and_filter(
    text: str, 
    tokenizer: PreTrainedTokenizer, 
    eos_token: str, 
    max_length: int = 512
) -> list[str]:
    # Manually truncate to max_length
    tokens = tokenizer.tokenize(text)[:max_length]
    
    pad_token = tokenizer.pad_token
    
    # Use a list comprehension for clean filtering
    filtered_tokens = [
        t for t in tokens if t not in (eos_token, pad_token)
    ]
    
    return filtered_tokens

def calculate_text_scores(
    data: list[dict], 
    save_path: str = None,
    tokenizer: PreTrainedTokenizer = None,
    eos_token = "<|end|>"
):
    assert tokenizer is not None, "Please provide a tokenizer for this metric"
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    predictions_tokenized = []
    references_tokenized = []
    meteor_scores = []
    rouge_scores_raw = []
    
    pbar = tqdm(total=len(data), desc="Text Scores")
    skipped_items = 0
    for item in data:
        pbar.update(1)
        pred_text = item.get('pred')
        gt_text = item.get('gt')

        if pred_text is None or gt_text is None:
            pbar.write(f"Warning: Skipping item with missing 'pred' or 'gt': {item}")
            skipped_items += 1
            continue

        pred_tokens = _tokenize_and_filter(
            pred_text, tokenizer, eos_token
        )
        gt_tokens = _tokenize_and_filter(
            gt_text, tokenizer, eos_token
        )
        
        predictions_tokenized.append(pred_tokens)
        references_tokenized.append([gt_tokens])

        meteor_scores.append(meteor_score([gt_tokens], pred_tokens))
        
        rouge_scores_raw.append(rouge_scorer.score(gt_text, pred_text))

    if not predictions_tokenized:
        pbar.write("Error: No valid data was processed.")
        return {
            "BLEU-2": None,
            "BLEU-4": None,
            "Meteor": None,
            "ROUGE-1": None,
            "ROUGE-2": None,
            "ROUGE-L": None
        }

    # BLEU (corpus-level)
    bleu2 = corpus_bleu(
        references_tokenized, predictions_tokenized, weights=(0.5, 0.5)
    )
    bleu4 = corpus_bleu(
        references_tokenized, predictions_tokenized, weights=(0.25, 0.25, 0.25, 0.25)
    )

    avg_meteor = np.mean(meteor_scores)

    avg_rouge1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores_raw])
    avg_rouge2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores_raw])
    avg_rougeL = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores_raw])
    
    metrics = {
        "BLEU-2": bleu2,
        "BLEU-4": bleu4,
        "Meteor": avg_meteor,
        "ROUGE-1": avg_rouge1,
        "ROUGE-2": avg_rouge2,
        "ROUGE-L": avg_rougeL
    }

    pbar.write("--- Evaluation Metrics ---")
    for key, value in metrics.items():
        pbar.write(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    pbar.write("--------------------------")
    
    if save_path:
        save_json(metrics, save_path)
        pbar.write(f"Metric file saved to {save_path}")
        
    pbar.close()
        
    return metrics

def calculate_moledit_metrics(
    data: list[dict], 
    save_path: str = None,
    eos_token = "<|end|>"
):
    return None

if __name__ == "__main__":
    # Test reaction metrics
    reaction_data = [
        {"prompt": "123", "gt": "[O][=N+1][Branch1][C][O-1][C][=C][C][Branch1][Ring1][C][O][=C][Branch1][C][F][C][=C][Ring1][=Branch2][F]", "pred": "[C][=C][C][=C][C]([C](=[O])[O])[O][C](=[O])[C]<|end|>extra"},
        {"prompt": "123", "gt": "[O][=N+1][Branch1][C][O-1][C][=C][C][Branch1][Ring1][C][O][=C][Branch1][C][F][C][=C][Ring1][=Branch2][F]", "pred": "[O][=N+1][Branch1][C][O-1][C][=C][C][Branch1][Ring1][C][O][=C][Branch1][C][F][C][=C][Ring1][=Branch2][F]<|end|>"},
        {"prompt": "123", "gt": "[O][=N+1][Branch1][C][O-1][C][=C][C][Branch1][Ring1][C][O][=C][Branch1][C][F][C][=C][Ring1][=Branch2][F]", "pred": "[invalid][selfies]<|end|>"},
    ]
    calculate_reaction_metrics(reaction_data, None, 2, "<|end|>")
    
    # Test HOMOLUMO metrics
    homolumo_data = [
        {"prompt": "HOMO query", "gt": "1.0", "pred": "1.2"}, # err = 0.2
        {"prompt": "HOMO query 2", "gt": "2.0", "pred": "2.2"}, # err = 0.2
        {"prompt": "HOMO LUMO", "gt": "2.2", "pred": "1.9"},
        {"prompt": "LUMO", "gt": "9.3", "pred": "4.4"},
        {"prompt": "HOMO", "gt": "3.4", "pred": "hello"}
    ]
    calculate_homolumo_metrics(homolumo_data, None, None)
    
    # Test MAE with text
    mae_text = [
        {"prompt": "123", "gt": "It is 1.7", "pred": "It is 2.2"},
        {"prompt": "123", "gt": "It is 1.3", "pred": "It is 2.8"},
        {"prompt": "123", "gt": "It is 1.7", "pred": "hahaha"},
        {"prompt": "123", "gt": "hahaha", "pred": "It is 2.2"}
    ]
    calculate_mae_with_text(mae_text, None, None)
    
    # Test Molcap scores
    molcap_text = [
        {"prompt": "123", "gt": "The molecule is a quinolinemonocarboxylate that is the conjugate base of xanthurenic acid, obtained by deprotonation of the carboxy group. It has a role as an animal metabolite. It is a conjugate base of a xanthurenic acid.", "pred": "The molecule is a quinolinemonocarboxylate that is the conjugate base of xanthurenic acid"},
        {"prompt": "123", "gt" : "The molecule is a 3beta-hydroxy-4,4-dimethylsteroid that is that is cholestan-3beta-ol in which the hydrogens at position 4 have been replaced by methyl groups and a double bond has been introduced between positions 8 and 14.", "pred": "The molecule is a 3beta-hydroxy-4,4-dimethylsteroid that is that is cholestan-3beta-ol in which the hydrogens"}
    ]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("model/Llama-3.2-1B-Instruct")
    calculate_text_scores(molcap_text, None, tokenizer, None)
