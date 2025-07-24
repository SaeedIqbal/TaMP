import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from dataLoader import get_dataloader, DATASET_CONFIG
from tpa import TemporalPrototypeAdapter
from dpm import DynamicPrototypeManager
from tcr import TemporalConsistencyRefinement
from agpn import AdaptiveGraphPropagationNetwork

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate TaMP-Med')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--dataset', type=str, required=True, choices=['NIH', 'BRaTS', 'Camelyon16', 'PANDA'])
    parser.add_argument('--domain_id', type=str, required=True, help='Domain ID to evaluate on')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='./results')
    return parser.parse_args()

def load_checkpoint(model_path, dataset, device):
    checkpoint = torch.load(model_path, map_location=device)
    config = DATASET_CONFIG[dataset]
    feature_dim = 2048 if 'resnet' in checkpoint.get('backbone', 'resnet50') else 384
    num_classes = len(dataLoader.CLASS_LABELS[dataset])

    # Recreate components
    tpa = TemporalPrototypeAdapter(feature_dim, num_classes, dataset).to(device)
    dpm = DynamicPrototypeManager(feature_dim, dataset).to(device)
    tcr = TemporalConsistencyRefinement(num_classes, dataset).to(device)
    agpn = AdaptiveGraphPropagationNetwork(feature_dim, num_classes, dataset).to(device)

    # Load states
    if 'tpa_state_dict' in checkpoint:
        tpa.set_hidden_states(checkpoint['tpa_state_dict'])
    if 'dpm_prototype_bank' in checkpoint:
        dpm.prototype_bank = checkpoint['dpm_prototype_bank']
    if 'tcr_history' in checkpoint:
        tcr.prediction_history = checkpoint['tcr_history']
    if 'agpn_prev_prototypes' in checkpoint:
        agpn.prev_prototypes = checkpoint['agpn_prev_prototypes']

    # Recreate classifier
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    classifier.load_state_dict(checkpoint['classifier_state_dict'])

    return classifier, tpa, dpm, tcr, agpn

def evaluate_model(classifier, dataloader, device):
    classifier.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())
    accuracy = 100 * correct / total
    all_preds = torch.softmax(torch.cat(all_preds, dim=0), dim=1)
    all_labels = torch.cat(all_labels, dim=0)
    return accuracy, all_preds, all_labels

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    classifier, tpa, dpm, tcr, agpn = load_checkpoint(args.model_path, args.dataset, device)
    print(f"Loaded model from {args.model_path}")

    # Load evaluation data
    dataloader = get_dataloader(args.dataset, args.domain_id, batch_size=args.batch_size, is_train=False)

    # Evaluate
    accuracy, predictions, labels = evaluate_model(classifier, dataloader, device)
    print(f"Evaluation Accuracy on {args.dataset} - {args.domain_id}: {accuracy:.2f}%")

    # Compute metrics
    tcs = tcr.get_temporal_consistency_score()
    me = dpm.get_memory_efficiency(current_domain=5)  # Example value

    # Save results
    results = {
        'accuracy': accuracy,
        'tcs': tcs,
        'me': me,
        'predictions': predictions.numpy(),
        'labels': labels.numpy()
    }
    results_path = os.path.join(args.output_dir, f'eval_results_{args.dataset}_{args.domain_id}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {results_path}")

if __name__ == '__main__':
    main()