import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from dataLoader import get_dataloader, DATASET_CONFIG
from tpa import TemporalPrototypeAdapter
from dpm import DynamicPrototypeManager
from tcr import TemporalConsistencyRefinement
from agpn import AdaptiveGraphPropagationNetwork
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Train TaMP-Med')
    parser.add_argument('--dataset', type=str, required=True, choices=['NIH', 'BRaTS', 'Camelyon16', 'PANDA'])
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'vit_s8'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=0.6, help='TPA fusion coefficient')
    parser.add_argument('--beta', type=float, default=0.8, help='TCR Kalman gain')
    parser.add_argument('--lambda_prop', type=float, default=0.7, help='AGPN propagation coefficient')
    parser.add_argument('--tau_prune', type=float, default=0.1, help='DPM prune threshold')
    parser.add_argument('--tau_merge', type=float, default=0.2, help='DPM merge threshold')
    parser.add_argument('--max_prototypes', type=int, default=50, help='Maximum prototypes in DPM')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--domain_sequence', type=str, nargs='+', default=['D1', 'D2', 'D3', 'D4', 'D5'], help='Sequence of domain IDs')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load dataset config
    config = DATASET_CONFIG[args.dataset]
    feature_dim = 2048 if args.backbone == 'resnet50' else 384  # ViT-S/8
    num_classes = len(dataLoader.CLASS_LABELS[args.dataset])

    # Initialize TaMP-Med components
    tpa = TemporalPrototypeAdapter(feature_dim, num_classes, args.dataset, alpha=args.alpha).to(device)
    dpm = DynamicPrototypeManager(feature_dim, args.dataset, 
                                  tau_prune=args.tau_prune, tau_merge=args.tau_merge, 
                                  max_prototypes=args.max_prototypes).to(device)
    tcr = TemporalConsistencyRefinement(num_classes, args.dataset, beta=args.beta).to(device)
    agpn = AdaptiveGraphPropagationNetwork(feature_dim, num_classes, args.dataset, 
                                           lambda_prop=args.lambda_prop).to(device)

    # Dummy classifier (in practice, use a real classifier)
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    # Domain adaptation loop
    for domain_idx, domain_id in enumerate(args.domain_sequence):
        print(f"\n--- Adapting to Domain {domain_id} ---")

        # 1. Load data
        dataloader = get_dataloader(args.dataset, domain_id, batch_size=args.batch_size, is_train=True)

        # 2. Extract features and get raw predictions (simulated pseudo-labels)
        with torch.no_grad():
            features_list, labels_list = [], []
            for images, labels in dataloader:
                images = images.to(device)
                feats = classifier[:-1](images)  # Extract features from backbone
                raw_preds = torch.softmax(classifier(feats), dim=-1)
                features_list.append(feats.cpu())
                labels_list.append(raw_preds.cpu())
            features = torch.cat(features_list, dim=0)
            raw_predictions = torch.cat(labels_list, dim=0)

        # 3. TPA: Compute fused prototypes
        fused_prototypes = tpa(features, raw_predictions.argmax(dim=-1))
        fused_prototypes_tensor = torch.stack([fused_prototypes[c] for c in range(num_classes)], dim=0).to(device)

        # 4. DPM: Add and manage prototypes
        dpm.add_prototypes(fused_prototypes)
        # Simulate domain shift for AGPN
        domain_shift = 0.1 * (domain_idx + 1)  # Simulated increasing shift

        # 5. AGPN: Propagate labels
        refined_labels = agpn(fused_prototypes_tensor, raw_predictions, domain_shift=domain_shift).to(device)

        # 6. TCR: Refine predictions
        final_pseudo_labels = tcr(refined_labels).to(device)

        # 7. Self-training with pseudo-labels
        classifier.train()
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            for images, _ in dataloader:
                images = images.to(device)
                feats = classifier[:-1](images)
                outputs = classifier(feats)
                # Use refined pseudo-labels (align with batch size)
                batch_size = outputs.size(0)
                pseudo_labels = final_pseudo_labels[:batch_size].argmax(dim=-1)
                loss = criterion(outputs, pseudo_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

        # 8. Evaluate on current domain
        classifier.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = classifier(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Accuracy on {domain_id}: {100 * correct / total:.2f}%")

        # 9. Save model and state
        checkpoint = {
            'classifier_state_dict': classifier.state_dict(),
            'tpa_state_dict': tpa.get_hidden_states(),
            'dpm_prototype_bank': dpm.get_prototype_bank(),
            'tcr_history': tcr.prediction_history,
            'agpn_prev_prototypes': agpn.prev_prototypes,
            'domain_idx': domain_idx
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'tamp_med_{args.dataset}_{domain_id}.pth'))

    print("Training completed.")

if __name__ == '__main__':
    main()