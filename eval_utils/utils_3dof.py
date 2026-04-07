import sys
sys.path.append('.')
import torch
import os
import pickle
import time
import numpy as np
import gzip
import faiss
from torch.utils.data import DataLoader
from tqdm import tqdm
import dataset.infer_datasets as infer_datasets

def infer(model, eval_set, device='cuda'):
    test_data_loader = DataLoader(dataset=eval_set, num_workers=4, batch_size=1, shuffle=False)

    model.to(device)
    model.eval()
    with torch.no_grad():
        
        all_global_descs = []
        all_local_feats = []
        for _, (imgs, _) in enumerate(tqdm(test_data_loader)):
            imgs = imgs.to(device)
            _ , local_feat, global_desc = model(imgs, return_local_feats=False)
            # _ , local_feat, global_desc = model(imgs)
            
            all_global_descs.append(global_desc.detach().cpu().numpy())
           
        return np.concatenate(all_global_descs, axis=0)

def infer_load_dataset(model, model_name, dataset_name, seq, device, skip_infer=False, **kwargs):
    real_dataset_name = dataset_name
    if dataset_name.startswith('GardenHusky') and seq == 'garden_db':
        real_dataset_name = 'Garden360'
    elif dataset_name.startswith('MCD-Cross'):
        real_dataset_name = 'MCD-OS' if seq == 'ntu_night_08' else 'MCD'
    elif dataset_name.startswith('Garden-Cross'):
        real_dataset_name = 'Garden360' if seq == 'garden_db' else 'Garden'
    
    test_set = infer_datasets.InferDataset(seq, real_dataset_name, **kwargs)
    test_set.cache_str = dataset_name
    pkl_file = f'{model_name}_{test_set.cache_str}_{test_set.raw_seq}_global_desc.pkl.gz'
    pkl_path = os.path.join('./cache_global_descs/', pkl_file)
    os.makedirs('./cache_global_descs/', exist_ok=True)
    if skip_infer and os.path.exists(pkl_path):
        print(f"Loading global descriptors from {pkl_path}")
        with gzip.open(pkl_path, 'rb') as f:
            global_desc = pickle.load(f)
    else:
        global_desc = infer(model, test_set, device=device)
        # save global descriptors to pkl
        with gzip.open(pkl_path, 'wb') as f:
            pickle.dump(global_desc, f)
        print(f"Global descriptors saved to {pkl_path}")
    
    return test_set, global_desc


def evaluate_pr(model, model_name, datasets_config, writer = None, epoch = None, device='cuda', search_time=False, skip_infer=False, **kwargs):
    iou_thres = 0.25
    if search_time:
        os.makedirs('./search_time/', exist_ok=True)

    all_results = []
    for dataset_name, seqs in datasets_config:

        print(f"Evaluating localization on {dataset_name} with scenes: {seqs}")

        database_set, database_global_desc = infer_load_dataset(model, model_name, dataset_name, seqs[0], device, skip_infer=skip_infer, **kwargs)

        # database for localization
        faiss_index = faiss.IndexFlatL2(database_global_desc.shape[1])
        faiss_index.add(database_global_desc)

        dataset_recalls = []
        for i in range(1, len(seqs)):
            eval_dataset, eval_global_desc = infer_load_dataset(model, model_name, dataset_name, seqs[i], device, skip_infer=skip_infer, **kwargs)
            _, predictions = faiss_index.search(eval_global_desc, 1)  # top1
            # save predictions to pkl
            predictions_file = f'{model_name}_{database_set.cache_str}_{eval_dataset.raw_seq}_{database_set.raw_seq}_predictions.pkl.gz'
            predictions_path = os.path.join('./cache_match/', predictions_file)
            os.makedirs('./cache_match/', exist_ok=True)
            with gzip.open(predictions_path, 'wb') as f:
                pickle.dump(predictions, f)
            print(f"Predictions saved to {predictions_path}")

            all_positives = 0
            tp = 0

            iou_matrix = infer_datasets.load_or_create_iou_cache(database_set, eval_dataset)

            search_time_records = []
            for q_idx, pred in enumerate(tqdm(predictions)):
                if q_idx in eval_dataset.skip_querys:
                    continue

                start_ptr = iou_matrix.indptr[q_idx]
                end_ptr = iou_matrix.indptr[q_idx + 1]
                db_indices = iou_matrix.indices[start_ptr:end_ptr]
                ious = iou_matrix.data[start_ptr:end_ptr]

                if search_time:
                    time_init = time.time()
                    _, test_pred = faiss_index.search(eval_global_desc[q_idx].reshape(1, -1), 1)  # top1
                    time_search = time.time()
                    search_time_records.append(time_search - time_init)

                positives = []
                for k in range(len(db_indices)):
                    db_idx = db_indices[k]
                    iou = ious[k]
                    if iou > iou_thres:
                        positives.append(db_idx)
                positives = np.array(positives)

                # caclulate recall
                if len(positives) > 0:
                    all_positives += 1
                    if pred[0] in positives:
                        tp += 1

            if search_time:
                search_time_records = np.array(search_time_records)
                search_time_path = os.path.join('./search_time/', f'{model_name}_{database_set.cache_str}_{database_set.raw_seq}_{eval_dataset.raw_seq}_search_time.pkl')
                with open(search_time_path, 'wb') as f:
                    pickle.dump(search_time_records, f)
                print(f"Search time saved to {search_time_path}")

            recall = tp / all_positives if all_positives > 0 else 0.0
            dataset_recalls.append(recall)
            print(f"===> Recall on {dataset_name} {eval_dataset.seq}: {recall}, TP: {tp}, All Positives: {all_positives}")

            if writer is not None:
                writer.add_scalars('val', {f'{dataset_name}_{seqs[i]}': recall}, epoch)
            else:
                # Save PR result to CSV            
                os.makedirs('./eval_results/', exist_ok=True)
                csv_path = './eval_results/pr_results.csv'
                file_exists = os.path.exists(csv_path)
                with open(csv_path, 'a') as f:
                    if not file_exists:
                        f.write('model,dataset,query_seq,db_seq,recall,tp,all_positives\n')
                    f.write(f'{model_name},{dataset_name},{eval_dataset.raw_seq},{database_set.raw_seq},{recall:.4f},{tp},{all_positives}\n')

            del eval_dataset, eval_global_desc, predictions

        all_results.append(dataset_recalls)
        print(f"===> Mean Recall on {dataset_name}: {np.mean(dataset_recalls)}")

    avg_recall = 0.0
    for i in range(len(all_results)):
        avg_recall += np.mean(all_results[i])
    avg_recall /= len(all_results) if len(all_results) > 0 else 1
    print(f"===> Mean Recall: {avg_recall}")

    return avg_recall, all_results