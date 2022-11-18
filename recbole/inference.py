import argparse
import torch, time
import pandas as pd
from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_scores, full_sort_topk

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str, default='saved/MacridVAE-Nov-18-2022_09-59-44.pth', help='name of models')

    # python run_inference.py --model_path=/opt/ml/input/RecBole/saved/SASRecF-Apr-07-2022_03-17-16.pth 로 실행
    
    args, _ = parser.parse_known_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model, dataset 불러오기
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(args.model_path)

    model.eval()
    
    df_input = pd.read_csv('../data/sample_submission.csv')
    
    for idx,  uid in enumerate(df_input['profile_id']):
        predict = []
        
        uid_series = dataset.token2id(dataset.uid_field, [str(uid)])
        _, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=25, device=config['device'])
        external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())[0]
        predict = list(external_item_list)
        df_input.at[idx, 'predicted_list'] = predict