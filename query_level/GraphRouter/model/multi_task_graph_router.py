import random
import numpy as np
import torch
from graph_nn import  form_data,GNN_prediction
from data_processing.utils import savejson,loadjson,savepkl,loadpkl
import pandas as pd
import json
import re
import yaml
from sklearn.preprocessing import MinMaxScaler
device = "cpu" if torch.cuda.is_available() else "cpu"

class graph_router_prediction:
    def __init__(self, router_data_path,llm_path,llm_embedding_path,config,wandb):
        self.config = config
        self.wandb = wandb
        self.data_df = router_data_path
        self.llm_description = loadjson(llm_path)
        self.llm_names = list(self.llm_description.keys())
        self.num_llms=len(self.llm_names)
        self.num_query=int(len(self.data_df)/self.num_llms)
        self.num_task=config['num_task']
        self.set_seed(self.config['seed'])
        self.llm_description_embedding=loadpkl(llm_embedding_path)
        self.prepare_data_for_GNN()
        self.split_data()
        self.form_data = form_data(device)
        # import pdb; pdb.set_trace()
        self.query_dim = self.query_embedding_list.shape[1]
        self.llm_dim = self.llm_description_embedding.shape[1]
        self.GNN_predict = GNN_prediction(query_feature_dim=self.query_dim, llm_feature_dim=self.llm_dim,
                                    hidden_features_size=self.config['embedding_dim'], in_edges_size=self.config['edge_dim'],wandb=self.wandb,config=self.config,device=device)
        print("GNN training successfully initialized.")
        self.train_GNN()


    def set_seed(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def split_data(self):
        self.query_per_task=int(self.num_query/self.num_task)
        split_ratio = self.config['split_ratio']

        # Calculate the size of each set for each task
        train_size = int(self.query_per_task * split_ratio[0])
        val_size = int(self.query_per_task * split_ratio[1])
        test_size = int(self.query_per_task * split_ratio[2])

        # Generate indices
        train_idx = []
        validate_idx = []
        test_idx = []

        for task_id in range(self.num_task):
            # Starting index for each task
            start_idx = task_id * self.query_per_task * self.num_llms

            # Add training set indices
            train_idx.extend(range(start_idx, start_idx + train_size* self.num_llms))

            # Add validation set indices
            validate_idx.extend(range(start_idx + train_size* self.num_llms,
                                      start_idx + train_size* self.num_llms + val_size* self.num_llms))

            # Add test set indices
            test_idx.extend(range(start_idx + train_size* self.num_llms + val_size* self.num_llms,
                                  start_idx + train_size* self.num_llms + val_size* self.num_llms + test_size* self.num_llms))


        self.combined_edge=np.concatenate((self.cost_list.reshape(-1,1),self.effect_list.reshape(-1,1)),axis=1)
        self.scenario=self.config['scenario']
        if self.scenario== "Performance First":
            self.effect_list = 1.0 * self.effect_list - 0.0 * self.cost_list
        elif self.scenario== "Balance":
            self.effect_list = 0.5 * self.effect_list - 0.5 * self.cost_list
        else:
            self.effect_list = 0.2 * self.effect_list - 0.8 * self.cost_list

        effect_re=self.effect_list.reshape(-1,self.num_llms)
        self.label=np.eye(self.num_llms)[np.argmax(effect_re, axis=1)].reshape(-1,1)
        self.edge_org_id=[num for num in range(self.num_query) for _ in range(self.num_llms)]
        self.edge_des_id=list(range(self.edge_org_id[0],self.edge_org_id[0]+self.num_llms))*self.num_query

        self.mask_train =torch.zeros(len(self.edge_org_id))
        self.mask_train[train_idx]=1

        self.mask_validate = torch.zeros(len(self.edge_org_id))
        self.mask_validate[validate_idx] = 1

        self.mask_test = torch.zeros(len(self.edge_org_id))
        self.mask_test[test_idx] = 1

    def check_tensor_values(self):
        def check_array(name, array):
            array = np.array(array)  # Ensure it's a numpy array
            has_nan = np.isnan(array).any()
            out_of_bounds = ((array < 0) | (array > 1)).any()
            if has_nan or out_of_bounds:
                print(f"[Warning] '{name}' has invalid values:")
                if has_nan:
                    print(f" - Contains NaN values.")
                if out_of_bounds:
                    min_val = np.min(array)
                    max_val = np.max(array)
                    print(f" - Values outside [0, 1] range. Min: {min_val}, Max: {max_val}")
            else:
                print(f"[OK] '{name}' is valid (all values in [0, 1] and no NaNs).")

        check_array("query_embedding_list", self.query_embedding_list)
        check_array("task_embedding_list", self.task_embedding_list)
        check_array("effect_list", self.effect_list)
        check_array("cost_list", self.cost_list)
    
    def prepare_data_for_GNN(self):
        unique_index_list = list(range(0, len(self.data_df), self.num_llms))
        query_embedding_list_raw = self.data_df['query_embedding'].tolist()
        task_embedding_list_raw = self.data_df['task_description_embedding'].tolist()
        self.query_embedding_list = []
        self.task_embedding_list = []

        def parse_embedding(tensor_str):
            if pd.isna(tensor_str) or not isinstance(tensor_str, str):
                return []

            tensor_str = tensor_str.replace('tensor(', '').replace(')', '')
            try:
                values = json.loads(tensor_str)
            except:
                numbers = re.findall(r'[-+]?\d*\.\d+|\d+', tensor_str)
                values = [float(x) for x in numbers]
            return np.nan_to_num(values, nan=0.0).tolist()

        # Extract and clean query embeddings
        for i in range(0, len(query_embedding_list_raw), self.num_llms):
            embedding = parse_embedding(query_embedding_list_raw[i])
            self.query_embedding_list.append(embedding)

        # Extract and clean task embeddings
        for i in range(0, len(task_embedding_list_raw), self.num_llms):
            embedding = parse_embedding(task_embedding_list_raw[i])
            self.task_embedding_list.append(embedding)

        # Convert to numpy arrays
        self.query_embedding_list = np.array(self.query_embedding_list, dtype=np.float32)
        self.task_embedding_list = np.array(self.task_embedding_list, dtype=np.float32)

        # Normalize embeddings to [0, 1]
        def normalize_array(arr):
            scaler = MinMaxScaler()
            return scaler.fit_transform(arr)

        # self.query_embedding_list = normalize_array(self.query_embedding_list)
        # self.task_embedding_list = normalize_array(self.task_embedding_list)

        # Process and normalize effect and cost lists
        effect_raw = np.nan_to_num(self.data_df['normalized_performance'].tolist(), nan=0.0)
        cost_raw = np.nan_to_num(self.data_df['normalized_cost'].tolist(), nan=0.0)

        # effect_raw = np.array(effect_raw, dtype=np.float32).reshape(-1, 1)
        # cost_raw = np.array(cost_raw, dtype=np.float32).reshape(-1, 1)

        # self.effect_list = normalize_array(effect_raw).flatten()
        # self.cost_list = normalize_array(cost_raw).flatten()
        self.effect_list=effect_raw.flatten()
        self.cost_list =cost_raw.flatten()


        self.check_tensor_values()

    def train_GNN(self):

        self.data_for_GNN_train = self.form_data.formulation(task_id=self.task_embedding_list,
                                                             query_feature=self.query_embedding_list,
                                                             llm_feature=self.llm_description_embedding,
                                                             org_node=self.edge_org_id,
                                                             des_node=self.edge_des_id,
                                                             edge_feature=self.effect_list, edge_mask=self.mask_train,
                                                             label=self.label, combined_edge=self.combined_edge,
                                                             train_mask=self.mask_train, valide_mask=self.mask_validate,
                                                             test_mask=self.mask_test)
        self.data_for_GNN_validate = self.form_data.formulation(task_id=self.task_embedding_list,
                                                                query_feature=self.query_embedding_list,
                                                                llm_feature=self.llm_description_embedding,
                                                                org_node=self.edge_org_id,
                                                                des_node=self.edge_des_id,
                                                                edge_feature=self.effect_list,
                                                                edge_mask=self.mask_validate, label=self.label,
                                                                combined_edge=self.combined_edge,
                                                                train_mask=self.mask_train,
                                                                valide_mask=self.mask_validate,
                                                                test_mask=self.mask_test)

        self.data_for_test = self.form_data.formulation(task_id=self.task_embedding_list,
                                                        query_feature=self.query_embedding_list,
                                                        llm_feature=self.llm_description_embedding,
                                                        org_node=self.edge_org_id,
                                                        des_node=self.edge_des_id,
                                                        edge_feature=self.effect_list, edge_mask=self.mask_test,
                                                        label=self.label, combined_edge=self.combined_edge,
                                                        train_mask=self.mask_train, valide_mask=self.mask_validate,
                                                        test_mask=self.mask_test)

        # import pdb; pdb.set_trace()
        self.GNN_predict.train_validate(data=self.data_for_GNN_train, data_validate=self.data_for_test,data_for_test=self.data_for_test)

    def test_GNN(self):
        predicted_result = self.GNN_predict.test(data=self.data_for_test,model_path=self.config['model_path'])




if __name__ == "__main__":
    import wandb
    with open("configs/config.yaml", 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    wandb_key = config['wandb_key']
    wandb.login(key=wandb_key)
    wandb.init(project="graph_router")
    graph_router_prediction(router_data_path=config['saved_router_data_path'],llm_path=config['llm_description_path'],
                            llm_embedding_path=config['llm_embedding_path'],config=config,wandb=wandb)