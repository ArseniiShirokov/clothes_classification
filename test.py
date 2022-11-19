from collections import defaultdict
from hydra.utils import get_original_cwd
from sklearn.metrics import balanced_accuracy_score
import hydra
from omegaconf import DictConfig, open_dict
from utils.augmentation import get_aug
from utils.data_utils import TestDataset, DataLoader
from utils.batch_samplers import get_sampler
from utils.models import get_model
from utils.test_time_mapping import mapping
from tqdm import tqdm
import torch
import os
import pandas as pd






class Evaluator:
    def __init__(self, config: DictConfig):
        self.config = config['version']
        self.data_dir = config['Data']['data directory']
        self.datasets = config['Data']['test datasets']
        self.transform = get_aug(config['version']['Transform'], 'test')
        self.save_dir = config['version']['Experiment']['logs directory']
        self.attributes = [attribute['name'] for attribute in self.config['mapping']]
        self._init_dataloader()
        self._init_model()
        self._init_save_dataframe()

    def _init_model(self):
        self.model = get_model(self.config['Model']['architecture'], classes=self.config['mapping'])
        params = os.path.join(self.save_dir, 'best_model.params')
        checkpoint = torch.load(params)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.cuda()
        self.model.eval()

    def _init_save_dataframe(self):
        columns = [f"{attribute}_pr" for attribute in self.attributes] + \
                  [f"{attribute}_conf" for attribute in self.attributes] + \
                  [f"{attribute}_gt" for attribute in self.attributes]
        self.results = {dataset['name']: pd.DataFrame(columns=columns, dtype=object)
                        for dataset in self.datasets}

    def _init_dataloader(self) -> None:
        self.test_dataloaders = {}
        for dataset in self.datasets:
            # Get test dataset
            test_dataset = TestDataset(self.data_dir + '/' + dataset['name'],
                                       self.config['mapping'], transform=self.transform)
            # Get sampler: oversampling or None
            test_sampler = get_sampler(self.config, test_dataset, 'test')
            # Get test iterator
            loader = DataLoader(test_dataset, self.config['Parameters']['batch size'],
                                sampler=test_sampler)
            self.test_dataloaders[dataset['name']] = {
                'loader': loader,
                'iters': len(loader),
            }

    def evaluate(self) -> None:
        softmax = torch.nn.Softmax(dim=1)
        for test_name in self.test_dataloaders:
            loader = self.test_dataloaders[test_name]['loader']
            iters = self.test_dataloaders[test_name]['iters']
            iterator = tqdm(loader, total=iters, unit='batch', desc=test_name)
            for images, labels in iterator:
                labels = labels.cuda(non_blocking=True)
                images = images.cuda(non_blocking=True)
                with torch.no_grad():
                    predictions = self.model(images)
                    # Test-time mapping
                    predictions = mapping(
                        predictions,
                        self.config['Model']['test_time_mapping'],
                        self.attributes
                    )
                outputs = defaultdict(list)
                confidences = defaultdict(list)
                gts = defaultdict(list)
                for i, attribute in enumerate(self.attributes):
                    pr_label = torch.argmax(predictions[i].to(torch.device('cpu')), dim=1).tolist()
                    confidence = torch.max(softmax(predictions[i]).to(torch.device('cpu')), dim=1).values.tolist()
                    for j, val in enumerate(pr_label):
                        outputs[j].append(val)
                        confidences[j].append(confidence[j])
                        gts[j].append(labels[j][i].item())
                for i in outputs:
                    row = outputs[i] + confidences[i] + gts[i]
                    self.results[test_name].loc[len(self.results[test_name].index)] = row
            self.results[test_name].to_csv(f"{self.save_dir}/{test_name}.csv")

    def compute_metrics(self):
        for test_name in self.test_dataloaders:
            metrics = pd.DataFrame(columns=self.attributes)
            row = []
            for attribute in self.attributes:
                results = self.results[test_name]
                local = results[results[f"{attribute}_gt"] != -1].copy()
                pr_values = local[f"{attribute}_pr"].tolist()
                gt_values = local[f"{attribute}_gt"].tolist()
                m_acc = balanced_accuracy_score(gt_values, pr_values)
                row.append(m_acc)
            metrics.loc[len(metrics.index)] = row
            metrics.to_csv(f"{self.save_dir}/{test_name}_M-Acc.csv")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def start_test(cfg: DictConfig) -> None:
    # Set dir to save exps
    if 'logs directory' not in cfg['version']['Experiment']:
        with open_dict(cfg):
            cfg['version']['Experiment']['logs directory'] = os.getcwd()
            os.chdir(get_original_cwd())
    tester = Evaluator(cfg)
    tester.evaluate()
    tester.compute_metrics()


if __name__ == "__main__":
    start_test()
