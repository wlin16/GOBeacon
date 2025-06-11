
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from sklearn.metrics import classification_report

from tqdm import tqdm
import logging
import math
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.ontology import Ontology
import src.utils as utils

import torch
import torch.nn.functional as F

BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'
FUNC_DICT = {
    'CC': CELLULAR_COMPONENT,
    'MF': MOLECULAR_FUNCTION,
    'BP': BIOLOGICAL_PROCESS}

NAMESPACES = {
    'CC': 'cellular_component',
    'MF': 'molecular_function',
    'BP': 'biological_process'
}


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc


def compute_mcc(labels, preds):
    # Compute ROC curve and ROC area for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc


def evaluate_annotations(go, real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    p_total= 0
    ru = 0.0
    mi = 0.0
    fps = []
    fns = []
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = set(real_annots[i]).intersection(set(pred_annots[i]))
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
        for go_id in fp:
            mi += go.get_ic(go_id)
        for go_id in fn:
            ru += go.get_ic(go_id)
        fps.append(fp)
        fns.append(fn)
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
    ru /= total
    mi /= total
    r /= total
    if p_total > 0:
        p /= p_total
    f = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
    s = math.sqrt(ru * ru + mi * mi)
    return f, p, r, s, ru, mi, fps, fns


class Predictor:
    def __init__(self, pred, label, cfg):
        self.cfg = cfg
        self.preds = pred
        self.label = label
        self.sub_ontology = self.cfg.dataset.load_data.sub_ontology
        self.label_num = self.cfg.dataset.load_data[f'{self.sub_ontology}_len']
        print("label num: ", self.label_num)

        self.best_threshold = 0.5  # default

        utils.create_folder(cfg.general.save_path_predictions)
        utils.create_folder(cfg.general.save_figure)

    def _load_obo(self):
        cfg = self.cfg.dataset.load_data
        obo_file = cfg.go_obo
        self.go_rels = Ontology(obo_file, with_rels=True)

    def _load_annotation(self):
        cfg = self.cfg.dataset.load_data

        full_df_path = os.path.join(cfg.ds_path, cfg.benchmark, cfg.train_centre_node_csv)
        full_df = pd.read_pickle(full_df_path)[cfg.sub_ontology]
        self.terms = full_df.loc['all']
        full_df.pop('all')

        annotations = full_df[full_df.apply(lambda x: x != [])].values
        self.annotations = list(map(lambda x: set(x), annotations))

        test_df_path = os.path.join(cfg.ds_path, cfg.benchmark, cfg.target_centre_node_csv)
        self.test_df = pd.read_pickle(test_df_path)[
            cfg.sub_ontology]
        test_annotations = self.test_df[self.test_df.apply(
            lambda x: x != [])].values
        self.test_annotations = list(map(lambda x: set(x), test_annotations))

    def _calcualte_ic(self):

        self.go_rels.calculate_ic(self.annotations + self.test_annotations)
        # Print IC values of terms
        ics = {}
        for term in self.terms:
            ics[term] = self.go_rels.get_ic(term)

    def _calculate_Fmax(self):
        cfg = self.cfg.dataset.load_data

        go_set = set(self.terms)
        if self.cfg.dataset.load_data.benchmark == "CAFA3":
            go_set.remove(FUNC_DICT[cfg.sub_ontology])
        labels = self.test_df.values
        labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels))

        deep_preds = []
        for i, prot in enumerate(self.test_df.index):
            annots_dict = {}
            for j, score in enumerate(self.preds[i]):
                go_id = self.terms[j]
                annots_dict[go_id] = score
                
            deep_preds.append(annots_dict)

        print('Computing Fmax')
        self.fmax = 0.0
        self.tmax = 0.0
        self.precisions = []
        self.recalls = []
        self.smin = 1000000.0

        for t in range(0, 101):
            threshold = t / 100.0
            preds = []
            for i, row in enumerate(labels):
                annots = set()
                for go_id, score in deep_preds[i].items():
                    if score >= threshold:
                        annots.add(go_id)

                preds.append(annots)

            # Filter classes
            preds = list(map(lambda x: set(filter(lambda y: y in go_set, x)), preds))
            fscore, prec, rec, s, ru, mi, fps, fns = evaluate_annotations(self.go_rels, labels, preds)
            avg_fp = sum(map(lambda x: len(x), fps)) / len(fps)
            avg_ic = sum(map(lambda x: sum(map(lambda go_id: self.go_rels.get_ic(go_id), x)), fps)) / len(fps)
            print(f'{avg_fp} {avg_ic}')
            self.precisions.append(prec)
            self.recalls.append(rec)
            print(f'Fscore: {fscore}, Precision: {prec}, Recall: {rec} S: {s}, RU: {ru}, MI: {mi} threshold: {threshold}')
            if self.fmax < fscore:
                self.fmax = fscore
                self.tmax = threshold
            if self.smin > s:
                self.smin = s

    def _plot_result(self):

        print(
            f'Fmax: {self.fmax:0.3f}, Smin: {self.smin:0.3f}, threshold: {self.tmax}')
        precisions = np.array(self.precisions)
        recalls = np.array(self.recalls)
        sorted_index = np.argsort(recalls)
        recalls = recalls[sorted_index]
        precisions = precisions[sorted_index]
        aupr = np.trapz(precisions, recalls)
        self.aupr = aupr
        print(f'AUPR: {aupr:0.3f}')

        plt.figure()
        lw = 2
        plt.plot(recalls, precisions, color='darkorange',
                 lw=lw, label=f'AUPR curve (area = {aupr:0.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Area Under the Precision-Recall curve')
        plt.legend(loc="lower right")
        plt.savefig(self.cfg.general.save_figure)
    
    def _save_result(self):
        model_info = f"{self.cfg.model.model_choice}-{self.cfg.model.loss_fn}-{self.cfg.model.dropout}"
        
        save_prediction_path = self.cfg.general.save_path_predictions
        os.makedirs(os.path.dirname(save_prediction_path), exist_ok=True)

        data_row = {
            "Model_Info": model_info,
            "BP_fmax": None, "BP_smin": None, "BP_aupr": None,
            "MF_fmax": None, "MF_smin": None, "MF_aupr": None,
            "CC_fmax": None, "CC_smin": None, "CC_aupr": None
        }

        data_row[f"{self.sub_ontology}_fmax"] = f"{self.fmax:.3f}"
        data_row[f"{self.sub_ontology}_smin"] = f"{self.smin:.3f}"
        data_row[f"{self.sub_ontology}_aupr"] = f"{self.aupr:.3f}"
        
        if not os.path.exists(save_prediction_path):
            df = pd.DataFrame([data_row]).set_index('Model_Info')
            df.to_csv(save_prediction_path)
        else:
            df = pd.read_csv(save_prediction_path, index_col='Model_Info', dtype=str)
            df.at[model_info, f"{self.sub_ontology}_fmax"] = f"{self.fmax:.3f}"
            df.at[model_info, f"{self.sub_ontology}_smin"] = f"{self.smin:.3f}"
            df.at[model_info, f"{self.sub_ontology}_aupr"] = f"{self.aupr:.3f}"
            df.to_csv(save_prediction_path)

    def run_metrics(self):
        
        self._load_obo()
        self._load_annotation()

        self._calcualte_ic()
        self._calculate_Fmax()
        self._plot_result()
        self._save_result()