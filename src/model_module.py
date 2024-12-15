import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MultilabelF1Score


class TrainingModule(pl.LightningModule):
    def __init__(self, model, cfg):
        super(TrainingModule, self).__init__()
        self.model = model
        self.cfg = cfg

        self.ssl = True if self.cfg.loss_fn == "ssl" else False
    
        self.val_step_results = []
        self.test_step_output = {'pred': [], "label": []}

    def training_step(self, batch, batch_idx):
        sample = batch
        if self.cfg.model_choice == "mlp":
            padded_data, mask, label = sample
            pred = self.model(padded_data, mask, label, SSL=self.ssl)
        elif self.cfg.model_choice == "graph":
            data = sample
            pred = self.model(data, SSL=self.ssl, mode="train")
            label = sample.y_true
        loss = self.calculate_loss(pred, label)
        scores = self.calculate_metrics(pred,label)
        record_dict = {"loss":loss, **scores}
        self._logger(record_dict, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        sample = batch
        if self.cfg.model_choice == "mlp":
            padded_data, mask, label = sample
            pred = self.model(padded_data, mask, label, SSL=self.ssl)
        elif self.cfg.model_choice == "graph":
            data = sample
            pred = self.model(data, SSL=self.ssl, mode="valid")
            label = sample.y_true
        loss = self.calculate_loss(pred, label)
        scores = self.calculate_metrics(pred,label)
        record_dict = {"loss":loss, **scores}
        self._logger(record_dict, "val")

        self.val_step_results.append(scores["f1"].item())

        return loss
    
    def on_validation_epoch_end(self):
        val_epoch_avg_result =  sum(self.val_step_results)/len(self.val_step_results)
        self.log("val_metric_score", val_epoch_avg_result)
        self.val_step_results.clear()
    
    def test_step(self, batch, batch_idx):
        sample = batch
        if self.cfg.model_choice == "mlp":
            padded_data, mask, label = sample
            pred = self.model(padded_data, mask, label, SSL=self.ssl)
        elif self.cfg.model_choice == "graph":
            data = sample
            pred = self.model(data, SSL=self.ssl, mode="test")
            label = sample.y_true
        if self.ssl:
            pred = pred[0]
        pred = F.sigmoid(pred)

        self.test_step_output['pred'].append(pred)
        self.test_step_output['label'].append(label)
        
    def on_test_epoch_end(self):

        test_preds = torch.cat(self.test_step_output['pred'], dim=0)
        test_labels = torch.cat(self.test_step_output['label'], dim=0)

        self.test_step_output.clear()

        self.test_result =  {'test_preds': test_preds, 'test_labels': test_labels}


    def predict_step(self, batch, batch_idx):
        sample = batch
        if self.cfg.model_choice == "mlp":
            padded_data, mask, label = sample
            pred = self.model(padded_data, mask, label, SSL=self.ssl)
        elif self.cfg.model_choice == "graph":
            data = sample
            pred = self.model(data, SSL=self.ssl, mode="predict")
            label = sample.y_true
        if self.ssl:
            pred = pred[0]
        pred = F.sigmoid(pred)
        return {"pred": pred, "label": label}
    
    def _logger(self, record_dict, dclass):
        for k,v in record_dict.items():
            name = f"{dclass}_{k}"
            self.log(name, v, 
            on_step = False,
            on_epoch= True,
            prog_bar = True,
            batch_size = self.cfg.batch_size
            )

    def calculate_loss(self, output, labels):

        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        
        if self.cfg.loss_fn == "ssl":
            common_loss = criterion(output[0], labels)
            triplet_loss = output[1]
            loss = common_loss + triplet_loss * 0.1
        else:
            loss = criterion(output, labels)
        
        return loss
    
    def configure_optimizers(self):
        
        optimizer_dict = {
                "adam": torch.optim.Adam,
                "adamw": torch.optim.AdamW
        }

        schedule_dict = {
                "reduce": torch.optim.lr_scheduler.ReduceLROnPlateau,
                "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
                "steplr": torch.optim.lr_scheduler.StepLR,

        }

        optimizer = optimizer_dict[self.cfg.optimizer](self.model.parameters
        (), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        
        if self.cfg.lrs:
            schedule_params = {}
            if self.cfg.lrs == "reduce":
                schedule_params = {"mode": "min", "factor": 0.1, "patience": 10}
            elif self.cfg.lrs == "cosine":
                schedule_params = {"T_max": 50, "eta_min": 0}
            elif self.cfg.lrs == "steplr":
                schedule_params = {"step_size": 30, "gamma": 0.1}
            
            scheduler = schedule_dict[self.cfg.lrs](optimizer, **schedule_params)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        
        return {"optimizer": optimizer}

    def calculate_metrics(self, output, labels):

        if self.cfg.loss_fn == "ssl":
            output = output[0]

        pred_flat = torch.round(torch.sigmoid(output)).int()
        f1_metric = MultilabelF1Score(
            num_labels=self.cfg.label_num, average='micro').cuda()
        f1 = f1_metric(pred_flat, labels)
        return {"f1": f1}

