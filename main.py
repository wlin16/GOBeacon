import os
import torch
import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.model import *
from src.model_module import TrainingModule
from src.mlp_dataset_module import MLPDataModule
from src.graph_dataset_module import GraphDataModule

from src.ProstT5_embed_extract import EmbedGenerator as GeneralEmbedGenerator
from src.graph_process import GraphGenerator

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import wandb

from src.predict import Predictor


class ModelTrainer(object):
    def __init__(self, cfg: HydraConfig):
        self.cfg = cfg

        """ model parameters """
        os.makedirs(os.path.dirname(cfg.model.model_save_path), exist_ok=True)
        self.model_checkpoint = os.path.join(self.cfg.model.model_save_path, f"{self.cfg.model.model_save_filename}.ckpt")

        
        self.load_device()
        self.define_logging()

        if cfg.general.usage in ("train", "infer"):
            self.load_data()
            self.load_model()
            self.define_trainer()
    
    def define_logging(self):
        log_file_dir = os.path.dirname(self.cfg.general.save_path_log)
        os.makedirs(log_file_dir, exist_ok=True)

        # Configure root logger
        logging.basicConfig(
            level=logging.DEBUG,
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %A %H:%M:%S',
            filename=self.cfg.general.save_path_log,
            filemode='w',
            force=True
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
    
    def load_wandb(self):
       
        cfg = self.cfg

        wandb_run_id = cfg.wandb.run_id if cfg.wandb.run_id else wandb.util.generate_id()
        self.wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.run_name}",
            id=wandb_run_id,
            resume="allow",
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    def load_device(self):
        cfg = self.cfg
        SEED = cfg.general.seed
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{cfg.general.gpu_id}")
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            logging.info(
                f'There are {torch.cuda.device_count()} GPU(s) available.')
            logging.info(f'Device name: {torch.cuda.get_device_name(0)}')
        else:
            logging.info("////////////////////////////////////////////////")
            logging.info("///// NO GPU DETECTED! Falling back to CPU /////")
            logging.info("////////////////////////////////////////////////")
            self.device = torch.device("cpu")
    
    def feature_extraction(self):
        cfg = self.cfg.dataset.feat_extract

        # generate embeddings for sequence data
        if cfg.extract_type == "sequence":
            cfg.model_name = "ESM2"
            embed_generator = GeneralEmbedGenerator(cfg, self.device)
            input_ds = os.path.join(
                cfg.ds_path, cfg.benchmark, cfg.centre_node_csv)
            embed_generator.embedding_generate(
            input_ds)

        # generate embeddings for structure data
        elif cfg.extract_type == "structure":
            cfg.model_name = "ProstT5"
            embed_generator = GeneralEmbedGenerator(cfg, self.device)
            input_ds = os.path.join(
                cfg.ds_path, cfg.benchmark, cfg.centre_node_csv)
            embed_generator.embedding_generate(
            input_ds)
            
        # generate embeddings for graph data
        elif cfg.extract_type == "graph":
            graph_data_loader = GraphGenerator(cfg, self.device)
            centre_node_csv = os.path.join(
                cfg.ds_path, cfg.benchmark, cfg.centre_node_csv)
            graph_data_loader.graph_generator(
                cfg.ppi_path, centre_node_csv)
        else:
            raise ValueError("Invalid extract_type, choose from `sequence`, `structure` or `graph`")

    def load_model(self):
        config = self.cfg.model
        config.label_num = self.label_num
        config.sub_ontology = self.cfg.dataset.load_data.sub_ontology.lower()

        logging.info(f'model_size: {config.num_features}')
        logging.info(f'num_hidden: {config.hidden}\n')

        logging.info("Loading model...")

        model_mgr = {
            "mlp": CustomMLP,
            "graph": GoModel,
        }

        self.model_object = model_mgr[config.model_choice](config)
        self.model = TrainingModule(self.model_object, cfg=self.cfg.model)

        logging.info(f"Model {config.model_choice} loaded.\n")
    
    def load_data(self):
        cfg = self.cfg.dataset.load_data
        self.label_num = cfg[f'{cfg.sub_ontology}_len']

        logging.info(f"Loading data for Category: {cfg.sub_ontology} ...")
        logging.info(f"Number of labels: {self.label_num}")
        logging.info(
            f"Loading data for the purpose of ** {self.cfg.general.usage} ** ...")
        logging.info(
            f'Loading csv from {cfg.ds_path}, graphs from {cfg.lmdb_path}...')
        
        if self.cfg.model.model_choice == "mlp":
            self.data_module = MLPDataModule(self.cfg.dataset.load_data)
        elif self.cfg.model.model_choice == "graph":
            self.data_module = GraphDataModule(self.cfg.dataset.load_data)
        else:
            raise ValueError("Invalid model_choice, choose from `mlp` or `graph`")

    def define_trainer(self):
        cfg = self.cfg
         # ModelCheckpoint callback configuration
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss', # val_loss, none
            dirpath=cfg.model.model_save_path,
            filename=cfg.model.model_save_filename,
            save_top_k=1,
            mode='min',
            save_last=False, 
        )

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=cfg.model.early_stop,
            verbose=True,
            mode='min'
        )

        if not self.cfg.model.debug:
            self.load_wandb()
        else:
            self.wandb_logger = None

        self.trainer = pl.Trainer(
            logger=self.wandb_logger,
            callbacks=[checkpoint_callback, early_stop_callback],
            max_epochs=cfg.model.n_epochs, 
            accelerator='gpu',
            devices="auto", 
            accumulate_grad_batches=cfg.model.grad_accum_steps,
            precision=32,  # 16 or 32,
            check_val_every_n_epoch=1
        )

    def train(self):
       
        if os.path.exists(self.model_checkpoint):
            logging.info(f"Checkpoint Found, Loading from: {self.model_checkpoint}...")
            model_checkpoint = self.model_checkpoint
        else:
            model_checkpoint = None

        # 🔥 Train the model!
        self.trainer.fit(self.model, datamodule=self.data_module, ckpt_path=model_checkpoint)

        if not self.cfg.model.debug:
            wandb.finish()
    
    def test(self):
        # 🧐 Test the model's performance with the test set
        best_model = TrainingModule.load_from_checkpoint(checkpoint_path=self.model_checkpoint, model=self.model_object, cfg=self.cfg.model)
        self.trainer.test(model=best_model, datamodule=self.data_module)
        pred = best_model.test_result["test_preds"]
        label = best_model.test_result["test_labels"]
        predictor = Predictor(pred, label, self.cfg)
        predictor.run_metrics()

    def predict(self, model_checkpoint):
        
        trained_model = TrainingModule.load_from_checkpoint(checkpoint_path=model_checkpoint, model=self.model_object, cfg=self.cfg.model)
        predictions = self.trainer.predict(model=trained_model, datamodule=self.data_module)

        pred = torch.cat([item['pred'] for item in predictions], dim= 0)
        label = torch.cat([item['label'] for item in predictions], dim= 0)
        
        torch.save(pred, self.cfg.predict.save_name)

    
    def ensemble_prediction(self, ensemble_prediction, label):

        predictor = Predictor(ensemble_prediction, label, self.cfg)
        predictor.run_metrics()


@hydra.main(version_base=None, config_path="./config", config_name="base")
def main(cfg: HydraConfig) -> None:
    pl.seed_everything(cfg.general.seed)

    mutable_cfg = OmegaConf.to_container(cfg, resolve=True)
    mutable_cfg = OmegaConf.create(mutable_cfg)

    model_trainer = ModelTrainer(mutable_cfg)
    if cfg.general.usage == 'train':
        model_trainer.train()
        model_trainer.test()
    
    elif cfg.general.usage == 'infer':
        model_trainer.predict(cfg.predict.model_ckpt)
        
    elif cfg.general.usage == 'predict':
        prediction_result_dir =  cfg.predict.ensemble_dir
        
        all_files = [file for file in os.listdir(prediction_result_dir) if file.startswith(cfg.dataset.load_data.sub_ontology)]
        print(f"\nFound {len(all_files)} files in {prediction_result_dir}")
        print(f"Reading files: {all_files}\n")
        
        ensemble_prediction = 0
        for file in all_files:
            pred = torch.load(os.path.join(cfg.predict.ensemble_dir, file))
            ensemble_prediction += pred
        ensemble_prediction /= len(all_files)
        label = torch.zeros_like(ensemble_prediction)
        model_trainer.ensemble_prediction(ensemble_prediction, label)
                
    elif cfg.general.usage == 'feat_extract':
        model_trainer.feature_extraction()

if __name__ == "__main__":
    main()
