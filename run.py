import os
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from argparse import ArgumentParser
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor
import torch
import yaml
from utils import get_config, UK_biobank_data_module, seed_everything, Camera_Lightning_Split, SLO_Lightning_Split, Retinal_Cond_Lightning_Split,  Pickle_Lightning, load_finetune_checkpoint, EMAWeightAveraging, Resampling_Lightning_Split
from model_architecture import LightningDDPM_monai, Pretrained_LightningDDPM_monai,Conditional_DDIM_monai,MLP_classifier, Restnet_50, EfficientNet_B3, Swin_B, EfficientNet_B0
torch.set_float32_matmul_precision('medium')

def pipeline(config):
    logger = TensorBoardLogger(config['exp']['logdir'], name=config['exp']["exp_name"])
    
    lr_callback = LearningRateMonitor(logging_interval='step')
    ema_callback = EMAWeightAveraging()
    # dm = FakeData_lightning(config=config)
    if config['exp']['training_type'] == 'scratch_UKBB':
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="diffusion-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback, ema_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        dm = UK_biobank_data_module(
            config=config
        )
        DDPM_lightning = LightningDDPM_monai(config=config)
        trainer.fit(model=DDPM_lightning, datamodule=dm)

    elif config['exp']['training_type'] == 'continue_UKBB':
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="diffusion-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback, ema_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        dm = UK_biobank_data_module(
            config=config
        )
        DDPM_lightning = LightningDDPM_monai(config=config)
        trainer.fit(model=DDPM_lightning , datamodule=dm, ckpt_path=config['exp']['model_ckpt_path'])
    elif config['exp']['training_type'] == 'scratch_conditional':
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="diffusion-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback, ema_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        dm = Retinal_Cond_Lightning_Split(
            config=config
        )
        DDIM_lightning = Conditional_DDIM_monai(config=config)
        trainer.fit(model=DDIM_lightning, datamodule=dm)

    elif config['exp']['training_type'] == 'scratch_SLO_conditional':
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="diffusion-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback, ema_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        dm = SLO_Lightning_Split(
            config=config
        )
        DDIM_lightning = Conditional_DDIM_monai(config=config)
        trainer.fit(model=DDIM_lightning, datamodule=dm)

    elif config['exp']['training_type'] == 'scratch_SLO_test':
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="diffusion-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback, ema_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        os.makedirs(config['exp']['csv_dir'],exist_ok=True)
        dm = SLO_Lightning_Split(
            config=config
        )
        dm.setup(stage='test')
        config['exp']['file_path_labels'] = dm.test.samples

        DDIM_lightning = Conditional_DDIM_monai.load_from_checkpoint(config["exp"]["model_ckpt_path"], strict=False, config=config)
        trainer.test(model=DDIM_lightning, dataloaders=dm.test_dataloader())


    elif config['exp']['training_type'] == 'pretrained':
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="diffusion-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback, ema_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        dm = Retinal_Cond_Lightning_Split(
            config=config
        )
        Pretrained_DDPM_lightning = Pretrained_LightningDDPM_monai.load_from_checkpoint(config['exp']['model_ckpt_path'],  config=config, strict=False)
        Pretrained_DDPM_lightning.model.num_class_embeds = config['hparams']['num_classes']
        trainer.fit(model=Pretrained_DDPM_lightning, datamodule=dm)





    elif config['exp']['training_type'] == 'continue_pretrained':
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="diffusion-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback, ema_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        dm = Retinal_Cond_Lightning_Split(
            config=config
        )
        Pretrained_DDPM_lightning = Pretrained_LightningDDPM_monai(config=config)
        trainer.fit(model=Pretrained_DDPM_lightning, datamodule=dm, ckpt_path=config['exp']['model_ckpt_path'])

    elif config['exp']['training_type'] == 'continue_pretrained_classification':
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_compound_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="diffusion-{epoch:02d}-{step}-{val_compound_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback, ema_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        dm = Retinal_Cond_Lightning_Split(
            config=config
        )
        Pretrained_DDPM_lightning = Pretrained_LightningDDPM_monai(config=config)
        trainer.fit(model=Pretrained_DDPM_lightning, datamodule=dm, ckpt_path=config['exp']['model_ckpt_path'])


    elif config['exp']['training_type'] == 'test':
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="diffusion-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        os.makedirs(config['exp']['csv_dir'],exist_ok=True)
        dm = Retinal_Cond_Lightning_Split(
            config=config
        )
        dm.setup(stage='test')
        config['exp']['file_path_labels'] = dm.test.imgs
        Pretrained_DDPM_lightning = Pretrained_LightningDDPM_monai.load_from_checkpoint(config['exp']['model_ckpt_path'], strict=False, config=config)
        trainer.test(model=Pretrained_DDPM_lightning, dataloaders=dm.test_dataloader() )

    elif config['exp']['training_type'] == 'predict':
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="diffusion-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        os.makedirs(config['exp']['csv_dir'],exist_ok=True)
        dm = Retinal_Cond_Lightning_Split(
            config=config
        )
        dm.setup(stage='predict')
        config['exp']['file_path_labels'] = dm.predict_set.file_paths
        Pretrained_DDPM_lightning = Pretrained_LightningDDPM_monai.load_from_checkpoint(config['exp']['model_ckpt_path'], strict=False, config=config)
        trainer.predict(model=Pretrained_DDPM_lightning, dataloaders=dm.predict_dataloader() )
    elif config['exp']['training_type'] == "mlp":
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="mlp-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        simple_mlp = MLP_classifier(config=config)
        dm = Pickle_Lightning(config=config)
        trainer.fit(model=simple_mlp, datamodule=dm)
        trainer.test(model=simple_mlp, datamodule=dm)
    elif config['exp']['training_type'] == "mlp_continue":
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="mlp-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        simple_mlp = MLP_classifier.load_from_checkpoint(config['exp']['model_ckpt_path'], strict=False, config=config)
        dm = Pickle_Lightning(config=config)
        # trainer.fit(model=simple_mlp, datamodule=dm, ckpt_path=config['exp']['model_ckpt_path'])
        trainer.test(model=simple_mlp, datamodule=dm)
    elif config['exp']['training_type'] == "cond_diff_continue":
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="diffusion-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback, ema_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        os.makedirs(config['exp']['csv_dir'],exist_ok=True)
        dm = Retinal_Cond_Lightning_Split(
            config=config
        )
        Pretrained_DDPM_lightning = Pretrained_LightningDDPM_monai(config=config)
        trainer.fit(model=Pretrained_DDPM_lightning, datamodule=dm, ckpt_path=config['exp']['model_ckpt_path'])

    elif config['exp']['training_type'] == "resnet_train":
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="resnet-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        resnet_50 = Restnet_50(config=config)
        dm = Retinal_Cond_Lightning_Split(config=config)
        
        trainer.fit(model=resnet_50, datamodule=dm)
        # trainer.test(model=resnet_50, datamodule=dm)


    elif config['exp']['training_type'] == "resnet_predict":
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="resnet-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        os.makedirs(config['exp']['csv_dir'],exist_ok=True)
        
        dm = Retinal_Cond_Lightning_Split(
            config=config
        )
        dm.setup(stage='predict')
        config['exp']['file_path_labels'] = dm.predict_set.file_paths
        resnet_50 = Restnet_50.load_from_checkpoint(config['exp']['model_ckpt_path'], strict=False, config=config)
        trainer.predict(model=resnet_50, dataloaders=dm.predict_dataloader() )

    elif config['exp']['training_type'] == "resnet_test":
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="resnet-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        os.makedirs(config['exp']['csv_dir'],exist_ok=True)
        
        dm = Retinal_Cond_Lightning_Split(
            config=config
        )
        dm.setup(stage='test')
        config['exp']['file_path_labels'] = dm.test.imgs
        resnet_50 = Restnet_50.load_from_checkpoint(config['exp']['model_ckpt_path'], strict=False, config=config)
        trainer.test(model=resnet_50, dataloaders=dm.test_dataloader() )

    elif config['exp']['training_type'] == "efficient_net_b3_train":
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="efficient_net_b3-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        efficient_net_b3 = EfficientNet_B3(config=config)
        dm = Retinal_Cond_Lightning_Split(config=config)
        
        trainer.fit(model=efficient_net_b3, datamodule=dm)
        # trainer.test(model=efficient_net_b3, datamodule=dm)

    elif config['exp']['training_type'] == "efficient_net_b3_test":
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="efficient_net_b3-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        os.makedirs(config['exp']['csv_dir'],exist_ok=True)
        
        dm = Retinal_Cond_Lightning_Split(
            config=config
        )
        dm.setup(stage='test')
        config['exp']['file_path_labels'] = dm.test.imgs
        efficient_net_b3 = EfficientNet_B3.load_from_checkpoint(config['exp']['model_ckpt_path'], strict=False, config=config)
        trainer.test(model=efficient_net_b3, dataloaders=dm.test_dataloader() )


    elif config['exp']['training_type'] == "efficient_net_b3_predict":
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="efficient_net_b3-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        os.makedirs(config['exp']['csv_dir'],exist_ok=True)
        
        dm = Retinal_Cond_Lightning_Split(
            config=config
        )
        dm.setup(stage='predict')
        config['exp']['file_path_labels'] = dm.predict_set.file_paths
        efficient_net_b3 = EfficientNet_B3.load_from_checkpoint(config['exp']['model_ckpt_path'], strict=False, config=config)
        trainer.predict(model=efficient_net_b3, dataloaders=dm.predict_dataloader() )

    elif config['exp']['training_type'] == "efficient_net_b0_train":
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="efficient_net_b3-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        efficient_net_b3 = EfficientNet_B0(config=config)
        dm = Retinal_Cond_Lightning_Split(config=config)
        
        trainer.fit(model=efficient_net_b3, datamodule=dm)
        # trainer.test(model=efficient_net_b3, datamodule=dm)

    elif config['exp']['training_type'] == "efficient_net_b0_test":
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="efficient_net_b3-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        os.makedirs(config['exp']['csv_dir'],exist_ok=True)
        
        dm = Retinal_Cond_Lightning_Split(
            config=config
        )
        dm.setup(stage='test')
        config['exp']['file_path_labels'] = dm.test.imgs
        efficient_net_b3 = EfficientNet_B0.load_from_checkpoint(config['exp']['model_ckpt_path'], strict=False, config=config)
        trainer.test(model=efficient_net_b3, dataloaders=dm.test_dataloader() )


    elif config['exp']['training_type'] == "efficient_net_b0_predict":
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="efficient_net_b3-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        os.makedirs(config['exp']['csv_dir'],exist_ok=True)
        
        dm = Retinal_Cond_Lightning_Split(
            config=config
        )
        dm.setup(stage='predict')
        config['exp']['file_path_labels'] = dm.predict_set.file_paths
        efficient_net_b3 = EfficientNet_B0.load_from_checkpoint(config['exp']['model_ckpt_path'], strict=False, config=config)
        trainer.predict(model=efficient_net_b3, dataloaders=dm.predict_dataloader() )
        
    elif config['exp']['training_type'] == "swin_b_train":
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="swin_b-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        swin_b = Swin_B(config=config)
        dm = Retinal_Cond_Lightning_Split(config=config)
        
        trainer.fit(model=swin_b, datamodule=dm)


    elif config['exp']['training_type'] == "swin_b_retrain":
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="swin_b-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        swin_b = Swin_B(config=config)
        dm = Retinal_Cond_Lightning_Split(config=config)
        
        trainer.fit(model=swin_b, datamodule=dm, ckpt_path=config['exp']['model_ckpt_path'])



    elif config['exp']['training_type'] == "swin_b_predict":
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="swin_b-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        os.makedirs(config['exp']['csv_dir'],exist_ok=True)
        
        dm = Retinal_Cond_Lightning_Split(
            config=config
        )
        dm.setup(stage='predict')
        config['exp']['file_path_labels'] = dm.predict_set.file_paths
        swin_b = Swin_B.load_from_checkpoint(config['exp']['model_ckpt_path'], strict=False, config=config)
        trainer.predict(model=swin_b, dataloaders=dm.predict_dataloader() )

    elif config['exp']['training_type'] == "swin_b_test":
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(config['exp']['ckpt_dir']),
                                              monitor='val_loss',
                                              verbose=False,
                                              save_last=True,
                                              save_top_k=config['exp']['save_top_k'],
                                              save_weights_only=False,
                                              mode='min',
                                              filename="swin_b-{epoch:02d}-{step}-{val_loss:.5f}"
                                              )
        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=config['exp']['log_every_n_steps'],
            devices=config['exp']['device'],
            min_epochs = config['hparams']['min_epochs'],
            max_epochs = config['hparams']['max_epochs'],
            num_sanity_val_steps=config['hparams']['num_sanity_val_steps'],
            accelerator=config['exp']['accelerator'],
            callbacks=[checkpoint_callback, lr_callback],
            precision='16-mixed',
            # profiler='pytorch',
            accumulate_grad_batches=8
        )
        os.makedirs(config['exp']['csv_dir'],exist_ok=True)
        
        dm = Retinal_Cond_Lightning_Split(
            config=config
        )
        dm.setup(stage='test')
        config['exp']['file_path_labels'] = dm.test.imgs
        swin_b = Swin_B.load_from_checkpoint(config['exp']['model_ckpt_path'], strict=False, config=config)
        trainer.test(model=swin_b, dataloaders=dm.test_dataloader() )



    
    

def main(args):
    config = get_config(args.conf)
    config_str = yaml.dump(config)
    print("Using settings:\n", config_str)
    seed_everything(config["hparams"]["seed"])
    pipeline(config)

if __name__=="__main__":
    parser = ArgumentParser("Driver code.")
    parser.add_argument(
        "--conf", type=str, required=True, help="Path to config.yaml file."
    )
    args = parser.parse_args()

    main(args)



