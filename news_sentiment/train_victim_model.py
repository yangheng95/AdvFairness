

import pyabsa.tasks.FairnessAttackClassification as TC

config = TC.TCConfigManager.get_tc_config_english()
config.pretrained_bert = 'bert-base-uncased'
config.num_epoch = 5
config.evaluate_begin = 0
config.log_step = -1
config.max_seq_len = 80
config.dropout = 0.1
config.verbose = True
config.cache_dataset = False

trainer = TC.TCTrainer(config=config,
                       dataset='news_sentiment',
                       checkpoint_save_mode=1,
                       auto_device=True
                       )

victim_model = trainer.load_trained_model()
