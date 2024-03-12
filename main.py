# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 20:04:44 2024

@author: anush
"""
import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from train import train_model, test_accuracy


def main(num_samples =  10, max_num_epochs=10, gpus_per_trial=2):

    config = {
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "l1": tune.grid_search([4, 8, 16, 64]),
        "epochs": tune.choice([100, 200, 250, 300]),
        "optimizer": tune.choice(["Adam", "AdamW","SGD","RMSprop"]),
        "weight_decay": tune.loguniform(1e-5, 1e-3),  # Added weight_decay,
        "activation_function": tune.choice(["ReLU", "LeakyReLU", "PReLU"]),
        "should_checkpoint": True
    }
    
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    
    # optuna_search = OptunaSearch(metric = 'loss', mode = 'min')
    
    # data_splits, metadata_splits = get_data(data_dir)
    
    # # Access the train, validation, and test splits
    # X_train_df, y_train_df = data_splits['train']
    # X_val_df, y_val_df = data_splits['val']
    # X_test_df, y_test_df = data_splits['test']
    
 
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model),
            resources={"cpu": 3, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            # search_alg = optuna_search,
            num_samples=num_samples,
        ),
        
        param_space=config,
        
        run_config=train.RunConfig(
            name="pm25_exp7",
            stop={"training_iteration": 100},
           checkpoint_config=train.CheckpointConfig(
               checkpoint_score_attribute="loss",
               num_to_keep=10,
           ),
            storage_path="/ray_results/logs",
            log_to_file=True
        ),)
    
    ray.shutdown()
    ray.init(dashboard_port=8265, dashboard_host='127.0.0.1', include_dashboard=True)
    
    results = tuner.fit()

    best_result = results.get_best_result("loss", "min")
    print('Best parameters...')
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_result.metrics["loss"]))
    print("Best trial final validation MSE: {}".format(best_result.metrics["mse"]))
    print("Best trial final validation RMSE: {}".format(best_result.metrics["rmse"]))
    print("Best trial final validation R2-score: {}".format(best_result.metrics["r2score"]))

    # best_trained_model = LSTMNet(best_trial.config["l1"], best_trial.config["l2"])
    test_loss, test_mse, test_rmse, test_r2score = test_accuracy(best_result)
    
    print("Best trial test set loss: {}".format(test_loss))
    print("Best trial test set mse: {}".format(test_mse))
    print("Best trial test set rmse: {}".format(test_rmse))
    print("Best trial test set r2score: {}".format(test_r2score))
    

main(num_samples = 5, max_num_epochs=25, gpus_per_trial=0)