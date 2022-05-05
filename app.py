import pandas as pd
import sys
import my_functions as mf
import matplotlib as mpl

# no displaying on screen, only saving the plots to file
mpl.use('Agg')

# Scaling
dataset_level = "DL"
window_level = "WL"

# Hyperparameters
sliding_windows = [8, 16, 32]
epochs = [50, 100]
batches = [60, 1440, 10080]

x_train, x_test, y_train, y_test = mf.preprocess_and_split_data()



#sliding_windows = [8]  # 10, 16
#epochs = [10]
#batches = [64]



def run(model_number="1"):
    model_number = sys.argv[1]

    path, pred_path = mf.create_path(model_number)

    # Evaluate with dataset level scaling
    dl_scores, dl_predictions, dl_duration_time = mf.evaluate_model(sliding_windows, epochs, batches, model_number=model_number, data_scaling=dataset_level,
                                                                 path=path)

    # Save results to file
    mf.save_results(dl_scores, dl_predictions, dl_duration_time, path, data_scaling=dataset_level)

    # Plot predictions and save to file
    mf.plot_predictions(dl_predictions, y_test, pred_path, dataset_level)

    # Evaluate with window level scaling
    wl_scores, wl_predictions, wl_duration_time = mf.evaluate_model(sliding_windows, epochs, batches, model_number=model_number, data_scaling=window_level,
                                                                 path=path)
    # Save results to file
    mf.save_results(wl_scores, wl_predictions, wl_duration_time, path, data_scaling=window_level)

    # Plot predictions and save to file
    mf.plot_predictions(wl_predictions, y_test, pred_path, window_level)

    # Table for result comparison
    df_compare = pd.merge(dl_scores, wl_scores, on=['SW', 'Epoch', 'Batches'], how="left", suffixes=("_DL", "_WL"))
    df_compare = df_compare.reindex(
        columns=['SW', 'Epoch', 'Batches', 'RMSE_DL', 'RMSE_WL'])

    # Save results to file
    df_compare.to_csv(path + "/overall_score_comparison.csv")


run()


# Plot history and future
# mf.plot_multistep(y_train, y_pred, y_test[:len_])
