import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_graph(test_df, k_days):
    """
    This function plots true close price along with predicted close price
    with blue and red colors respectively
    """
    plt.plot(test_df[f'true_Adj Close_{k_days}'], c='b')
    plt.plot(test_df[f'Adj Close_{k_days}'], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()


def get_final_df(model, data, scale, k_days):
    """
    This function takes the `model` and `data` dict to
    construct a final dataframe that includes the features along
    with true and predicted prices of the testing dataset
    """
    X_test = data["X_test"]
    y_test = data["y_test"]
    # perform prediction and get prices
    y_pred = model.predict(X_test)

    if scale:
        y_test = np.squeeze(data["column_scaler"]["Adj Close"].inverse_transform(y_test))
        y_pred = np.squeeze(data["column_scaler"]["Adj Close"].inverse_transform(y_pred))

    test_df = data["test_df"]
    # test_df = add_nan_rows(test_df, k_days-1)
    # add predicted future prices to the dataframe
    for i in range(0,k_days):
        test_df[f"Adj Close_{i+1}"] = y_pred[:,i]
        # add true future prices to the dataframe
        test_df[f"true_Adj Close_{i+1}"] = y_test[:,i]

    # sort the dataframe by date
    test_df.sort_index(inplace=True)

    final_df = test_df
    return final_df

def test_model_LSTM(data, model, model_name, scale, k_days, loss_name, n_steps):
    # load optimal model weights from results folder
    model_path = os.path.join("results", model_name) + ".h5"
    model.load_weights(model_path)

    # evaluate the model
    loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    # calculate the mean absolute error (inverse scaling)
    if scale:
        mean_absolute_error = data["column_scaler"]["Adj Close"].inverse_transform([[mae]])[0][0]
    else:
        mean_absolute_error = mae
    # get the final dataframe for the testing set
    final_df = get_final_df(model, data, scale, k_days)
    # predict the future price
    future_price = predict(model, data, n_steps, scale)
    print(f"Future price after {k_days} days is {future_price:.2f}$")
    print(f"{loss_name} loss:", loss)
    print("Mean Absolute Error:", mean_absolute_error)
    # plot true/pred prices graph
    plot_graph(final_df, k_days)
    # save the final dataframe to csv-results folder
    csv_results_folder = "csv-results"
    if not os.path.isdir(csv_results_folder):
        os.mkdir(csv_results_folder)
    csv_filename = os.path.join(csv_results_folder, model_name + ".csv")
    final_df.to_csv(csv_filename)

    return final_df

def reshape_y(y, k_days):
    new_y = []
    if k_days > 1:
        for i in range(0, len(y)):
            if i == 0:
                for num in y[i]:
                    new_y.append([num])
            else:
                new_y.append([y[i][-1]])
    else:
        new_y = y
    new_y = np.array(new_y)
    return new_y

def add_nan_rows(df, num_rows):
    # Create rows of NaN values for future values
    nan_rows = pd.DataFrame(np.nan, index=pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=num_rows), columns=df.columns)

    # Concatenate nan_rows to the original DataFrame
    df = pd.concat([df, nan_rows])

    return df

def predict(model, data, n_steps, scale):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-n_steps:]
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    if scale:
        predicted_price = data["column_scaler"]["Adj Close"].inverse_transform(prediction)[0][-1]
    else:
        predicted_price = prediction[0][-1]
    return predicted_price