import numpy as np
import matplotlib.pyplot as plt
import os

def plot_graph(test_df, lookup_step):
    """
    This function plots true close price along with predicted close price
    with blue and red colors respectively
    """
    plt.plot(test_df[f'true_Adj Close_{lookup_step}'], c='b')
    plt.plot(test_df[f'Adj Close_{lookup_step}'], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()


def get_final_df(model, data, scale, lookup_step):
    """
    This function takes the `model` and `data` dict to
    construct a final dataframe that includes the features along
    with true and predicted prices of the testing dataset
    """
    # if predicted future price is higher than the current,
    # then calculate the true future price minus the current price, to get the buy profit
    buy_profit  = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
    # if the predicted future price is lower than the current price,
    # then subtract the true future price from the current price
    sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0
    X_test = data["X_test"]
    y_test = data["y_test"]
    # perform prediction and get prices
    y_pred = model.predict(X_test)
    if scale:
        y_test = np.squeeze(data["column_scaler"]["Adj Close"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["Adj Close"].inverse_transform(y_pred))
    test_df = data["test_df"]
    # add predicted future prices to the dataframe
    test_df[f"Adj Close_{lookup_step}"] = y_pred
    # add true future prices to the dataframe
    test_df[f"true_Adj Close_{lookup_step}"] = y_test
    # sort the dataframe by date
    test_df.sort_index(inplace=True)
    final_df = test_df
    # add the buy profit column
    final_df["buy_profit"] = list(map(buy_profit,
                                    final_df["Adj Close"],
                                    final_df[f"Adj Close_{lookup_step}"],
                                    final_df[f"true_Adj Close_{lookup_step}"])
                                    # since we don't have profit for last sequence, add 0's
                                    )
    # add the sell profit column
    final_df["sell_profit"] = list(map(sell_profit,
                                    final_df["Adj Close"],
                                    final_df[f"Adj Close_{lookup_step}"],
                                    final_df[f"true_Adj Close_{lookup_step}"])
                                    # since we don't have profit for last sequence, add 0's
                                    )
    return final_df


def predict(model, data, n_steps, scale):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-n_steps:]
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    if scale:
        predicted_price = data["column_scaler"]["Adj Close"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
    return predicted_price

def test_model(data, model, model_name, scale, lookup_step, loss_name, n_steps):
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
    final_df = get_final_df(model, data, scale, lookup_step)
    # predict the future price
    future_price = predict(model, data, n_steps, scale)
    # we calculate the accuracy by counting the number of positive profits
    accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(final_df)
    # calculating total buy & sell profit
    total_buy_profit  = final_df["buy_profit"].sum()
    total_sell_profit = final_df["sell_profit"].sum()
    # total profit by adding sell & buy together
    total_profit = total_buy_profit + total_sell_profit
    # dividing total profit by number of testing samples (number of trades)
    profit_per_trade = total_profit / len(final_df)
    # printing metrics
    print(f"Future price after {lookup_step} days is {future_price:.2f}$")
    print(f"{loss_name} loss:", loss)
    print("Mean Absolute Error:", mean_absolute_error)
    print("Accuracy score:", accuracy_score)
    print("Total buy profit:", total_buy_profit)
    print("Total sell profit:", total_sell_profit)
    print("Total profit:", total_profit)
    print("Profit per trade:", profit_per_trade)
    # plot true/pred prices graph
    plot_graph(final_df, lookup_step)
    print(final_df.tail(10))
    # save the final dataframe to csv-results folder
    csv_results_folder = "csv-results"
    if not os.path.isdir(csv_results_folder):
        os.mkdir(csv_results_folder)
    csv_filename = os.path.join(csv_results_folder, model_name + ".csv")
    final_df.to_csv(csv_filename)
