from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer, RNN, GRU

from data_processing import load_data
from visulization import visualization
from train import create_model, train_model
from test import test_model

# Company name
COMPANY = "TSLA"

# start = '2015-01-01', end='2020-01-01'
TRAIN_START = '2015-01-01'
TRAIN_END = '2020-01-01'

# Window size or the sequence length
N_STEPS = 50
# Lookup step, 1 is the next day
LOOKUP_STEP = 15
STORE = True

# whether to scale feature columns & output price as well
SCALE = True
scale_str = f"sc-{int(SCALE)}"
# whether to shuffle the dataset
SHUFFLE = True
shuffle_str = f"sh-{int(SHUFFLE)}"
# whether to split the training/testing set by date
SPLIT_BY_DATE = True
BREAKPOINT_DATE = "2019-01-01"
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"
# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ['Close','Open','High','Low','Adj Close', 'Volume']
STORE_SCALE = True

### Visulization parameters
TRADING_DAYS = 1

### model parameters

N_LAYERS = 2
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = False

### training parameters

# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 25



### Main code

# load_data function
data_loaded = load_data(COMPANY, TRAIN_START, TRAIN_END, N_STEPS, SCALE, SHUFFLE, STORE, LOOKUP_STEP, SPLIT_BY_DATE, TEST_SIZE, FEATURE_COLUMNS, STORE_SCALE, breakpoint_date=BREAKPOINT_DATE)

# Assign dataframe
data = data_loaded[0]
# Filename
filename = data_loaded[1]

# Visulize candlestick and boxplot
visualization(data['df'], TRADING_DAYS)

# Create model
model = create_model(50, len(data['feature_columns']), GRU)

train = True
# Train model
if train:
    train_model(model, filename, data, 64, 25)

# Test model
test_model(data, model, filename, SCALE, LOOKUP_STEP, LOSS, N_STEPS)

