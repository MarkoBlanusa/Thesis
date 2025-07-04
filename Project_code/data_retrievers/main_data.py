import logging
from binance import BinanceClient
from data_collector import collect_all

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s %(levelname)s :: %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler("info.log")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)

if __name__ == "__main__":
    exchange = "binance"  # Set your desired exchange here

    client = BinanceClient(True)  # Initialize the Binance client

    all_symbols = False  # Set this to True to collect data for all symbols, False for a specific symbol
    symbols = [
        # "ETHUSDT",
        "BTCUSDT",
        # "BNBUSDT",
        # "NEOUSDT",
        # "LTCUSDT",
        # "ADAUSDT",
        # "XRPUSDT",
        # "XLMUSDT",
        # "TRXUSDT",
        # "DOGEUSDT",
        # "BCHUSDT",
        # "ETCUSDT",
        # "LINKUSDT",
        # "XMRUSDT",
        # "DASHUSDT",
        # "ZECUSDT",
        # "XTZUSDT",
        # "ATOMUSDT",
        # "ONTUSDT",
        # "IOTAUSDT",
    ]  # Set your desired symbol here if not collecting for all symbols

    if all_symbols:
        for n_symbol in client.symbols:
            try:
                collect_all(client, exchange, n_symbol)
            except TypeError:
                continue
    else:
        for symbol in symbols:
            try:
                collect_all(client, exchange, symbol)
            except TypeError:
                continue
