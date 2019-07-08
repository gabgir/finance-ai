# Author: Victor BG
# Date: June 2nd, 2019

# This script extracts financial data.

# Imports
import numpy as np
import requests
import json
import time
import sys

#----------------#
# TICKER QUERIES #
#----------------#
# The text for my URL request to FRED
# url = "https://api.stlouisfed.org/fred/series/observations?series_id=GOOG&api_key={0}&file_type=json".format(api_key_FRED)
# r = requests.get(url)
# print(r.text)

class FinancialData():
    def __init__(self):
        self.tickers = None

    #--------------------#
    # SAVING AND LOADING #
    #--------------------#
    def load_tickers(self, inpath):
        self.tickers = np.genfromtxt(inpath, dtype=str)

    def load_compiled_data(self, inpath):
        with open(inpath, "r") as f:
            self.compiled = json.load(f)

    #----------------------#
    # PARSING AND QUERYING #
    #----------------------#
    # Tickers
    def parse_for_category(self, inpath, outpath, category_interest):
        # Parsing the infile
        f = open(inpath, "r")
        # category_interest = "Technology"
        list_ticker = []
        ind = 0
        for line in f:
            ind += 1
            line = line.split('","')
            ticker = line[0][1:]
            name = line[1]
            sector = line[6]

            if sector == category_interest:
                # print("Company: {}, ticker: {}".format(name, ticker))
                list_ticker.append(ticker)
        f.close()

        # Writing the outfile
        f = open(outpath, "w")
        for elem in list_ticker:
            f.write(elem+"\n")
        f.close()

        # Returning the list of tickers
        return list_ticker

    def query_alphavantage(self, ticker, key, outfolder):
        url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"\
            "&symbol={0}&outputsize=full&apikey={1}".format(ticker, key)
        try:
            r = requests.get(url)
            with open(outfolder+"/"+ticker+".json", "w") as f:
                json.dump(r.json(), f)

        except:
            print("Could not query the data for ticker: {}".format(ticker))

    def autoquery_alphavantage(self, inpath, outfolder, keypath):
        # The API key for my URL requests to Alphavantage
        with open(keypath) as f:
            key_alphavantage = f.read()

        # Querying alphavantage
        list_ticker = np.genfromtxt(inpath, dtype=str)
        for tick in list_ticker:
            self.query_alphavantage(tick, key_alphavantage, outfolder)
            print("{} done".format(tick))
            time.sleep(12)

    #-------------------#
    # DATA ORGANIZATION #
    #-------------------#
    # Routine to extract the closing values
    def extract_closing_values(self, tick):
        # Getting the data
        with open("data/"+tick+".json") as f:
            data = json.load(f)

        # Extracting the closing values
        time_series = data["Time Series (Daily)"]
        close_value = []
        for item in time_series:
            val = float(time_series[item]["4. close"])
            close_value.append(val)

        # reversing to account for the sorting by alphavantage
        close_value = close_value[::-1]
        return close_value

    # Routine to compile data
    def compile_data(self, inpath, outpath):
        # List of tickers
        tickss = np.genfromtxt(inpath, dtype=str)
        complete_data = {}

        # Creating a dictionary of values
        for tick in tickss:
            try:
                close = self.extract_closing_values(tick)
                complete_data[tick] = close
            except:
                print("Problem loading ticker: {}".format(tick))
        print("Data loaded")

        # Saving to JSON
        with open(outpath, "w") as f:
            json.dump(complete_data, f)
        print("Data saved to disk")

    def create_dataset(self, inpath, outpath, target_yearly_interest):
        # Loading the compiled data
        with open(inpath, "r") as f:
            data = json.load(f)

        # The data
        X = []
        Y = []
        for item in data:
            # Extracting X and Y
            len_year = 365
            len_data = len(data[item])

            if len_data > 2*len_year:
                try:
                    # the data
                    d = data[item]

                    # A random starting point for sampling
                    start_ind = np.random.randint(0,len_data-2*len_year)

                    # The neural net input
                    x = d[start_ind:start_ind+len_year]

                    # The result - value of 1 if we are above the target interest rate.
                    y = d[start_ind+2*len_year-1]/d[start_ind+len_year-1]
                    if y >= target_yearly_interest:
                        y=1.
                    else:
                        y=0.

                    # Appending the result
                    X.append(x)
                    Y.append(y)

                except Exception as err:
                    print("Problem creating a dataset for ticker {}.\nError: {}".format(item, err))

        # Reshaping the arrays - assuming dot(W,X) + b = Y
        X = np.array(X).T
        Y = np.array(Y).reshape(1,-1)
        print("Dataset created with shape: {}".format(X.shape))

        # Normalizing - dividing by the initial value
        X = X / X[0, :]

        # Checking the number of "successes"
        print("The percentage of 'successes': {}".format(np.sum(Y)/len(Y[0])))

        # Saving to disk
        dataset = {
        "X":X.tolist(),
        "Y":Y.tolist()}
        with open(outpath, "w") as f:
            json.dump(dataset, f)

    def create_timeseries(self, tick, length=None):
        data = self.compiled[tick]
        if not length:
            length = len(data)
        return data[-length:]




#------#
# MAIN #
#------#
if __name__=="__main__":
    # Initialization
    fd = FinancialData()

    # Creating the list of tickers
    # list_ticker = fd.parse_for_category(inpath="companylist.csv",
    #     outpath="ticker_technology.csv",
    #     category_interest="Technology")

    # Automatic querying of alphavantage
    # fd.autoquery_alphavantage(inpath="./config/ticker_technology.csv",
    #     outfolder="./data",
    #     keypath="./config/alphavantage.key")

    # Creating the compilation file - takes 5-10 seconds
    # fd.compile_data(inpath="./config/ticker_technology.csv",
    #     outpath="./config/compiled_data.json")

    # Creating a dataset - short time
    fd.create_dataset(inpath="./config/compiled_data.json",
        outpath="./config/dataset.json",
        target_yearly_interest=1.3)
