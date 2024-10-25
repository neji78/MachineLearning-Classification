import json
import dataGenerator as dg
import DataAnalysis as da
def run_task(config,data_numbers):
    for head in config:
        label = head["name"]
        mean = head["mean"]
        var = head["var"]
        directory = label+"/count_"+str(data_numbers)
        dataGen = dg.DataGenerator(mean,var,directory)
        dataGen.generate(data_numbers)
        orginalParameter = dict()
        orginalParameter["mean"] = mean
        orginalParameter["covariance matrix"] = var
        da.analysis(directory,orginalParameter)
with open('distribution_config.json', 'r') as file:
    data = json.load(file)
conf = data["data"]
run_task(conf,1000)
run_task(conf,2000)
run_task(conf,5000)
