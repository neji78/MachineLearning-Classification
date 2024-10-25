import json

class Json:
    def __init__(self,name):
        self.fileName = name
        self.dictionary = dict()
    def add(self,key,value):
        self.dictionary[key] = value
    def save(self):
        json_object = json.dumps(self.dictionary, indent=4)
        with open(self.fileName + ".json", "w") as outfile:
            outfile.write(json_object)
 