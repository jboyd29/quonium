# config.py
#

def parse_string(input_string):
    try:
        # Try to parse as an int 
        result = int(input_string)
    except ValueError:
        try:
            # try parsing as a float
            result = float(input_string)
        except ValueError:
            # keep it as a string
            result = input_string
    return result

class config:
    def __init__(self, filename="params"):
        self.data = {}
        self.filename = filename
        self.grabParams()
    def __getitem__(self,key):
        return self.data[key]
    def __setitem__(self,key,value):
        self.data[key] = value
    def __len__(self):
        return len(self.data)
    def __str__(self):
        return str(self.data)
    def __iter__(self):
        return iter(self.data)
    def keys(self):
        return self.data.keys()
    def grabParams(self):
        try:
            with open(self.filename, 'r') as file:
                for line in file:
                    if not line.startswith('#'):
                        items = line.strip().split()
                        if len(items) == 2:
                            key, val = items
                            self.data[key] = parse_string(val)
        except FileNotFoundError:
            print(f"File '{self.filename}' not found.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    def echoParams(self):
        for key in self.data.keys():
            print(key,':',self.data[key])