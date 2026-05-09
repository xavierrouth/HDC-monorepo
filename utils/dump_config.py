import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import zlib
import pprint

# Connect to your SQLite database
conn = sqlite3.connect(f'/home/xrouth2/hdc/HDC-monorepo/approx-tuner/opentuner.db/miranda.cs.illinois.edu.db')  # change to your database path
cursor = conn.cursor()

# Query all records
cursor.execute('SELECT accuracy, time FROM result WHERE tuning_run_id = 50')
rows = cursor.fetchall()

# Dump configurations
# configuration_ids = [i for i in range(964, 1084, 1)]
# configuration_ids = [59605, 59363, 59889, 59717]
configuration_ids = [63216, 63218]
print(configuration_ids)
for id in configuration_ids:
    cursor.execute(f"SELECT data FROM Configuration WHERE id = {id}")
    row = cursor.fetchone()

    if row:
        data_json = row[0]  # this is a JSON string
        # decompressed_data = zlib.decompress(data_json)
        config = pickle.loads(data_json)  # now unpickle it!
        pprint.pprint(f'Configuration id: {id} is {config}\n')
        print(config)
        # print the accuracy and time for this configuration
        cursor.execute(f"SELECT accuracy, time FROM result WHERE configuration_id = {id}")
        result_row = cursor.fetchone()
        if result_row:
            accuracy, time = result_row
            print(f'Accuracy: {accuracy}, Time: {time}\n')  
        # print(data_json)  # parse into dict
        # print('Configuration:', config)
    else:
        print('No configuration found for:', id)

# Dump all configs
# if True:
#     cursor.execute('SELECT data FROM Configuration WHERE tuning_run_id = 20')
#     rows = cursor.fetchall()
print('---')