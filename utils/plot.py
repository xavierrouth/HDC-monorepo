import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import zlib

# Connect to your SQLite database
db_name = "gpu-2"
conn = sqlite3.connect(f'/home/xrouth2/hdc/HDC-monorepo/approx-tuner/opentuner.db/miranda.cs.illinois.edu.db')  # change to your database path
cursor = conn.cursor()

# Query all records
cursor.execute('SELECT accuracy, time FROM result WHERE tuning_run_id = 50')
rows = cursor.fetchall()

# Dump configurations
configuration_ids = [20, 67, 69, 70, 71, 72, 77, 78, 79, 80, 82, 83, 84, 85, 87, 88, 93, 95, 96, 568];
configuration_ids = []
for id in configuration_ids:
    cursor.execute(f"SELECT data FROM Configuration WHERE id = {id}")
    row = cursor.fetchone()

    if row:
        data_json = row[0]  # this is a JSON string
        decompressed_data = zlib.decompress(data_json)
        config = pickle.loads(decompressed_data)  # now unpickle it!
        print(f'Configuration id: {id} is', config)
        # print(data_json)  # parse into dict
        # print('Configuration:', config)
    else:
        print('No configuration found for:', id)

print('---')




# Separate the data into two lists
accuracies = [row[0] for row in rows]
times = [row[1] for row in rows]

slowest_time = np.max(times)
speedups = slowest_time / times 

times = 1 / np.array(times)

#accuracies.append(.678)
#times.append(40)

# Close the database connection
conn.close()

# Plot accuracy vs time
# plt.ylim(top=50)
# plt.xlim(left=.4, right= .6)

plt.scatter(accuracies, speedups)
plt.xlabel('Accuracy')
plt.ylabel('Speedup')
plt.title('Speedup vs Accuracy')
plt.grid(True)
plt.savefig(f'misc/{db_name}.png')
plt.show()
