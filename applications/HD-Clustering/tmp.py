with open('/home/xrouth2/hdc/HDC-monorepo/applications/HD-Clustering/stats.txt', 'r') as file:
    first_column = []
    third_column = []
    
    for line in file:
        parts = line.split(',')
        first_column.append(float(parts[0]))
        third_column.append(float(parts[2]))
    
    first_average = sum(first_column) / len(first_column)
    third_average = sum(third_column) / len(third_column)
    
    print(f"execution Average: {first_average}")
    print(f"third column Average: {third_average}")