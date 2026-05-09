from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
import sys 

true_labels = []
predicted_labels = []
if len(sys.argv) != 2:
    print("Please give predictions  file as the first argument")
    sys.exit()
prediction_file = sys.argv[1]

if prediction_file is None:
    print("Please give predctions as the first argument")

    # print(f"Calculating score for: {prediction_file}")
# Read the true and predicted labels from file
with open(prediction_file, "r") as f:
    for line in f:
        true_label, predicted_label = map(int, line.strip().split())
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

# Calculate Normalized Mutual Information Score
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)
print(f"{nmi_score}")

print(f"Normalized Mutual Information Score: {nmi_score}")

# Calculate Adjusted Mutual Information Score
ami_score = adjusted_mutual_info_score(true_labels, predicted_labels)
print(f"Adjusted Mutual Information Score: {ami_score}")