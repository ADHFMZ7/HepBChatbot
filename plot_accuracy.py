import matplotlib.pyplot as plt

# Final accuracies from your evaluation (48/50 = 96% overall)
intents = [
    "greeting", "lab_markers", "prevention", "symptoms",
    "testing", "transmission", "treatment", "urgent",
    "vaccination", "window"
]

final_accuracy = [100, 100, 100, 100, 100, 100, 100, 100, 60, 100]

# Overall accuracy
overall = 96.00

plt.figure(figsize=(12, 6))
plt.bar(intents, final_accuracy)

plt.ylim(0, 110)
plt.ylabel("Accuracy (%)")
plt.title(f"Chatbot Intent Accuracy (Overall: {overall:.2f}%)")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("final_accuracy_bar_graph.png")
print("Saved final_accuracy_bar_graph.png")

