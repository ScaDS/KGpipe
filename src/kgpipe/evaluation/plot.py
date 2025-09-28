# Plot the evaluation results

import matplotlib.pyplot as plt

def plot_evaluation_results(evaluation_report):
    """Plot the evaluation results."""
    plt.figure(figsize=(10, 6))
    plt.bar(evaluation_report.aspect_results.keys(), evaluation_report.aspect_results.values())
    plt.xlabel("Aspect")
    plt.ylabel("Score")
    plt.title("Evaluation Results")
    plt.show()