import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return np.array(data['trajectory'])

# Convert trajectory to steps (+1/-1)
def convert_to_steps(trajectory):
    return np.sign(np.diff(trajectory))

# Divide data into 100-step walks
def divide_into_walks(data, walk_length=100):
    return [data[i:i + walk_length] for i in range(0, len(data), walk_length) if len(data[i:i + walk_length]) == walk_length]

# Calculate run lengths within a walk
def calculate_run_lengths(walk):
    run_lengths = []
    current_run = 1
    for i in range(1, len(walk)):
        if walk[i] == walk[i - 1]:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)
    return run_lengths

# Perform bootstrapping
def bootstrap_combined(run_counts, run_lengths, n_iterations=10000):
    bootstrap_means = []
    bootstrap_vars = []
    for _ in range(n_iterations):
        # Draw a random number of runs
        sampled_run_count = np.random.choice(run_counts)
        # Draw sampled_run_count run lengths
        sampled_run_lengths = np.random.choice(run_lengths, size=sampled_run_count, replace=True)
        bootstrap_means.append(np.mean(sampled_run_lengths))
        bootstrap_vars.append(np.var(sampled_run_lengths))
    return np.mean(bootstrap_means), np.mean(bootstrap_vars)

def classify_walk(run_lengths, human_mean, human_stddev, computer_mean, computer_stddev, epsilon=1e-10):
    """
    Classify a given walk as human or computer-generated using joint probability calculations.
    Includes a tie-breaker for cases where probabilities are equal.
    """
    print(f"Run Lengths of Given Walk: {run_lengths}")

    # Compute log probabilities for human and computer models
    log_prob_human = 0
    log_prob_computer = 0

    for run_length in run_lengths:
        # Avoid probabilities converging to zero
        prob_human = max(norm.pdf(run_length, loc=human_mean, scale=human_stddev), epsilon)
        prob_computer = max(norm.pdf(run_length, loc=computer_mean, scale=computer_stddev), epsilon)

        # Sum log probabilities
        log_prob_human += np.log(prob_human)
        log_prob_computer += np.log(prob_computer)

    print(f"Log Joint Probability (Human): {log_prob_human:.6f}")
    print(f"Log Joint Probability (Computer): {log_prob_computer:.6f}")

    # Compute likelihood ratio
    log_likelihood_ratio = log_prob_human - log_prob_computer
    print(f"Log-Likelihood Ratio: {log_likelihood_ratio:.6f}")

    # Decision with tie-breaker
    if log_likelihood_ratio > 0:
        print("Decision: The walk is more likely HUMAN.")
        return "HUMAN"
    elif log_likelihood_ratio < 0:
        print("Decision: The walk is more likely COMPUTER.")
        return "COMPUTER"
    else:
        print("Decision: Tie detected. Defaulting to HUMAN.")
        return "HUMAN"  # Default classification


# Main analysis function
def main_with_refined_stats():
    # Load the JSON files
    file1_path = 'run_data_1732880428240.json'
    file2_path = 'run_data_2.json'

    trajectory1 = load_json_data(file1_path)
    trajectory2 = load_json_data(file2_path)

    # Convert to step increments
    steps1 = convert_to_steps(trajectory1)
    steps2 = convert_to_steps(trajectory2)

    # Combine both datasets
    combined_steps = np.concatenate((steps1, steps2))

    # Divide into 100-step walks
    walks = divide_into_walks(combined_steps)

    # Gather all run lengths and run counts
    run_counts = []
    all_run_lengths = []
    for walk in walks:
        run_lengths = calculate_run_lengths(walk)
        run_counts.append(len(run_lengths))
        all_run_lengths.extend(run_lengths)

    # Plot histogram of run counts
    plt.figure(figsize=(12, 6))
    plt.hist(run_counts, bins=range(min(run_counts), max(run_counts) + 2), density=True, alpha=0.7, color='orange', edgecolor='black')
    plt.title("Histogram of Number of Runs In 100-Step Walks", fontsize=16)
    plt.xlabel("Number of Runs", fontsize=12)
    plt.ylabel("Density", fontsize=12)

    # Set custom x-axis ticks at intervals of 10
    tick_range = np.arange(min(run_counts), max(run_counts) + 1, 10)  # Tick every 10
    plt.xticks(tick_range, fontsize=10)
    plt.grid(alpha=0.4)
    plt.show()

    # Plot histogram of all run lengths
    plt.figure(figsize=(12, 6))
    plt.hist(all_run_lengths, bins=range(1, max(all_run_lengths) + 2), density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.title("Histogram of Run Lengths In 100-Step Walks", fontsize=16)
    plt.xlabel("Run Length", fontsize=12)
    plt.ylabel("Density", fontsize=12)

    # Set custom x-axis ticks at intervals of 10
    tick_range = np.arange(1, max(all_run_lengths) + 1, 10)  # Tick every 10
    plt.xticks(tick_range, fontsize=10, rotation=45)  # Rotate labels if needed for clarity
    plt.grid(alpha=0.4)
    plt.show()

    # Perform bootstrapping
    mean_run, var_run = bootstrap_combined(run_counts, all_run_lengths, n_iterations=100000)

    # Plot the normal distribution from bootstrapping results
    x = np.linspace(min(all_run_lengths), max(all_run_lengths), 1000)
    normal_dist = norm.pdf(x, loc=mean_run, scale=np.sqrt(var_run / np.mean(run_counts)))

    plt.figure(figsize=(12, 6))
    plt.plot(x, normal_dist, label="Normal Distribution (CLT)", color='red')
    plt.title("CLT Approximation for Human-Generated Walks", fontsize=16)
    plt.xlabel("Mean Run Length In 100-Step Walks", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.text(0.7 * max(all_run_lengths), 0.5 * max(normal_dist), f"Bootstrapped Mean: {mean_run:.2f}\nBootstrapped Variance: {var_run:.2f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.legend()
    plt.grid(alpha=0.4)
    plt.show()

    print(f"Bootstrapped Mean: {mean_run:.2f}, Bootstrapped Variance: {var_run:.2f}, Bootstrapped Denominator: {np.mean(run_counts):.2f} ")

     # Parameters for Geo(0.5)
    mu_geo = 2  # Mean of geometric distribution
    var_geo = 2  # Variance of geometric distribution
    expected_runs = 50  # Expected number of runs in a 100-step walk

    # CLT for sample mean (Computer)
    mean_clt_computer = mu_geo
    var_clt_computer = var_geo / expected_runs

    # Plotting the CLT-based normal distribution for computer
    x_computer = np.linspace(1, 3, 1000)  # Range for run lengths
    normal_dist_computer = norm.pdf(x_computer, loc=mean_clt_computer, scale=np.sqrt(var_clt_computer))

    plt.figure(figsize=(12, 6))
    plt.plot(x_computer, normal_dist_computer, label="Computer Walk Normal Distribution (CLT)", color='green')
    plt.title("CLT Approximation for Computer-Generated Walks (Geo(0.5))", fontsize=16)
    plt.xlabel("Mean Run Length In 100-Step Walks", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    #plt.text(2.2, max(normal_dist_computer) * 0.8, f"Mean: {mean_clt_computer:.2f}\nVariance: {var_clt_computer:.2f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # Loop through all walks in the dataset for classification
    num_human_classified = 0
    num_computer_classified = 0

    for i, walk in enumerate(walks):
        run_lengths = calculate_run_lengths(walk)

        print(f"\nClassifying Walk {i + 1}:")
        decision = classify_walk(
            run_lengths,
            human_mean=mean_run,
            human_stddev=np.sqrt(var_run / np.mean(run_counts)),
            computer_mean=mean_clt_computer,
            computer_stddev=np.sqrt(var_clt_computer)
        )

        # Update counts based on decision
        if decision == "HUMAN":
            num_human_classified += 1
        else:
            num_computer_classified += 1

    # Output summary of classifications
    print("\nClassification Results:")

    print(f"Bootstrapped Mean: {mean_run:.2f}, Bootstrapped Variance: {var_run:.2f}, Bootstrapped Denominator: {np.mean(run_counts):.2f} ")

    print(f"Total Walks: {len(walks)}")
    print(f"Classified as Human: {num_human_classified}")
    print(f"Classified as Computer: {num_computer_classified}")
    print(f"Accuracy of Human Classification: {num_human_classified / len(walks):.2%}")
    print(f"Accuracy of Computer Classification: {num_computer_classified / len(walks):.2%}")


main_with_refined_stats()

from scipy.stats import geom

bucket_ranges = [
    (1, 1),   # Single steps
    (2, 2),   # Two-step runs
    (3, 3),   # Three-step runs
    (4, 5),   # Short runs
    (6, 7),   # Medium-short runs
    (8, 10),  # Medium runs
    (11, 15), # Long runs
    (16, 20), # Longer runs
    (21, 30), # Very long runs
    (31, 50), # Extremely long runs
    (51, 100) # Rarely seen long runs
]

#def count_consecutive_runs(run_lengths, length):
   # """Count consecutive occurrences of a specific run length."""
#   return sum(1 for run in run_lengths if run == length)

def classify_walk_with_bucketed_pmf(run_lengths, bucketed_human_pmf, geo_p=0.5, epsilon=1e-10):
    log_prob_human = 0
    log_prob_computer = 0

    # Add penalties for excessive short runs (e.g., runs of length 1)
    #penalty = -0.1 * count_consecutive_runs(run_lengths, length=1)

    for run_length in run_lengths:
        prob_human = epsilon

        # Handle both single-value and range buckets
        for bucket, prob in bucketed_human_pmf.items():
            if "-" in bucket:  # Handle range buckets
                start, end = map(int, bucket.split('-'))
                if start <= run_length <= end:
                    prob_human = max(prob, epsilon)
                    break
            else:  # Handle single-value buckets
                start = int(bucket)
                if run_length == start:
                    prob_human = max(prob, epsilon)
                    break

        prob_computer = max(geom.pmf(run_length, geo_p), epsilon)
        log_prob_human += np.log(prob_human)
        log_prob_computer += np.log(prob_computer)

    # Apply penalty
    #log_prob_human += penalty

    log_likelihood_ratio = log_prob_human - log_prob_computer
    decision = "HUMAN" if log_likelihood_ratio > 0 else "COMPUTER"

    return decision, log_prob_human, log_prob_computer, log_likelihood_ratio


def export_bucketed_pmf(bucketed_pmf, output_file="bucketed_pmf.json"):
    """
    Save the bucketed PMF to a JSON file.
    """
    with open(output_file, 'w') as file:
        json.dump(bucketed_pmf, file, indent=4)
    print(f"Bucketed PMF exported to {output_file}")


def bucket_pmf(non_bucketed_pmf, bucket_ranges):
    """
    Aggregate non-bucketed PMF values into manually defined bucket ranges.
    """
    bucketed_pmf = {}
    for start, end in bucket_ranges:
        prob_sum = sum(
            prob for length, prob in non_bucketed_pmf.items() if start <= length <= end
        )
        bucket_label = f"{start}-{end}" if start != end else f"{start}"
        bucketed_pmf[bucket_label] = prob_sum
    return bucketed_pmf


def main_with_manual_buckets():
    # Load the JSON files
    file1_path = 'run_data_1732880428240.json'
    file2_path = 'run_data_2.json'

    trajectory1 = load_json_data(file1_path)
    trajectory2 = load_json_data(file2_path)

    # Convert to step increments
    steps1 = convert_to_steps(trajectory1)
    steps2 = convert_to_steps(trajectory2)

    # Combine both datasets
    combined_steps = np.concatenate((steps1, steps2))

    # Divide into 100-step walks
    walks = divide_into_walks(combined_steps)

    # Gather all run lengths
    all_run_lengths = []
    for walk in walks:
        run_lengths = calculate_run_lengths(walk)
        all_run_lengths.extend(run_lengths)

    # Calculate non-bucketed PMF
    non_bucketed_pmf = {
        length: all_run_lengths.count(length) / len(all_run_lengths)
        for length in range(1, max(all_run_lengths) + 1)
    }

    # Create bucketed PMF based on manual ranges
    bucketed_pmf = bucket_pmf(non_bucketed_pmf, bucket_ranges)

    # Export the bucketed PMF for review
    export_bucketed_pmf(bucketed_pmf, output_file="manual_bucketed_pmf.json")

    # Visualize bucketed PMF
    x_labels = list(bucketed_pmf.keys())
    y_values = list(bucketed_pmf.values())

    plt.figure(figsize=(12, 6))
    plt.bar(x_labels, y_values, alpha=0.7, color='blue', edgecolor='black')
    plt.title("Manual Bucketed PMF of Run Lengths for Human-Generated 100-Step Walks", fontsize=16)
    plt.xlabel("Run Length Bucket", fontsize=12)
    plt.ylabel("Probability", fontsize=12)

    # Customize x-axis ticks
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45, fontsize=10)
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()

    # Classify walks using the manual bucketed PMF
    num_human_classified = 0
    num_computer_classified = 0

    for i, walk in enumerate(walks):
        run_lengths = calculate_run_lengths(walk)

        # Classify the walk and retrieve log probabilities
        decision, log_prob_human, log_prob_computer, log_likelihood_ratio = classify_walk_with_bucketed_pmf(run_lengths, bucketed_pmf)

        # Update classification counts
        if decision == "HUMAN":
            num_human_classified += 1
        else:
            num_computer_classified += 1

        # Debug misclassified walks
        true_label = "HUMAN"  # Replace with actual label if available
        if decision != true_label:
            print(f"Misclassified Walk Index: {i + 1}")
            print(f"Run Lengths: {run_lengths}")
            print(f"Decision: {decision}, True Label: {true_label}")
            print(f"Log Joint Probability (Human): {log_prob_human:.6f}")
            print(f"Log Joint Probability (Computer): {log_prob_computer:.6f}")
            print(f"Log-Likelihood Ratio: {log_likelihood_ratio:.6f}")
            print("")


    # Output summary of classifications
    print("\nClassification Results with Manual Buckets:")
    print(f"Total Walks: {len(walks)}")
    print(f"Classified as Human: {num_human_classified}")
    print(f"Classified as Computer: {num_computer_classified}")
    print(f"Accuracy of Human Classification: {num_human_classified / len(walks):.2%}")
    print(f"Accuracy of Computer Classification: {num_computer_classified / len(walks):.2%}")


# Run the main function with manual bucketing
main_with_manual_buckets()
