# PSO Optimized Naive Bayes for Spam Detection

This repository contains a Python script that leverages Particle Swarm Optimization (PSO) to fine-tune the hyperparameters of a Naive Bayes classifier used for spam email detection. The script demonstrates a practical application of PSO in improving machine learning model performance by optimizing a key hyperparameter (`alpha`) of the Naive Bayes algorithm.

## Features

- **Data Visualization**: Analyze the distribution of spam and non-spam emails in the dataset.
- **Text Preprocessing**: Includes functions to clean and prepare email text data for modeling.
- **Vectorization**: Utilizes `CountVectorizer` to convert text data into a numerical format that machine learning algorithms can process.
- **Hyperparameter Optimization**: Employs PSO to find the optimal `alpha` value for the Naive Bayes classifier, enhancing model accuracy.

## Getting Started

To use this script, clone the repository, install the required libraries listed in `requirements.txt`, and run the script with a dataset named 'emails.csv' in the same directory. The script will preprocess the data, visualize it, train a baseline Naive Bayes model, optimize the model using PSO, and evaluate the performance of both the baseline and optimized models.

## Requirements

- Python 3.x
- pandas
- numpy
- pyswarms
- scikit-learn

Install the dependencies with `pip install -r requirements.txt`.

## Usage

1. Ensure your dataset is named 'emails.csv' and placed in the root directory.
2. Run the script: `python nb_pso.py`
3. View the performance of the baseline and optimized Naive Bayes models in the console output.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to suggest improvements or add new features.

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.
