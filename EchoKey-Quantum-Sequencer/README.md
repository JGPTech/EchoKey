# Quantum-Classical Hybrid Sequencer with EchoKey Integration

This project combines quantum computing with classical machine learning to create a sophisticated sequencer capable of predicting and extending multi-dimensional fractal sequences. The integration of the EchoKey system enhances performance by introducing entropy injection, synergy measurements, and refraction effects.

## Features

* **Quantum Base-10 Encoding:** Efficiently encodes base-10 digits (0-9) using a quantum system.
* **Machine Learning Integration:** Employs Random Forest and LSTM networks to learn and predict sequence patterns.
* **EchoKey System Integration:**  Enhances the sequencer with:
    * Cyclicity
    * Fractality
    * Entropy injection
    * Synergy measurements
* **Refraction Effects:** Adjusts measurement probabilities based on fractal layers and synergy parameters.
* **Extensible and Configurable:**  Easy customization of critical parameters.
* **Visualization:**  Plots predicted digits for analysis.

## EchoKey System Integration

The EchoKey system is a novel framework that enhances computational models by introducing cyclicity, fractality, and entropy injection. In this sequencer, EchoKey components improve prediction accuracy and system robustness.

### EchoKey Components

* **RollingWindow:**
    * Manages a fixed-size rolling window to track recent state values.
    * Used in calculating synergy parameters.
* **KeystreamScrambler:**
    * Injects entropy into measurement probabilities.
    * Enhances quantum state measurements by adding controlled randomness.

### Synergy Measurements

Synergy measurements capture interdependencies within the system's states. The sequencer calculates synergy parameters (alpha, beta, gamma) using the `RollingWindow`:

* **Alpha (α):** Mean of neighboring state probabilities.
* **Beta (β):** Standard deviation of neighboring state probabilities.
* **Gamma (γ):** Minimum of neighboring state probabilities.

These parameters inform refraction effects and enhance sequence prediction.

### Refraction Effect

The refraction effect dynamically adjusts measurement probabilities based on fractal complexity and recent state interactions.

## Architecture

The sequencer operates through an iterative process:

1.  **Data Initialization:** Reads the target sequence and assigns fractal layers.
2.  **Quantum State Preparation:** Encodes digits into quantum states and injects entropy.
3.  **Synergy Calculations:** Calculates synergy parameters using the `RollingWindow`.
4.  **Refraction Application:** Adjusts measurement probabilities.
5.  **Machine Learning Integration:**  
    * Random Forest classifies digits.
    * LSTM learns temporal patterns.
6.  **Prediction and Extension:** Predicts and extends the sequence.
7.  **Data Logging and Visualization:** Logs data and visualizes predicted digits.

## Installation

1.  Ensure you have Python 3.7 or higher.
2.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/quantum-classical-hybrid-sequencer.git](https://github.com/yourusername/quantum-classical-hybrid-sequencer.git)
    cd quantum-classical-hybrid-sequencer
    ```
3.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
4.  Install dependencies:
    ```bash
    pip install -r requirements.txt 
    ```
    or
    ```bash
    pip install numpy pandas pennylane scikit-learn tensorflow matplotlib
    ```

## Usage

### Configuration

Adjust the parameters at the top of the `sequencer.py` script as needed.

### Running the Sequencer

1.  Place your target CSV file (e.g., `recaman_puzzle.csv`) in the working directory.
2.  Execute the script:
    ```bash
    python sequencer.py
    ```

## Output

* **CSV Files:** Detailed logs of each run are saved in the `simulation_results` directory.
* **Plots:** Visualizations of predicted digits are displayed.

## Project Structure

```
quantum-classical-hybrid-sequencer/
│
├── sequencer.py                  # Main script
├── recaman_puzzle.csv            # Sample CSV file
├── simulation_results/           # Output directory
│   ├── Run_1_Data_YYYYMMDD_HHMMSS.csv
│   └── Run_1_Extended_Data_YYYYMMDD_HHMMSS.csv
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Dependencies

* NumPy
* pandas
* PennyLane
* scikit-learn
* TensorFlow
* Matplotlib
* psutil (optional)

## Results

The sequencer outputs progress updates, final predictions, CSV logs, and visual plots for analysis.

## Troubleshooting

Refer to the README for troubleshooting tips related to syntax errors, CSV file issues, dependencies, and quantum simulation errors.

## Contributing

Contributions are welcome! Fork the repository, create a new branch, commit your changes, and open a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgements

* PennyLane
* scikit-learn
* TensorFlow
* EchoKey Framework
