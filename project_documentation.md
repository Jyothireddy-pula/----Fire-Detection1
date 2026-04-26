# Wildfire Risk Prediction System - Documentation

## 1. Training Plan & Architecture
The modeling pipeline is designed to evaluate multiple soft computing approaches and automatically select the best performing model. The training plan follows this progression:
1. **Fuzzy Logic System**: An explicit rule-based system using expert knowledge. It applies fuzzy inference (with trapezoidal membership functions) over 5 key environmental factors.
2. **ANFIS (Adaptive Neuro-Fuzzy Inference System)**: A hybrid neural network and fuzzy logic model that learns membership function parameters and consequent rules from the data. It uses 10 inputs with Gaussian membership functions.
3. **PSO-ANFIS (Particle Swarm Optimization ANFIS)**: To overcome the local minima problems of standard gradient descent in ANFIS, Particle Swarm Optimization (PSO) is used to tune the parameters globally.
**Selection Strategy**: The pipeline trains all three models, evaluates them using a weighted score (RMSE 40%, MAE 30%, Accuracy 30%), and automatically registers the best model to be used for final predictions.

## 2. Feature Extraction
A total of **10 features** are extracted from the dataset to train the models:
1. **Temperature** (°C)
2. **RH** (Relative Humidity %)
3. **Ws** (Wind Speed km/h)
4. **Rain** (Rainfall mm)
5. **FFMC** (Fine Fuel Moisture Code)
6. **DMC** (Duff Moisture Code)
7. **DC** (Drought Code)
8. **ISI** (Initial Spread Index)
9. **BUI** (Buildup Index)
10. **FWI** (Fire Weather Index)

## 3. Data Preprocessing
The preprocessing pipeline applies the following steps before training:
1. **Data Cleaning**: Removal of empty spaces in headers and dropping rows with NaN or invalid string values in numeric columns.
2. **Target Encoding**: Converting the continuous `FWI` index into 5 linguistic multi-class targets: `No Fire`, `Low Fire`, `Medium Fire`, `High Fire`, and `Extreme Fire`.
3. **Outlier Removal**: Using the Interquartile Range (IQR) method (threshold 1.5) to remove extreme statistical outliers from the feature space.
4. **Data Splitting**: Splitting the dataset into 80% training and 20% testing sets.
5. **Normalization**: Applying Min-Max Scaling (`MinMaxScaler`) to map all feature values to a range between 0 and 1, ensuring stable training for the ANFIS neural networks.

## 4. Fuzzy Rules & Memberships Calculation
When calculating how many rules and memberships can be applied, we must consider the algorithm being used:

### ANFIS (Neural-Fuzzy) Calculation
- **Total Input Features**: 10
- **Membership Function**: Gaussian (defined by center and sigma)
- **Memberships per Feature**: 2
- **Maximum Rule Calculation**: Using Grid Partitioning, the number of rules is $M^N$ (where M = memberships, N = features).
- **Current Rules Applied**: $2^{10} = \mathbf{1024 \text{ Rules}}$.
- **Alternative (3 Memberships)**: If you switch to 3 memberships per feature, it would generate $3^{10} = \mathbf{59,049 \text{ Rules}}$, which is computationally expensive.

### Explicit Fuzzy System Calculation
- **Aggregated Inputs**: 5 (Temperature, Humidity, Wind, Rain, Vegetation Dryness)
- **Memberships per Feature**: 3 (e.g., Low, Medium, High) using Trapezoidal shapes.
- **Maximum Possible Rules**: $3^5 = \mathbf{243 \text{ Rules}}$.
- **Current Rules Applied**: The expert system specifically defines **31 critical rules** (weighted) rather than exhaustively computing all 243.

## 5. Knowledge Base Tabular (Fuzzy Rules Table)
Below is the tabular representation of the knowledge base used in the explicit Fuzzy System, mapping the Gaussian/Trapezoidal memberships to the final risk output.

| Rule # | Temperature | Humidity | Wind | Rain | Vegetation (Dryness) | Risk Level Output | Weight |
|--------|-------------|----------|------|------|-----------------------|-------------------|--------|
| 1 | High | Low | High | Light | Dry | Extreme Fire | 1.0 |
| 2 | High | Low | Medium | Light | Dry | Extreme Fire | 0.95 |
| 3 | High | Low | *Any* | Light | Dry | Extreme Fire | 0.9 |
| 4 | High | Medium | High | Light | Dry | High Fire | 0.85 |
| 5 | High | Low | High | Light | Medium | High Fire | 0.85 |
| 6 | High | Low | Low | Light | Medium | High Fire | 0.8 |
| 7 | Medium | Low | High | Light | Dry | High Fire | 0.8 |
| 8 | High | Medium | *Any* | Light | Medium | Medium Fire | 0.65 |
| 9 | Medium | Low | Medium | Light | Medium | Medium Fire | 0.7 |
| 10 | Medium | Medium | High | Light | Dry | Medium Fire | 0.7 |
| 11 | Low | Low | Medium | Light | Dry | Medium Fire | 0.65 |
| 12 | Medium | High | High | Light | Medium | Low Fire | 0.5 |
| 13 | Medium | Medium | Medium | Light | Medium | Low Fire | 0.5 |
| 14 | Low | Low | Low | Light | Dry | Low Fire | 0.5 |
| 15 | High | High | Low | Light | Wet | Low Fire | 0.4 |
| 16 | Low | High | Medium | Medium | Wet | No Fire | 0.3 |
| 17 | Low | Medium | Low | Medium | Medium | No Fire | 0.35 |
| 18 | Medium | High | High | Heavy | Wet | No Fire | 0.2 |
| 19 | High | Medium | High | Heavy | Wet | No Fire | 0.2 |
| 20 | High | Low | Low | Heavy | Wet | No Fire | 0.25 |
*(Note: Asterisks indicate the variable isn't explicitly constrained in the rule condition, triggering across all its memberships).*

## 6. Suggested Additional Evaluation Diagrams
Since you want to add evaluation results to your documentation and Google Colab, here are the diagrams you should generate during training:
1. **Confusion Matrix (Heatmap)**: To show the classification accuracy across the 5 linguistic classes (No Fire vs Extreme Fire).
2. **Training Loss / Convergence Curve**: Specifically for PSO-ANFIS, plotting the best particle fitness score across the 20 iterations.
3. **ROC Curve (One-vs-Rest)**: To evaluate the True Positive Rate vs False Positive Rate for multi-class classification.
4. **Feature Importance (Bar Chart)**: Extracted via Random Forest, showing which of the 10 features influences the FWI/Risk Level the most.
5. **3D Surface Plots (Rule Surface)**: Visualizing the Fuzzy inference space (e.g., mapping Temperature and Humidity on the X/Y axes against Risk Score on the Z axis).
6. **Model Comparison Bar Chart**: Comparing the RMSE and Accuracy percentages of Fuzzy, ANFIS, and PSO-ANFIS side-by-side.
