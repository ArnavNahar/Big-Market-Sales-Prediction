# Big Market Sales Prediction 🏪🛒

This project focuses on predicting sales outcomes based on historical data. Leveraging data collected from Kaggle, I applied Artificial Neural Networks (ANN) to model and predict future sales. The project also includes a comprehensive Exploratory Data Analysis (EDA) and Data Visualization to better understand the data and its trends.

While the current model uses neural networks, you can explore and experiment with other machine learning algorithms to improve prediction accuracy.

### Key Features:
- **Exploratory Data Analysis (EDA):** Gain insights into sales data with detailed analysis.
- **Data Visualization:** Graphical representation of data patterns using Python libraries.
- **Neural Networks (ANN):** Predictive model to forecast sales based on input features.
- **Customizable ANN Parameters:** Experiment with different network architectures and parameters for better performance.

<p align="center">
  <img src="img/market_sales_prediction.png" alt="Market Sales Prediction" width="450"/>
</p>

---

### Dataset

The dataset used for this project is publicly available on Kaggle and contains sales-related features like product type, store information, and sales figures. You can find and download the dataset here.

**Dataset Features:**
- **Item Identifier:** Unique identifier for the product.
- **Item Weight:** Weight of the product.
- **Item Visibility:** The amount of visibility the product gets.
- **Item Type:** Type/category of the product.
- **Outlet Identifier:** Unique identifier for the store/outlet.
- **Outlet Location Type:** Type of location where the store is based (city, tier).
- **Outlet Sales:** The target variable representing sales.

---

### Project Workflow:

1. **Data Preprocessing**
   - **Missing Data Handling:** Fill missing values in essential columns such as Item Weight.
   - **Encoding Categorical Variables:** Use techniques like one-hot encoding to handle categorical variables like Item Type and Outlet Identifier.
   - **Normalization:** Normalize features for improved performance in the neural network.

2. **Exploratory Data Analysis (EDA)**
   - **Trend Analysis:** Discover patterns in sales based on item types, outlet location, and visibility.
   - **Correlation Analysis:** Check relationships between variables and how they affect sales.
   - **Data Visualization:** Visualize sales distributions, sales by store type, and other key insights using libraries like Matplotlib and Seaborn.

   <p align="center">
     <img src="img/sales_trend_analysis.png" alt="Sales Trend Analysis" width="450"/>
   </p>

3. **Model Building: Artificial Neural Network (ANN)**
   - **Model Architecture:** Build a simple ANN using Python's TensorFlow or Keras libraries with an input layer, hidden layers, and output layer.
   - **Activation Functions:** Use ReLU for hidden layers and Linear for the output layer.
   - **Loss Function and Optimizer:** Use Mean Squared Error (MSE) as the loss function and Adam optimizer for training the model.
   - **Training:** Train the model with the processed data and fine-tune the parameters for better accuracy.

4. **Model Evaluation**
   - **Evaluate the performance of the model** using metrics such as Mean Squared Error (MSE) and R-squared.
   - **Analyze the difference between predicted and actual sales.**

5. **Further Improvements**
   - **Experiment with different machine learning algorithms** (e.g., Decision Trees, Random Forests, XGBoost) to enhance the prediction.
   - **Hyperparameter tuning:** Adjust the ANN's architecture (e.g., number of layers, neurons, learning rate) to improve accuracy.

---

### How to Use:

1. **Install required libraries:**  
   Install the necessary dependencies using the `requirements.txt` file.

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the notebook:**
   Use Jupyter Notebook or any Python IDE to open and run the notebook for data exploration and model training.

3. **Experiment:**
   You can modify the ANN architecture and parameters in the `train_model.py` script to try and improve the model's performance.

---

### Technologies Used:
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, TensorFlow/Keras, Matplotlib, Seaborn
- **Platform:** Kaggle (for dataset)

---

### Conclusion

The Big Market Sales Prediction project demonstrates how to use neural networks for forecasting sales, but it also encourages experimentation with other machine learning models and parameters to achieve higher accuracy. By performing thorough EDA and using appropriate features, the project helps in understanding the factors driving sales and makes predictions based on historical data.
