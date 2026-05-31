from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# LOAD MODEL & SCALER
classifier = joblib.load('kernel_svm.pkl')
scaler = joblib.load('scaler.pkl')

# LOAD DATASET
dataset = pd.read_csv('Social_Network_Ads.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# ORIGINAL DATA
x_original = x
y_original = y

# HOME
@app.route('/')
def home():

    return jsonify({

        'success': True,

        'message':
            'Kernel SVM API Running'
    })

# PREDICT
@app.route('/predict', methods=['POST'])
def predict():

    try:

        data = request.get_json()

        age = float(data['age'])

        estimated_salary = float(
            data['estimated_salary']
        )

        # ORIGINAL INPUT
        original_input = [[
            age,
            estimated_salary
        ]]

        # SCALE INPUT
        scaled_input = scaler.transform(
            original_input
        )

        # PREDICTION
        prediction = classifier.predict(
            scaled_input
        )[0]
        
        # PROBABILITY
        probability = classifier.predict_proba(
            scaled_input
        )[0]

        probability_not_buy = float(probability[0])
        probability_buy = float(probability[1])

        # LABEL
        label = (

            'BUY_SUV'

            if prediction == 1

            else 'NOT_BUY_SUV'
        )

        # DESCRIPTION
        description = (

            'Customer diprediksi akan membeli SUV'

            if prediction == 1

            else
            'Customer diprediksi tidak membeli SUV'
        )

        # DECISION FUNCTION
        decision_score = float(

            classifier.decision_function(
                scaled_input
            )[0]
        )

        # RESPONSE
        response = {

            'success': True,

            'model':
                'Kernel SVM',

            'kernel':
                'linear',

            'prediction':
                int(prediction),

            'label':
                label,

            'description':
                description,

            'input': {

                'age':
                    age,

                'estimated_salary':
                    estimated_salary
            },

            'scaled_input': {

                'scaled_age':
                    float(scaled_input[0][0]),

                'scaled_estimated_salary':
                    float(scaled_input[0][1])
            },
            'probability': {
                'not_buy_suv': probability_not_buy,
                'buy_suv': probability_buy
            },

            'decision_score':
                decision_score,

            'visualization_info': {

                'x_axis':
                    'Age',

                'y_axis':
                    'Estimated Salary',

                'blue_region':
                    'BUY SUV',

                'salmon_region':
                    'NOT BUY SUV',

                'decision_boundary':
                    'Maximum Margin Hyperplane'
            }
        }

        return jsonify(response)

    except Exception as e:

        return jsonify({

            'success': False,

            'error': str(e)

        }), 500

@app.route('/plot-data', methods=['GET'])
def plot_data():

    try:

        # =========================
        # ORIGINAL DATA
        # =========================

        x_set = x_original
        y_set = y_original

        # =========================
        # CREATE GRID
        # =========================

        x1, x2 = np.meshgrid(

            np.arange(
                start=x_set[:, 0].min() - 10,
                stop=x_set[:, 0].max() + 10,
                step=1
            ),

            np.arange(
                start=x_set[:, 1].min() - 1000,
                stop=x_set[:, 1].max() + 1000,
                step=1000
            )
        )

        grid_points = np.array([

            x1.ravel(),
            x2.ravel()

        ]).T

        # =========================
        # SCALE GRID
        # =========================

        scaled_grid = scaler.transform(
            grid_points
        )

        # =========================
        # PREDICTION
        # =========================

        predictions = classifier.predict(
            scaled_grid
        )

        # =========================
        # DECISION FUNCTION
        # =========================

        decision_values = (
            classifier.decision_function(
                scaled_grid
            )
        )

        # =========================
        # SUPPORT VECTORS
        # =========================

        support_vectors = (
            scaler.inverse_transform(
                classifier.support_vectors_
            )
        )

        # =========================
        # PREDICTION REGIONS
        # =========================

        prediction_regions = []

        for i in range(len(grid_points)):

            prediction_regions.append({

                'age':
                    float(grid_points[i][0]),

                'estimated_salary':
                    float(grid_points[i][1]),

                'prediction':
                    int(predictions[i]),

                'region_color': (

                    'dodgerblue'

                    if predictions[i] == 1

                    else 'salmon'
                )
            })

        # =========================
        # CUSTOMER POINTS
        # =========================

        customer_points = []

        for i in range(len(x_set)):

            customer_points.append({

                'age':
                    float(x_set[i][0]),

                'estimated_salary':
                    float(x_set[i][1]),

                'actual_class':
                    int(y_set[i]),

                'point_color': (

                    'dodgerblue'

                    if y_set[i] == 1

                    else 'salmon'
                )
            })

        # =========================
        # SUPPORT VECTOR POINTS
        # =========================

        support_vector_points = []

        for sv in support_vectors:

            support_vector_points.append({

                'age':
                    float(sv[0]),

                'estimated_salary':
                    float(sv[1]),

                'point_color':
                    'yellow',

                'point_type':
                    'support_vector'
            })

        # =========================
        # NON-LINEAR DECISION BOUNDARY
        # =========================

        decision_boundary = []

        threshold = 0.05

        for i in range(len(grid_points)):

            if abs(decision_values[i]) < threshold:

                decision_boundary.append({

                    'age':
                        float(grid_points[i][0]),

                    'estimated_salary':
                        float(grid_points[i][1]),

                    'decision_value':
                        float(decision_values[i])
                })

        # =========================
        # RESPONSE
        # =========================

        response = {

            'success': True,

            'model':
                'Kernel SVM',

            'kernel':
                'rbf',

            'axis': {

                'x_axis':
                    'Age',

                'y_axis':
                    'Estimated Salary'
            },

            'plot_range': {

                'x_min':
                    float(x1.min()),

                'x_max':
                    float(x1.max()),

                'y_min':
                    float(x2.min()),

                'y_max':
                    float(x2.max())
            },

            'customer_points':
                customer_points,

            'prediction_regions':
                prediction_regions,

            'support_vectors':
                support_vector_points,

            'decision_boundary':
                decision_boundary,

            'legend': {

                'salmon':
                    'NOT BUY SUV',

                'dodgerblue':
                    'BUY SUV',

                'yellow':
                    'SUPPORT VECTOR',

                'decision_boundary':
                    'Kernel SVM Decision Boundary'
            }
        }

        return jsonify(response)

    except Exception as e:

        return jsonify({

            'success': False,

            'error': str(e)

        }), 500

# RUN SERVER
if __name__ == '__main__':

    app.run(
        debug=True
    )