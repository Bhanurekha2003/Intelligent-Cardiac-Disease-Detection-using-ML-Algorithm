from flask import Flask, request, jsonify
import mysql.connector
from flask_cors import CORS
import joblib
import pandas as pd
import traceback  # For debugging
import sys
sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)

con=mysql.connector.connect(
  host='localhost',
  user='root',
  password='root',
  database='HEARTDISEASE'
)


@app.route('/getTable',methods=['GET'])
def get_tables():
  cursor=con.cursor()
  cursor.execute("SHOW TABLES;")
  tables=cursor.fetchall()
  cursor.close()
  print(tables)
  table_names=[table[0] for table in tables]
  return jsonify({"tables":table_names}),200

@app.route("/register", methods=["POST"])
def register_user():
    try:
        data = request.json
        name = data.get("name")
        username = data.get("username")
        phone = data.get("phone")
        email = data.get("email")
        place = data.get("place")
        password = data.get("password")  # In real-world apps, hash passwords before storing

        if not name or not username or not phone or not password:
            return jsonify({"error": "Missing required fields"}), 400

        cursor = con.cursor()
        cursor.execute(
            "INSERT INTO users (name, username, phone, email, place, password) VALUES (%s, %s, %s, %s, %s, %s)",
            (name, username, phone, email, place, password)
        )
        con.commit()
        cursor.close()
        
        return jsonify({"message": "User registered successfully"}), 201
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {err}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/login", methods=["POST"])
def login_user():
    try:
        data = request.json
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify({"error": "Missing username or password"}), 400

        cursor = con.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cursor.fetchone()
        cursor.close()

        if user:
            return jsonify({"message": "Login successful", "user": user}), 200
        else:
            return jsonify({"error": "Invalid username or password"}), 401

    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {err}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500



CORS(app)


# Load models, encoders, scalers, and feature importance
models = {
    "model_1": joblib.load("model_1.pkl"),
    "model_2": joblib.load("model_2.pkl")
}
encoders = {
    "model_1": joblib.load("model_1_encoders.pkl"),
    "model_2": joblib.load("model_2_encoders.pkl")
}
scalers = {
    "model_1": joblib.load("model_1_scaler.pkl"),
    "model_2": joblib.load("model_2_scaler.pkl")
}
feature_importances = {
    "model_1": joblib.load("model_1_features.pkl"),
    "model_2": joblib.load("model_2_features.pkl")
}

# Function to generate Risk Shield suggestions
def generate_risk_shield(top_feature):
    suggestions = {
        "RestingBP": "Try to maintain BP under 120 mmHg. Reduce salt and stress.",
        "Cholesterol": "Avoid high-fat foods. Eat oats, nuts, and fiber-rich meals.",
        "MaxHR": "Improve heart endurance with regular walking or cardio workouts.",
        "Oldpeak": "Get stress ECG tests. Avoid high physical or emotional stress.",
        "Age_In_Days": "Schedule regular heart checkups and maintain an active life.",
        "ExerciseAngina": "Avoid high exertion. Seek evaluation for chest discomfort.",
        "ChestPainType": "Asymptomatic patients should still monitor cardiovascular health.",
        "ST_Slope": "Flat ST slope may need further evaluation for ischemia.",
        "Sex": "Men are at higher risk. Lifestyle changes can significantly help.",
        "FastingBS": "Control blood sugar with a low-glycemic diet and meds if needed.",
        "BMI": "Reduce weight with balanced diet and consistent workouts.",
        "Smoking": "Quit smoking completely. Support groups and nicotine therapy help.",
        "Alcohol_Consumption": "Limit intake to 1 drink/day (women) or 2 (men) max.",
        "Physical_Activity_Level": "Walk daily and build active routines."
    }
    return suggestions.get(top_feature, "Maintain a heart-healthy lifestyle with balanced diet and regular exercise.")

# Function to preprocess user input
def preprocess_input(data, model_key):
    try:
        df = pd.DataFrame([data])
        print("\n📩 Raw Input Data:\n", df)

        # Expected features, excluding 'HeartDisease' (target variable)
        expected_features = [col for col in scalers[model_key].feature_names_in_ if col != "HeartDisease"]
        received_features = df.columns.tolist()

        # Check for missing features
        missing_features = [col for col in expected_features if col not in received_features]
        extra_features = [col for col in received_features if col not in expected_features]

        print("✅ Expected Features (without target):", expected_features)
        print("🔍 Received Features:", received_features)

        if missing_features:
            print(f"❌ Missing features: {missing_features}")
            return None

        if extra_features:
            print(f"⚠ Extra features in input: {extra_features}. These will be ignored.")

        # Drop extra features that are not needed
        df = df[expected_features]

        # Apply label encoding for categorical variables
        for col, encoder in encoders[model_key].items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except Exception as e:
                    print(f"❌ Encoding error for column {col}: {e}")
                    return None

        # Apply scaling for numerical variables
        numeric_cols = scalers[model_key].feature_names_in_
        try:
            df[numeric_cols] = scalers[model_key].transform(df[numeric_cols])
        except Exception as e:
            print(f"❌ Scaling error: {e}")
            return None

        print("🔄 Processed Data for Prediction:\n", df)
        return df

    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        traceback.print_exc()
        return None

@app.route("/", methods=["GET"])
def home():
    return "Flask API is Running! Use /predict_model1 or /predict_model2 for predictions."

@app.route("/predict_model1", methods=["POST"])
def predict_model1():
    return make_prediction("model_1")

@app.route("/predict_model2", methods=["POST"])
def predict_model2():
    return make_prediction("model_2")


@app.route("/users", methods=["GET"])
def get_users():
    try:
        cursor = con.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users")
        users = cursor.fetchall()
        cursor.close()
        return jsonify(users), 200
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {err}"}), 500


@app.route("/delete_user/<int:user_id>", methods=["DELETE"])
def delete_user(user_id):
    try:
        cursor = con.cursor()
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        con.commit()
        cursor.close()
        return jsonify({"message": "User deleted successfully"}), 200
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {err}"}), 500


from flask import send_from_directory
import os

@app.route('/files/<filename>')
def serve_file(filename):
    # Get current directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(root_dir, filename)




def make_prediction(model_key):
    try:
        data = request.json
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request"}), 400

        input_data = data["features"]
        processed_data = preprocess_input(input_data, model_key)
        if processed_data is None:
            return jsonify({"error": "Data preprocessing failed. Check feature names and encoding."}), 400

        prediction = models[model_key].predict(processed_data)[0]

        # ⬇️ Add risk_ranges here ⬇️
        risk_ranges = {
            "RestingBP": lambda x: x > 130,
            "Cholesterol": lambda x: x > 240,
            "MaxHR": lambda x: x < 100,
            "Oldpeak": lambda x: x > 2.0,
            "Age_In_Days": lambda x: x > 55,
            "ExerciseAngina": lambda x: x == "Y" or x == 1,
            "ChestPainType": lambda x: x == "ASY" or x == 2,
            "ST_Slope": lambda x: x == "Flat" or x == 1,
            "Sex": lambda x: x == "M" or x == 1,
            "FastingBS": lambda x: x == 1,
            "BMI": lambda x: x > 25,
            "Smoking": lambda x: x == 1,
            "Alcohol_Consumption": lambda x: x in ["High", 2],
            "Physical_Activity_Level": lambda x: x in ["Low", 0]
        }

        # Evaluate top risk feature from user input
        importance = feature_importances[model_key].sort_values(ascending=False)
        top_risk = "Unknown"
        for feature in importance.index:
            if feature in input_data and feature in risk_ranges:
                val = input_data[feature]
                if risk_ranges[feature](val):
                    top_risk = feature
                    break

        risk_suggestion = generate_risk_shield(top_risk)

        return jsonify({
            "prediction": int(prediction),
            "top_risk_factor": top_risk if prediction == 1 else "N/A",
            "risk_suggestion": risk_suggestion if prediction == 1 else "N/A"
        })

    except Exception as e:
        print("❌ Prediction error:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    print("connecting to DB...")
    app.run(debug=True)