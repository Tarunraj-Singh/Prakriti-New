# from flask import Flask, render_template, request, jsonify
# import pickle
# import numpy as np
# import pandas as pd
# from model import model 
# app = Flask(__name__)

# conversation_history = []  # To store user inputs

# # Define the column names that match your model
# columns = [
#     "Visual_features", "Tactile_features", "Joints", "Eyes", "Nails", "Teeth", "Mouth", 
#     "Palm_and_Sole", "Hair", "Voice_assessment", "Sleep_pattern", "Movement_and_Gait", 
#     "Diet_and_Lifestyle", "Excretory_products"
# ]

# @app.route('/')
# def man():
#     return render_template('chatbot.html') 

# @app.route("/process_user_input", methods=['POST'])
# def process_user_input():
#     user_input = request.json.get('userInput')
    
#     # Append the user input to the conversation history
#     conversation_history.append(user_input)
    
#     # Process the user input and generate a bot response
#     # You can use your machine learning model here
    
#     # Access and print the userChoices sent from JavaScript
#     user_choices = request.json.get('userChoices')
#     user_choices = np.array(user_choices).reshape(1, -1)
#     user_choices = np.array(user_choices)

#     # Load the model from model.pkl
#     # model = pickle.load(open('model.pkl', 'rb'))

#     # Create a DataFrame using user choices and the column names
#     # user_choices_df = pd.DataFrame(user_choices, columns=columns)

#     # Make predictions using your model
#     predictions = model.predict(user_choices)
    
#     # Provide the predicted Prakriti as a bot response
#     prakriti_mapping = {0: 'Vata', 1: 'Pitta', 2: 'Kapha'}  # Modify as per your label mapping
#     predicted_prakriti = prakriti_mapping.get(predictions[0], 'Unknown')  # Get the Prakriti label based on prediction
    
#     bot_response = f"Your Prakriti is {predicted_prakriti}."
#     print(bot_response)
#     return jsonify({'botResponse': bot_response})

# if __name__ == '__main__':
#     app.run(debug=True)
#---------------------------------------------------------------------------------------------------------------------
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import json
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

conversation_history = []  # To store user inputs

# Define the column names that match your model
columns = [
    "Visual_features", "Tactile_features", "Joints", "Eyes", "Nails", "Teeth", "Mouth", 
    "Palm_and_Sole", "Hair", "Voice_assessment", "Sleep_pattern", "Movement_and_Gait", 
    "Diet_and_Lifestyle", "Excretory_products"
]

# Load the label encoder state from JSON file
with open('label_encoder_state.json', 'r') as le_file:
    label_encoder_classes = json.load(le_file)

label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes

# Load the model's architecture and weights
model = XGBClassifier()
model.load_model('model.h5')

@app.route('/')
def man():
    return render_template('chatbot.html') 

@app.route("/process_user_input", methods=['POST'])
def process_user_input():
    user_choices = request.json.get('userChoices')

    if not user_choices:
        # Handle the case when no user choices are provided
        return jsonify({'botResponse': 'Please provide valid input.'})

    # Encode user choices using the label encoder
    encoded_choices = label_encoder.transform(user_choices)
    user_choices_encoded = np.array(encoded_choices).reshape(1, -1)

    # Make predictions using your model
    predictions = model.predict(user_choices_encoded)

    # Provide the predicted Prakriti as a bot response
    prakriti_mapping = {0: 'Vata', 1: 'Pitta', 2: 'Kapha'}  # Modify as per your label mapping
    predicted_prakriti = prakriti_mapping.get(predictions[0], 'Unknown')  # Get the Prakriti label based on prediction
    
    bot_response = f"Your Prakriti is {predicted_prakriti}."
    print(bot_response)
    return jsonify({'botResponse': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
