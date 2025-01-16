from flask import Flask,request,render_template,jsonify
import pickle

#Loading the pickle file
with open('spam_clf_with_tfidf.pkl','rb') as file:
    data = pickle.load(file)
    model = data['model']
    tfidf_vectorizer = data['vectorizer']

#Creating the Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    Predict whether a message is spam or not
    '''

    try:
        #Extract the message from the request
        data = request.get_json() #Expecting a json request
        message = data.get('message','')

        #Validate input
        if not message:
            return jsonify({'error':'No message provided'}), 400
        
        #Preprocess the raw message using the Tf-Idf vectorizer
        transformed_message = tfidf_vectorizer.transform([message.strip()])

        #Perform prediction
        prediction = model.predict(transformed_message)[0]

        #Return the result
        result = 'spam' if prediction == 1 else 'ham'
        return jsonify({'message':message,'prediction':result})
    
    except Exception as e:
        return jsonify({'error':str(e)}), 500
    
if __name__ == '__main__':
    #Run the app
    app.run(debug=True)

