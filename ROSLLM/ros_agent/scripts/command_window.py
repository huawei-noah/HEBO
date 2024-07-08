from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['POST'])
def receive_string():
    if request.method == 'POST':
        # Receive the JSON data
        data = request.get_json()

        # Extract the string from the JSON data
        received_string = data.get('msg')

        # Prompt the user to input another string
        user_input_string = input(f"{received_string}\n")

        # Prepare the response JSON data
        response_data = {
            'received_string': received_string,
            'user_input_string': user_input_string
        }
        return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True, use_reloader=False)
