import requests


class RosApi:

      # Defined on ros side
    default_timeout = 3 * 60  # default timeout is 3 minutes

    def __init__(self, timeout=None):
        self.timeout = self.default_timeout if timeout is None else timeout
        self._response = None

    @staticmethod
    def bad_response(obs):
        return {
            "success": False,
            "done": True,
            "reward": 0,
            "obs": obs,
        }

    def send_action(self, action):
        url = "http://localhost:5000/llmreq"
        self._response = None  # reset response
        try:
            data = {"action": action}
            resp = requests.post(url, json=data, timeout=self.timeout)
            response = resp.json()
        except requests.exceptions.Timeout:
            response = self.bad_response("Request timeout.")
        except requests.exceptions.RequestException as e:
            response = self.bad_response(f"Request exception: {e}")
        self._response = response

    def get_env_observation(self):
        url = "http://localhost:5000/rosenv"
        self._response = None  # reset response
        try:
            data = {"": ""}
            resp = requests.post(url, json=data, timeout=self.timeout)
            response = resp.json()
        except requests.exceptions.Timeout:
            response = self.bad_response("Request timeout.")
        except requests.exceptions.RequestException as e:
            response = self.bad_response(f"Request exception: {e}")
        self._response = response
        
        return response
    
    def get_feedback(self):
        url = "http://localhost:5000/rosfdb"
        self._response = None  # reset response
        try:
            data = {"": ""}
            resp = requests.post(url, json=data, timeout=self.timeout)
            response = resp.json()
        except requests.exceptions.Timeout:
            response = self.bad_response("Request timeout.")
        except requests.exceptions.RequestException as e:
            response = self.bad_response(f"Request exception: {e}")
        self._response = response
        
        return response

    def receive_response(self):
        assert self._response is not None, "did not receive a response"
        return self._response
