import requests

# Replace the URL with the actual URL where your Flask app is running
BASE_URL = 'http://127.0.0.1:5002'

def test_home():
    response = requests.get(BASE_URL + '/')
    assert response.status_code == 200

def test_index_valid_input():
    data = {
        'location': 'Whitefield',
        'bhk': '3',
        'bath': '2',
        'totalsqft': '1500'
    }
    response = requests.post(BASE_URL + '/index', data=data)
    assert response.status_code == 200
    assert "Predicted Price" in response.text

def test_index_missing_values():
    data = {
        'bhk': '2',
        'totalsqft': '1200'
    }
    response = requests.post(BASE_URL + '/index', data=data)
    assert response.status_code == 200
    assert "Please fill in all fields" in response.text

def test_index_non_numeric_values():
    data = {
        'location': 'Electronic City',
        'bhk': 'Two',
        'bath': '2',
        'totalsqft': '1300'
    }
    response = requests.post(BASE_URL + '/index', data=data)
    assert response.status_code == 200
    assert "Invalid input values" in response.text

def test_predict_valid_input():
    data = {
        'location': 'Marathahalli',
        'bhk': '3',
        'bath': '2',
        'totalsqft': '1500'
    }
    response = requests.post(BASE_URL + '/predict', data=data)
    assert response.status_code == 200
    assert "Predicted Price" in response.text

def test_predict_invalid_input():
    data = {
        'location': 'Marathahalli',
        'bhk': 'Three',  # Invalid value
        'bath': '2',
        'totalsqft': '1500'
    }
    response = requests.post(BASE_URL + '/predict', data=data)
    assert response.status_code == 200
    assert "Invalid input values" in response.text

if __name__ == "__main__":
    test_home()
    test_index_valid_input()
    test_index_missing_values()
    test_index_non_numeric_values()
    test_predict_valid_input()
    test_predict_invalid_input()

    print("All tests passed!")
