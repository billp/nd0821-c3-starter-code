"""Script to POST to the live deployed API and print the result."""
import requests

# Replace with your actual deployed app URL.
URL = "https://YOUR-APP-NAME.onrender.com/predict"


def main():
    sample = {
        "age": 52,
        "workclass": "Self-emp-inc",
        "fnlgt": 287927,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    response = requests.post(URL, json=sample)
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.json()}")


if __name__ == "__main__":
    main()
