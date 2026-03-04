import requests

url = "http://127.0.0.1:5000/predict"
data = {"review": "I absolutely loved this movie!"}
response = requests.post(url, json=data)
print(response.json())
data2 = {"review": "This was the worst film ever."}
response2 = requests.post(url, json=data2)
print(response2.json())