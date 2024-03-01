import requests 

res = requests.post("http://localhost:5000/predict", files={'file':open('mnist.png', 'rb')})
print(res.text)