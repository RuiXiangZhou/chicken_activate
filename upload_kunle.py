import requests

url = "http://sfems.nchu.kunlex.com.tw/api/iot_device/rawdata.php"

payload = 'kun_sn=kun-MPtptyxA&logtime=2023-06-19%2015%3A18%3A09&Activity=12&iottype=chicken_activity'
headers = {
  'Content-Type': 'application/x-www-form-urlencoded'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)