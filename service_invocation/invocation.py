import requests

url = "https://servicio-api-first-deploy-1095600528292.us-east1.run.app/recibir_data_&_predecir"

file_path = "final_test.csv"

headers = {
    "Accept": "aplication/json"
}

try:

    with open(file_path, "rb") as file:
        files = {
            "file": ("test.csv", file, "text/csv")
        }
        response = requests.post(url, files = files, headers = headers)

    if response.status_code == 200:

        with open("resultado_api.csv", "wb") as f:
            f.write(response.content)
        print("Archivo CSV recibido y guardado como resultado_api.csv")

    else:

        print(f"Error al invocar la API. Código de estado: {response.status_code}")
        print("Detalles del error:", response.text)
    
except FileNotFoundError:
    print(f"Error: El archivo {file_path} no se encontró, por favor verifica la ruta")
except requests.exceptions.RequestException as e:
    print(f"Error en la solicitud HTTP: {e}")