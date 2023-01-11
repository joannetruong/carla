import carla

client = carla.Client(host='127.0.0.1', port=2000)
server_version = client.get_server_version()
client_version = client.get_client_version()
